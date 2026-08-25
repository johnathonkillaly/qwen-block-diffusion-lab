"""Phase 3: held-out denoising, tracked over training.

This is the first scientifically meaningful v0.2 experiment. Everything before it was
plumbing (smoke test) or memorisation (one-batch overfit, which SATURATES at 4B and
cannot discriminate the arms).

DESIGN
------
Four arms, identical in every respect except the DeltaNet mechanism:

    A  causal DeltaNet + bidirectional periodic full attention        (v0.1)
    B  aligned forward/reverse shared-weight fusion                   (v0.2 hypothesis)
    C  shuffled-reverse negative control                              (position alignment destroyed)
    D  FLARE-style block-end recurrent-state readout                  (arXiv:2606.01774v2)

Same checkpoint, same LoRA config, same steps, same seeds, same training data, same
held-out data, same corruption draws at evaluation.

The held-out split is **document-disjoint**: wikitext-2's `test` split is a different
set of articles from `train`, not merely different windows of the same articles.

THE MEASUREMENT THE HYPOTHESIS LIVES OR DIES ON
-----------------------------------------------
    Delta(t) = corrupted_accuracy_aligned(t) - corrupted_accuracy_shuffled(t)

evaluated at t in {0.50, 0.75, 0.90} at every checkpoint during training. At
initialisation this was measured to be <= 0 (the shuffled control was *better* -- see
docs/BIDIRECTIONAL_DELTANET.md). If Delta(t) starts near zero and becomes positive
during training, the model has learned to exploit the *positional structure* of the
reverse recurrence rather than merely benefiting from an extra perturbation branch.
If it stays flat or negative, the second recurrence pass is a perturbation, not
information, and the pre-registered kill criterion fires.

A NOTE ON t = 1.0
-----------------
Cross entropy at t = 1.0 is retained, but exact reconstruction there is NOT treated as
equivalent to reconstruction at partial noise. At full corruption the canvas carries
no information about which continuation was the true one, so the particular held-out
continuation is not identifiable; a model can only produce *a* plausible continuation.
Loss at t = 1.0 measures the prior, not denoising.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from ..config import ExperimentConfig
from ..data.datasets import build_dataset
from .build import build_mlx_setup
from .ops import corrupt_canvas, cross_entropy, diffusion_metrics

#: Fixed so every arm and every checkpoint sees identical held-out corruption.
EVAL_SEED = 777
TRACK_LEVELS = (0.50, 0.75, 0.90)
FULL_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90, 1.00)


@dataclass
class EvalPoint:
    step: int
    t: float
    loss: float
    corrupted_accuracy: float
    identity_accuracy: float
    copy_baseline_accuracy: float
    lift_over_copy: float
    copy_rate: float
    exact_reconstruction: float
    next_token_accuracy: float

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)


@dataclass
class HeldOutResult:
    arm: str
    points: list[EvalPoint] = field(default_factory=list)

    def at(self, step: int, t: float) -> EvalPoint | None:
        for p in self.points:
            if p.step == step and abs(p.t - t) < 1e-9:
                return p
        return None


def _fixed_batches(dataset, batch_size: int, num_batches: int):
    """Deterministic held-out batches: the first N windows, in order."""
    out = []
    for i in range(0, min(num_batches * batch_size, len(dataset)), batch_size):
        ex = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        if len(ex) < batch_size:
            break
        out.append(
            (
                np.stack([e.canvas_ids for e in ex]),
                np.stack([e.prefix_ids for e in ex]),
            )
        )
    if not out:
        raise ValueError("held-out set produced no full batches")
    return out


def evaluate(setup, batches, vocab, levels, step: int, cfg) -> list[EvalPoint]:
    """Held-out evaluation at fixed noise levels, identical corruption for every arm."""
    model = setup.model
    model.eval()
    points: list[EvalPoint] = []

    for t_val in levels:
        rng = np.random.default_rng(EVAL_SEED)  # reset per level -> reproducible
        agg = {
            "loss": [], "corrupted_accuracy": [], "identity_accuracy": [],
            "copy_baseline_accuracy": [], "lift_over_copy": [], "copy_rate": [],
            "exact": [], "next_token_accuracy": [],
        }
        for x0, prefix in batches:
            res = corrupt_canvas(
                x0, np.full(x0.shape[0], t_val, np.float32), vocab, rng,
                mode=cfg.diffusion.corruption, mask_token_id=cfg.diffusion.mask_token_id,
            )
            out = model(
                canvas_ids=mx.array(res.xt), t=mx.array(res.t), prefix_ids=mx.array(prefix)
            )
            mx.eval(out.logits)
            agg["loss"].append(float(cross_entropy(out.logits, mx.array(res.x0))))
            m = diffusion_metrics(out.logits, res.x0, res.xt, res.corrupted_mask)
            for k in ("corrupted_accuracy", "identity_accuracy", "copy_baseline_accuracy",
                      "lift_over_copy", "copy_rate", "next_token_accuracy"):
                agg[k].append(m[k])
            pred = np.asarray(mx.argmax(out.logits, axis=-1).astype(mx.int32))
            agg["exact"].append(float((pred == res.x0).all(axis=-1).mean()))

        points.append(
            EvalPoint(
                step=step,
                t=float(t_val),
                loss=float(np.mean(agg["loss"])),
                corrupted_accuracy=float(np.nanmean(agg["corrupted_accuracy"])),
                identity_accuracy=float(np.mean(agg["identity_accuracy"])),
                copy_baseline_accuracy=float(np.mean(agg["copy_baseline_accuracy"])),
                lift_over_copy=float(np.mean(agg["lift_over_copy"])),
                copy_rate=float(np.mean(agg["copy_rate"])),
                exact_reconstruction=float(np.mean(agg["exact"])),
                next_token_accuracy=float(np.nanmean(agg["next_token_accuracy"])),
            )
        )
    model.train()
    return points


def gate_values(setup) -> dict:
    """Read the learned fusion gates. Converging back to g >= 0.9 is a kill criterion."""
    from .deltanet import ScalarGateFusion, TokenGateFusion, deltanet_modules

    out: dict[str, float] = {}
    scalars, tokens = [], []
    for i, mod in deltanet_modules(setup.model.base_model):
        f = mod.fusion
        if isinstance(f, ScalarGateFusion):
            scalars.append(float(mx.sigmoid(f.alpha)[0]))
        elif isinstance(f, TokenGateFusion):
            tokens.append(float(mx.sigmoid(f.b_g).mean()))
    if scalars:
        out.update(
            gate_mean=float(np.mean(scalars)), gate_min=float(np.min(scalars)),
            gate_max=float(np.max(scalars)), gate_kind="scalar",
        )
    if tokens:
        out.update(
            gate_mean=float(np.mean(tokens)), gate_min=float(np.min(tokens)),
            gate_max=float(np.max(tokens)), gate_kind="token_bias",
        )
    return out


def run_arm(cfg: ExperimentConfig, arm: str, echo=print) -> dict:
    """Train one arm and evaluate held-out throughout."""
    setup = build_mlx_setup(cfg, echo=echo)
    model, tokenizer = setup.model, setup.tokenizer
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset(
        cfg.data, tokenizer, cfg.diffusion.canvas_length,
        split=cfg.data.hf_split, max_examples=cfg.data.max_examples,
    )
    eval_ds = build_dataset(
        cfg.data, tokenizer, cfg.diffusion.canvas_length,
        split=cfg.data.eval_split, max_examples=cfg.data.eval_max_examples,
    )
    echo(f"[data] train windows {len(train_ds)} | held-out windows {len(eval_ds)} "
         f"(split '{cfg.data.eval_split}', document-disjoint)")

    eval_batches = _fixed_batches(eval_ds, cfg.training.batch_size, num_batches=16)
    vocab = setup.load_report.vocab_size
    rng = np.random.default_rng(cfg.training.seed)

    optimizer = optim.AdamW(
        learning_rate=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )

    def loss_fn(m, xt, t, prefix, target):
        return cross_entropy(m(canvas_ids=xt, t=t, prefix_ids=prefix).logits, target)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    curve_path = run_dir / "heldout_curve.jsonl"
    if curve_path.exists():
        curve_path.unlink()
    (run_dir / "setup.json").write_text(json.dumps(setup.summary(), indent=2, default=str))

    eval_every = max(cfg.training.save_every or 50, 1)
    result = HeldOutResult(arm=arm)
    train_log: list[dict] = []
    t_start_all = time.time()

    def do_eval(step: int, levels):
        pts = evaluate(setup, eval_batches, vocab, levels, step, cfg)
        result.points.extend(pts)
        gates = gate_values(setup)
        with curve_path.open("a") as f:
            for p in pts:
                f.write(json.dumps({"arm": arm, **p.to_dict(), **gates}) + "\n")
        line = " | ".join(
            f"t={p.t:.2f} corr {p.corrupted_accuracy:6.2%} lift {p.lift_over_copy:+6.2%} "
            f"ce {p.loss:6.3f}"
            for p in pts
        )
        g = f" | gate {gates['gate_mean']:.3f}" if "gate_mean" in gates else ""
        echo(f"  [eval @ {step:>4}] {line}{g}")

    do_eval(0, TRACK_LEVELS)

    warmup = max(cfg.training.warmup_steps, 0)
    lr = cfg.training.learning_rate
    B = cfg.training.batch_size

    for step in range(1, cfg.training.max_steps + 1):
        t0 = time.time()
        idx = rng.integers(0, len(train_ds), size=B)
        examples = [train_ds[int(i)] for i in idx]
        x0 = np.stack([e.canvas_ids for e in examples])
        prefix_np = np.stack([e.prefix_ids for e in examples])
        t_vec = rng.uniform(cfg.diffusion.t_min, cfg.diffusion.t_max, size=B).astype(np.float32)
        res = corrupt_canvas(
            x0, t_vec, vocab, rng, mode=cfg.diffusion.corruption,
            mask_token_id=cfg.diffusion.mask_token_id,
        )

        optimizer.learning_rate = lr * min(1.0, step / warmup) if warmup else lr
        loss, grads = loss_and_grad(
            model, mx.array(res.xt), mx.array(res.t), mx.array(prefix_np), mx.array(res.x0)
        )
        if cfg.training.max_grad_norm > 0:
            grads = optim.clip_grad_norm(grads, cfg.training.max_grad_norm)[0]
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        elapsed = time.time() - t0

        train_log.append(
            {"step": step, "loss": float(loss), "t_mean": float(res.t.mean()),
             "seconds": elapsed, "peak_unified_gb": mx.get_peak_memory() / 1e9}
        )
        if step % cfg.training.log_every == 0:
            echo(f"step {step:>4} | loss {float(loss):7.4f} | t {float(res.t.mean()):.3f} "
                 f"| {B * cfg.diffusion.canvas_length / elapsed:6.1f} tok/s "
                 f"| {mx.get_peak_memory() / 1e9:.2f} GB unified")
        if step % eval_every == 0 and step < cfg.training.max_steps:
            do_eval(step, TRACK_LEVELS)

    do_eval(cfg.training.max_steps, FULL_LEVELS)

    from .trainer import save_adapters

    ckpt = save_adapters(model, cfg, run_dir / "checkpoint", step=cfg.training.max_steps)

    summary = {
        "arm": arm,
        "mode": cfg.bidirectional_deltanet.mode,
        "bidirectional": cfg.bidirectional_deltanet.enabled,
        "fusion": cfg.bidirectional_deltanet.fusion,
        "steps": cfg.training.max_steps,
        "batch_size": B,
        "train_windows": len(train_ds),
        "heldout_windows": len(eval_ds),
        "heldout_batches": len(eval_batches),
        "first_loss": train_log[0]["loss"] if train_log else None,
        "final_loss": train_log[-1]["loss"] if train_log else None,
        "mean_seconds_per_step": float(np.mean([r["seconds"] for r in train_log])),
        "tokens_per_sec": float(
            B * cfg.diffusion.canvas_length / np.mean([r["seconds"] for r in train_log])
        ),
        "peak_unified_gb": max(r["peak_unified_gb"] for r in train_log),
        "wall_seconds": time.time() - t_start_all,
        "gates": gate_values(setup),
        "final": {f"t={p.t}": p.to_dict() for p in result.points if p.step == cfg.training.max_steps},
        "params": setup.params,
        "checkpoint": str(ckpt),
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "train_log.json").write_text(json.dumps(train_log, indent=2))
    return summary
