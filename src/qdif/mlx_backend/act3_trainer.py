"""Act III trainer: AR + diffusion transfer with a health gate.

Logs `loss_ar`, `loss_diff` and `loss_total` separately, plus the health metrics that
make the Act II Phase 3 pathology (canvas-ignoring) impossible to mistake for
progress. Checkpoints and held-out evaluations are frequent enough to plot the
transition from autoregressive model to dual-mode AR/diffusion model.
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
from ..data.datasets import build_packed_dataset
from .act3 import (
    act3_loss,
    ar_reference_perplexity,
    canvas_conditioning,
    health_metrics,
    mask_corrupt,
    next_token_accuracy,
    resolve_mask_token_id,
)
from .build import build_mlx_setup
from .ops import cross_entropy

EVAL_SEED = 777
HEALTH_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90, 1.00)


@dataclass
class Act3Setup:
    setup: object
    train_ds: object
    eval_ds: object
    eval_batches: list
    ar_reference: mx.array
    mask_token_id: int
    notes: list[str] = field(default_factory=list)


def _batches(dataset, batch_size: int, num_batches: int):
    out = []
    for i in range(0, min(num_batches * batch_size, len(dataset)), batch_size):
        ex = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        if len(ex) < batch_size:
            break
        out.append(
            (np.stack([e.canvas_ids for e in ex]), np.stack([e.prefix_ids for e in ex]))
        )
    if not out:
        raise ValueError("dataset produced no full batches")
    return out


def prepare(cfg: ExperimentConfig, echo=print, eval_batches: int = 8) -> Act3Setup:
    setup = build_mlx_setup(cfg, echo=echo)
    tok = setup.tokenizer

    mask_id = resolve_mask_token_id(
        setup.load_report.vocab_size, tok.vocab_size, cfg.diffusion.mask_token_id
    )
    echo(
        f"[act3] absorbing mask token id {mask_id} "
        f"(model vocab {setup.load_report.vocab_size}, tokenizer vocab {tok.vocab_size} "
        f"-> unreachable by the tokenizer, untrained embedding row)"
    )

    eos = getattr(tok, "eos_token_id", None)
    train_ds = build_packed_dataset(
        cfg.data, tok, cfg.diffusion.canvas_length, cfg.data.hf_split,
        cfg.data.max_examples, eos_id=eos,
    )
    eval_ds = build_packed_dataset(
        cfg.data, tok, cfg.diffusion.canvas_length, cfg.data.eval_split,
        cfg.data.eval_max_examples, eos_id=eos,
    )
    echo(
        f"[data] train {len(train_ds)} windows / {train_ds.total_tokens:,} tokens | "
        f"held-out {len(eval_ds)} windows / {eval_ds.total_tokens:,} tokens "
        f"(split '{cfg.data.eval_split}')"
    )

    batches = _batches(eval_ds, cfg.training.batch_size, eval_batches)
    # Fixed AR reference set: the first two held-out batches, reassembled into full
    # [prefix | canvas] sequences. Held constant for every evaluation so AR health is
    # comparable across steps and across runs.
    ar_ref = mx.array(
        np.stack(
            [
                np.concatenate([prefix[j], canvas[j]])
                for canvas, prefix in batches[: min(2, len(batches))]
                for j in range(canvas.shape[0])
            ]
        )
    )
    return Act3Setup(
        setup=setup, train_ds=train_ds, eval_ds=eval_ds, eval_batches=batches,
        ar_reference=ar_ref, mask_token_id=mask_id,
    )


def evaluate_health(a3: Act3Setup, cfg: ExperimentConfig, step: int) -> list[dict]:
    """Held-out health evaluation at every noise level."""
    model = a3.setup.model
    model.eval()
    rows = []
    vocab = a3.setup.load_report.vocab_size

    for t_val in HEALTH_LEVELS:
        rng = np.random.default_rng(EVAL_SEED)
        agg: dict[str, list[float]] = {}
        for x0, prefix in a3.eval_batches:
            canvas = mask_corrupt(
                x0, np.full(x0.shape[0], t_val, np.float32), a3.mask_token_id, rng,
                exact_count=cfg.diffusion.exact_mask_count,
            )
            pre = mx.array(prefix)
            out = model(canvas_ids=mx.array(canvas.xt), t=mx.array(canvas.t), prefix_ids=pre)
            mx.eval(out.logits)
            h = health_metrics(out.logits, canvas, a3.mask_token_id)
            sel = mx.array(canvas.mask_set.astype(np.float32))
            from .act3 import masked_positions_ce

            h["diffusion_ce"] = float(masked_positions_ce(out.logits, mx.array(canvas.x0), sel))
            h["full_canvas_ce"] = float(cross_entropy(out.logits, mx.array(canvas.x0)))
            cc = canvas_conditioning(model, pre, canvas, np.random.default_rng(EVAL_SEED + 1), vocab)
            h.update(cc)
            for k, v in h.items():
                agg.setdefault(k, []).append(v)
        row = {"step": step, "t": float(t_val)}
        row.update({k: float(np.nanmean(v)) for k, v in agg.items()})
        rows.append(row)

    ids = a3.ar_reference
    rows.append(
        {
            "step": step, "t": -1.0, "kind": "ar",
            "ar_reference_perplexity": ar_reference_perplexity(model, ids),
            "next_token_accuracy": next_token_accuracy(model, ids),
        }
    )
    model.train()
    return rows


def _fmt(rows) -> str:
    parts = []
    for r in rows:
        if r.get("kind") == "ar":
            parts.append(f"AR ppl {r['ar_reference_perplexity']:7.2f} nt {r['next_token_accuracy']:5.1%}")
        elif r["t"] in (0.10, 0.50, 0.90):
            parts.append(
                f"t={r['t']:.2f} m {r['masked_accuracy']:5.1%} v {r['visible_preservation']:5.1%} "
                f"cc {r['canvas_l1']:.3f}"
            )
    return " | ".join(parts)


def train_act3(cfg: ExperimentConfig, echo=print, dry_run: bool = False) -> dict:
    a3 = prepare(cfg, echo=echo)
    setup = a3.setup
    model = setup.model
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("health.jsonl", "train.jsonl"):
        p = run_dir / name
        if p.exists():
            p.unlink()

    from ..provenance import write_run_metadata

    write_run_metadata(
        run_dir, cfg,
        extra={"act3": {"mask_token_id": a3.mask_token_id,
                        "train_tokens": a3.train_ds.total_tokens,
                        "heldout_tokens": a3.eval_ds.total_tokens}},
    )
    (run_dir / "setup.json").write_text(
        json.dumps(
            {
                **setup.summary(),
                "act3": {
                    "mask_token_id": a3.mask_token_id,
                    "lambda_diff": cfg.diffusion.lambda_diff,
                    "complementary_views": cfg.diffusion.complementary_views,
                    "train_tokens": a3.train_ds.total_tokens,
                    "heldout_tokens": a3.eval_ds.total_tokens,
                    "dataset": cfg.data.hf_dataset,
                },
            },
            indent=2, default=str,
        )
    )

    if dry_run:
        return {"dry_run": True, "setup": setup.summary(), "mask_token_id": a3.mask_token_id}

    rng = np.random.default_rng(cfg.training.seed)
    B = cfg.training.batch_size
    lr = cfg.training.learning_rate
    warmup = max(cfg.training.warmup_steps, 0)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=cfg.training.weight_decay)

    def loss_fn(m, prefix, canvas_obj):
        out = act3_loss(
            m, prefix, canvas_obj,
            lambda_diff=cfg.diffusion.lambda_diff,
            complementary_views=cfg.diffusion.complementary_views,
            ar_weight=cfg.diffusion.ar_weight,
        )
        return out.total, (out.loss_ar, out.loss_diff)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def do_eval(step):
        rows = evaluate_health(a3, cfg, step)
        with (run_dir / "health.jsonl").open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        echo(f"  [health @ {step:>5}] {_fmt(rows)}")
        return rows

    do_eval(0)
    history = []
    t_all = time.time()

    for step in range(1, cfg.training.max_steps + 1):
        t0 = time.time()
        idx = rng.integers(0, len(a3.train_ds), size=B)
        ex = [a3.train_ds[int(i)] for i in idx]
        x0 = np.stack([e.canvas_ids for e in ex])
        prefix_np = np.stack([e.prefix_ids for e in ex])
        t_vec = rng.uniform(cfg.diffusion.t_min, cfg.diffusion.t_max, size=B).astype(np.float32)
        canvas = mask_corrupt(
            x0, t_vec, a3.mask_token_id, rng, exact_count=cfg.diffusion.exact_mask_count
        )

        optimizer.learning_rate = lr * min(1.0, step / warmup) if warmup else lr
        (total, (l_ar, l_diff)), grads = loss_and_grad(model, mx.array(prefix_np), canvas)
        gnorm = 0.0
        if cfg.training.max_grad_norm > 0:
            grads, gn = optim.clip_grad_norm(grads, cfg.training.max_grad_norm)
            gnorm = float(gn)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, total)
        dt = time.time() - t0

        rec = {
            "step": step,
            "loss_total": float(total),
            "loss_ar": float(l_ar),
            "loss_diff": float(l_diff),
            "grad_norm": gnorm,
            "t_mean": float(canvas.t.mean()),
            "actual_masked_fraction": canvas.masked_fraction,
            "num_masked": canvas.num_masked,
            "num_visible": canvas.num_visible,
            "seconds": dt,
            "tokens_per_sec": B * cfg.diffusion.canvas_length / dt,
            "peak_unified_gb": mx.get_peak_memory() / 1e9,
        }
        history.append(rec)
        with (run_dir / "train.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if step % cfg.training.log_every == 0:
            echo(
                f"step {step:>5} | total {rec['loss_total']:7.4f} | ar {rec['loss_ar']:7.4f} "
                f"| diff {rec['loss_diff']:7.4f} | mask {rec['actual_masked_fraction']:5.1%} "
                f"| g {gnorm:6.2f} | {rec['tokens_per_sec']:6.1f} tok/s "
                f"| {rec['peak_unified_gb']:.2f} GB"
            )
        if cfg.training.save_every and step % cfg.training.save_every == 0:
            do_eval(step)
            from .trainer import save_adapters

            save_adapters(model, cfg, run_dir / f"ckpt-{step}", step=step)

    final_rows = do_eval(cfg.training.max_steps)
    from .trainer import save_adapters

    ckpt = save_adapters(model, cfg, run_dir / "checkpoint", step=cfg.training.max_steps)

    summary = {
        "steps": cfg.training.max_steps,
        "batch_size": B,
        "canvas_length": cfg.diffusion.canvas_length,
        "prefix_length": cfg.data.prefix_length,
        "mask_token_id": a3.mask_token_id,
        "lambda_diff": cfg.diffusion.lambda_diff,
        "complementary_views": cfg.diffusion.complementary_views,
        "train_tokens": a3.train_ds.total_tokens,
        "first_loss_total": history[0]["loss_total"],
        "final_loss_total": history[-1]["loss_total"],
        "first_loss_ar": history[0]["loss_ar"],
        "final_loss_ar": history[-1]["loss_ar"],
        "first_loss_diff": history[0]["loss_diff"],
        "final_loss_diff": history[-1]["loss_diff"],
        "mean_seconds_per_step": float(np.mean([h["seconds"] for h in history])),
        "mean_tokens_per_sec": float(np.mean([h["tokens_per_sec"] for h in history])),
        "peak_unified_gb": max(h["peak_unified_gb"] for h in history),
        "wall_seconds": time.time() - t_all,
        "final_health": {f"t={r['t']}": r for r in final_rows},
        "params": setup.params,
        "checkpoint": str(ckpt),
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
