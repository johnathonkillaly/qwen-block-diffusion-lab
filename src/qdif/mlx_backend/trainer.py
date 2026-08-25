"""MLX trainer for the diffusion objective.

Uses `mlx.optimizers` directly rather than `unsloth_zoo.mlx.trainer`, for the reason
documented in docs/UNSLOTH_BACKEND.md: Unsloth's MLX trainer is an SFT trainer whose
loss is next-token cross entropy over a chat-formatted batch. Our loss is
x0-prediction over a corrupted canvas with a bidirectional mask, which its batch
format cannot express. What we *do* keep from Unsloth is the part that matters for
this architecture: the Qwen3.5 GatedDeltaNet custom VJP, applied at load time.

Diagnostics are the v0.1 set, unchanged and deliberately so -- copy collapse and
mode collapse are the failure modes that made v0.1 interesting, and v0.2 must be
judged by the same yardstick.
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


@dataclass
class StepRecord:
    step: int
    loss: float
    learning_rate: float
    t_mean: float
    corruption_fraction: float
    changed_fraction: float
    grad_norm: float
    trainable_norm: float
    seconds: float
    tokens_per_sec: float
    examples_per_sec: float
    peak_unified_gb: float
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"step {self.step:>4d} | loss {self.loss:8.4f} | lr {self.learning_rate:.2e} "
            f"| t {self.t_mean:.3f} | corrupt {self.corruption_fraction:5.1%} "
            f"| id-acc {m.get('identity_accuracy', float('nan')):5.1%} "
            f"| corr-acc {m.get('corrupted_accuracy', float('nan')):5.1%} "
            f"| lift {m.get('lift_over_copy', float('nan')):+6.1%} "
            f"| copy {m.get('copy_rate', float('nan')):5.1%} "
            f"| next-tok {m.get('next_token_accuracy', float('nan')):5.1%} "
            f"| gnorm {self.grad_norm:7.3f} | {self.tokens_per_sec:6.1f} tok/s "
            f"| {self.peak_unified_gb:.2f} GB unified"
        )


def _global_norm(tree) -> float:
    from mlx.utils import tree_flatten

    total = 0.0
    for _, v in tree_flatten(tree):
        total += float(mx.sum(v.astype(mx.float32) ** 2))
    return total**0.5


def _tail_mean(history: list[StepRecord], key: str, fraction: float = 0.25) -> float | None:
    if not history:
        return None
    tail = history[-max(1, int(len(history) * fraction)) :]
    vals = [h.metrics[key] for h in tail if key in h.metrics and h.metrics[key] == h.metrics[key]]
    return sum(vals) / len(vals) if vals else None


def format_reconstruction(tokenizer, x0, xt, pred, prefix=None, step=None, t=None, max_chars=400):
    def dec(ids):
        text = tokenizer.decode([int(i) for i in ids]).replace("\n", "\\n")
        return text[:max_chars] + ("..." if len(text) > max_chars else "")

    header = "--- reconstruction"
    if step is not None:
        header += f" @ step {step}"
    if t is not None:
        header += f" (t = {t:.3f})"
    lines = [header + " ---"]
    if prefix is not None and len(prefix):
        lines += ["PREFIX:", dec(prefix), ""]
    matched = int((np.asarray(pred) == np.asarray(x0)).sum())
    lines += [
        "CLEAN:", dec(x0), "",
        "NOISY:", dec(xt), "",
        "PREDICTED:", dec(pred), "",
        f"exact-token match: {matched}/{len(x0)} ({matched / len(x0):.1%})", "",
    ]
    return "\n".join(lines)


def train(cfg: ExperimentConfig, echo=print, fixed_batch: bool = False) -> dict:
    setup = build_mlx_setup(cfg, echo=echo)
    model, tokenizer = setup.model, setup.tokenizer
    model.train()

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    samples_path = run_dir / "reconstructions.txt"
    for p in (metrics_path, samples_path):
        if p.exists():
            p.unlink()

    dataset = build_dataset(cfg.data, tokenizer, cfg.diffusion.canvas_length)
    echo(f"[setup] dataset: {len(dataset)} examples of "
         f"{cfg.data.prefix_length}+{cfg.diffusion.canvas_length} tokens")

    vocab = setup.load_report.vocab_size
    rng = np.random.default_rng(cfg.training.seed)

    B = cfg.training.batch_size
    # The overfit batch is the first B rows -- deterministic, so the run reproduces
    # and evaluation can target exactly the examples that were memorised (v0.1 fix).
    frozen = [dataset[i] for i in range(B)] if fixed_batch else None

    warmup = max(cfg.training.warmup_steps, 0)
    lr = cfg.training.learning_rate
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=cfg.training.weight_decay)

    def loss_fn(m, xt, t, prefix, target):
        return cross_entropy(m(canvas_ids=xt, t=t, prefix_ids=prefix).logits, target)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    (run_dir / "setup.json").write_text(json.dumps(setup.summary(), indent=2, default=str))

    history: list[StepRecord] = []
    for step in range(cfg.training.max_steps):
        t_start = time.time()
        examples = frozen if fixed_batch else [
            dataset[int(i)] for i in rng.integers(0, len(dataset), size=B)
        ]
        x0 = np.stack([np.asarray(e.canvas_ids) for e in examples])
        prefix_np = np.stack([np.asarray(e.prefix_ids) for e in examples])

        t_vec = rng.uniform(cfg.diffusion.t_min, cfg.diffusion.t_max, size=B).astype(np.float32)
        res = corrupt_canvas(
            x0, t_vec, vocab, rng, mode=cfg.diffusion.corruption,
            mask_token_id=cfg.diffusion.mask_token_id,
        )

        xt = mx.array(res.xt)
        target = mx.array(res.x0)
        prefix = mx.array(prefix_np)
        t = mx.array(res.t)

        cur_lr = lr * min(1.0, (step + 1) / warmup) if warmup else lr
        optimizer.learning_rate = cur_lr

        loss, grads = loss_and_grad(model, xt, t, prefix, target)
        gnorm = _global_norm(grads)
        if cfg.training.max_grad_norm > 0:
            grads = optim.clip_grad_norm(grads, cfg.training.max_grad_norm)[0]
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        elapsed = time.time() - t_start
        out = model(canvas_ids=xt, t=t, prefix_ids=prefix)
        mx.eval(out.logits)
        metrics = diffusion_metrics(out.logits, res.x0, res.xt, res.corrupted_mask)

        record = StepRecord(
            step=step,
            loss=float(loss),
            learning_rate=cur_lr,
            t_mean=float(res.t.mean()),
            corruption_fraction=float(res.corrupted_mask.mean()),
            changed_fraction=float((res.xt != res.x0).mean()),
            grad_norm=gnorm,
            trainable_norm=_global_norm(model.trainable_parameters()),
            seconds=elapsed,
            tokens_per_sec=B * cfg.diffusion.canvas_length / elapsed,
            examples_per_sec=B / elapsed,
            peak_unified_gb=mx.get_peak_memory() / 1e9,
            metrics=metrics,
        )
        history.append(record)
        with metrics_path.open("a") as f:
            f.write(json.dumps(record.__dict__) + "\n")
        if step % cfg.training.log_every == 0 or step == cfg.training.max_steps - 1:
            echo(record.summary())

        if cfg.training.show_reconstruction and (
            step % cfg.training.sample_every == 0 or step == cfg.training.max_steps - 1
        ):
            pred = np.asarray(mx.argmax(out.logits, axis=-1).astype(mx.int32))
            block = format_reconstruction(
                tokenizer, res.x0[0], res.xt[0], pred[0], prefix_np[0], step, float(res.t[0])
            )
            with samples_path.open("a") as f:
                f.write(block + "\n")
            echo(block)

    ckpt = save_adapters(model, cfg, run_dir / "checkpoint", step=cfg.training.max_steps)

    summary = {
        "backend": "mlx/unsloth",
        "config_name": cfg.run.name,
        "bidirectional": cfg.bidirectional_deltanet.enabled,
        "fusion": cfg.bidirectional_deltanet.fusion,
        "lora_families": setup.lora_report.families if setup.lora_report else [],
        "steps": len(history),
        "first_loss": history[0].loss if history else None,
        "final_loss": history[-1].loss if history else None,
        "best_loss": min((h.loss for h in history), default=None),
        "first_identity_accuracy": history[0].metrics.get("identity_accuracy") if history else None,
        "final_identity_accuracy": history[-1].metrics.get("identity_accuracy") if history else None,
        "final_corrupted_accuracy": history[-1].metrics.get("corrupted_accuracy") if history else None,
        "final_copy_rate": history[-1].metrics.get("copy_rate") if history else None,
        "final_copy_baseline_accuracy": history[-1].metrics.get("copy_baseline_accuracy") if history else None,
        "final_lift_over_copy": history[-1].metrics.get("lift_over_copy") if history else None,
        "final_next_token_accuracy": history[-1].metrics.get("next_token_accuracy") if history else None,
        "mean_corrupted_accuracy_last_quarter": _tail_mean(history, "corrupted_accuracy"),
        "mean_lift_last_quarter": _tail_mean(history, "lift_over_copy"),
        # Same key names as the torch backend so the shared verdict logic in the CLI
        # reads either backend without branching.
        "peak_memory_gb": max((h.peak_unified_gb for h in history), default=0.0),
        "memory_kind": "unified",
        "mean_tokens_per_sec": sum(h.tokens_per_sec for h in history) / len(history) if history else 0.0,
        "mean_seconds_per_step": sum(h.seconds for h in history) / len(history) if history else 0.0,
        "checkpoint": str(ckpt),
        "run_dir": str(run_dir),
        "params": setup.params,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def save_adapters(model, cfg: ExperimentConfig, path: Path, step: int = 0) -> Path:
    """Save only what training changed: LoRA factors, fusion parameters, conditioner."""
    from mlx.utils import tree_flatten

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in tree_flatten(model.trainable_parameters())}
    if not state:
        raise RuntimeError("nothing to save: no trainable parameters")
    mx.save_safetensors(str(path / "adapters.safetensors"), state)
    (path / "adapter_meta.json").write_text(
        json.dumps(
            {
                "step": step,
                "num_tensors": len(state),
                "backend": "mlx",
                "model_id": cfg.model.id,
                "bidirectional": cfg.bidirectional_deltanet.enabled,
                "fusion": cfg.bidirectional_deltanet.fusion,
                "lora_rank": cfg.lora.rank,
                "canvas_length": cfg.diffusion.canvas_length,
            },
            indent=2,
        )
    )
    from ..config import save_config

    save_config(cfg, path / "config.yaml")
    return path


def load_adapters(model, path: str | Path) -> dict:
    from mlx.utils import tree_unflatten

    path = Path(path)
    state = mx.load(str(path / "adapters.safetensors"))
    model.update(tree_unflatten(list(state.items())))
    mx.eval(model.parameters())
    meta_path = path / "adapter_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["loaded_tensors"] = len(state)
    return meta
