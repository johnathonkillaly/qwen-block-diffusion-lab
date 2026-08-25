"""Diffusion math in MLX.

Deliberately a thin port of `qdif.diffusion.{corruption,objective}` rather than a
reimplementation. Corruption RNG stays on numpy so a seed reproduces the *same*
corrupted positions and replacement tokens as the torch v0.1 backend -- which is
what makes any v0.1 / v0.2 comparison a comparison of models rather than of noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class CorruptionResult:
    xt: np.ndarray  # [B, C] corrupted canvas
    x0: np.ndarray  # [B, C] clean canvas
    corrupted_mask: np.ndarray  # [B, C] bool, positions the process targeted
    t: np.ndarray  # [B] float32


def corrupt_canvas(
    x0: np.ndarray,
    t: np.ndarray,
    vocab_size: int,
    rng: np.random.Generator,
    mode: str = "uniform",
    mask_token_id: int | None = None,
) -> CorruptionResult:
    """Independent per-position corruption; see docs/DIFFUSION_OBJECTIVE.md."""
    x0 = np.asarray(x0)
    if x0.ndim != 2:
        raise ValueError(f"x0 must be [B, C], got {x0.shape}")
    B, C = x0.shape
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    if t.size == 1:
        t = np.repeat(t, B)
    if t.size != B:
        raise ValueError(f"t must have {B} entries, got {t.size}")
    if (t < 0).any() or (t > 1).any():
        raise ValueError("timesteps must lie in [0, 1]")

    corrupted = rng.random((B, C)).astype(np.float32) < t[:, None]
    if mode == "uniform":
        replacement = rng.integers(0, vocab_size, size=(B, C), dtype=np.int64)
    elif mode == "mask":
        if mask_token_id is None:
            raise ValueError("mode='mask' requires mask_token_id")
        replacement = np.full((B, C), int(mask_token_id), dtype=np.int64)
    else:
        raise ValueError(f"unknown corruption mode {mode!r}")

    xt = np.where(corrupted, replacement, x0)
    return CorruptionResult(xt=xt, x0=x0, corrupted_mask=corrupted, t=t)


def cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    """Mean token cross entropy over [B, C, V] logits, computed in fp32.

    fp32 matters: a bf16 logsumexp over 248,320 logits distorts the reported loss.
    """
    logits = logits.astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picked = mx.take_along_axis(logprobs, targets[..., None].astype(mx.int32), axis=-1)
    return -picked.squeeze(-1).mean()


def diffusion_metrics(
    logits: mx.array, x0: np.ndarray, xt: np.ndarray, corrupted_mask: np.ndarray
) -> dict[str, float]:
    """The v0.1 anti-self-deception metric set, unchanged.

    `lift_over_copy` remains the headline number: identity accuracy alone rises to
    the copy baseline for free under uniform corruption.
    """
    pred = np.asarray(mx.argmax(logits, axis=-1).astype(mx.int32))
    x0 = np.asarray(x0)
    xt = np.asarray(xt)
    correct = pred == x0
    corrupted = np.asarray(corrupted_mask, dtype=bool)
    clean = ~corrupted

    def masked_mean(a, m):
        return float(a[m].mean()) if m.any() else float("nan")

    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    top1 = float(mx.max(probs, axis=-1).mean())
    entropy = float((-(mx.log(mx.clip(probs, 1e-9, 1.0)) * probs).sum(axis=-1)).mean())

    copy_baseline = float((xt == x0).mean())
    identity = float(correct.mean())
    out = {
        "identity_accuracy": identity,
        "corrupted_accuracy": masked_mean(correct, corrupted),
        "clean_accuracy": masked_mean(correct, clean),
        "copy_rate": float((pred == xt).mean()),
        "copy_baseline_accuracy": copy_baseline,
        "lift_over_copy": identity - copy_baseline,
        "mean_top1_prob": top1,
        "mean_entropy": entropy,
    }
    out["next_token_accuracy"] = (
        float((pred[:, :-1] == x0[:, 1:]).mean()) if x0.shape[1] > 1 else float("nan")
    )
    return out
