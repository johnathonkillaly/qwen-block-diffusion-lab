"""Distillation objectives, transcribed from `nano_vllm_uno/training/losses.py`.

Uno's public default is **TV-only**: `CE_ALPHA=0`, `KL_BETA=0`, `TV_GAMMA=1`
(`nano_vllm_uno/training/constants.py`). CE and reverse KL are implemented and wired
to the same config knobs so that "TV is the right choice" stays a measurable claim
rather than an inherited assumption.

Why total variation and not KL, since KL is the reflex for distillation: the decode
rule accepts a proposal iff the student's argmax equals the teacher's argmax. TV
bounds the probability that two distributions disagree on *any* event, which is the
quantity acceptance actually depends on. KL spends gradient on tail mass that greedy
acceptance never observes.

We follow IFM in using the raw L1 sum rather than the ½-scaled statistician's TV, so
values here are twice the formal total-variation distance and lie in [0, 2].
"""

from __future__ import annotations

import mlx.core as mx


def total_variation(
    student_logits: mx.array,
    teacher_logits: mx.array,
    num_tokens: int | None = None,
    chunk_size: int = 32768,
) -> mx.array:
    """‖softmax(student) − softmax(teacher)‖₁, summed and normalised by token count.

    `teacher_logits` must already be detached (`mx.stop_gradient`) by the caller; this
    function does not do it, so that a mistake is visible at the call site.

    The vocabulary axis is chunked. At 248,320 classes a dense fp32 softmax pair costs
    ~2 MB per row, which is survivable for a handful of rows and is not for a batch of
    a few hundred, and MLX keeps every intermediate alive for the backward pass.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"logit shapes differ: {tuple(student_logits.shape)} vs "
            f"{tuple(teacher_logits.shape)}"
        )
    vocab = student_logits.shape[-1]
    student = student_logits.reshape(-1, vocab).astype(mx.float32)
    teacher = teacher_logits.reshape(-1, vocab).astype(mx.float32)
    rows = student.shape[0]
    if rows == 0:
        return mx.zeros(())
    if num_tokens is None:
        num_tokens = rows

    student_log_probs = student - mx.logsumexp(student, axis=-1, keepdims=True)
    teacher_log_probs = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)

    step = min(chunk_size, vocab)
    total = mx.zeros(())
    for start in range(0, vocab, step):
        stop = min(start + step, vocab)
        difference = (
            mx.exp(student_log_probs[:, start:stop])
            - mx.exp(teacher_log_probs[:, start:stop])
        )
        total = total + mx.sum(mx.abs(difference))
    return total / max(num_tokens, 1)


def reverse_kl(
    student_logits: mx.array,
    teacher_logits: mx.array,
    num_tokens: int | None = None,
) -> mx.array:
    """KL(student ‖ teacher), normalised per supervised token.

    Reverse, i.e. mode-seeking: mass the student puts where the teacher has none is
    what gets punished. Matches `token_normalized_reverse_kl`.
    """
    vocab = student_logits.shape[-1]
    student = student_logits.reshape(-1, vocab).astype(mx.float32)
    teacher = teacher_logits.reshape(-1, vocab).astype(mx.float32)
    rows = student.shape[0]
    if rows == 0:
        return mx.zeros(())
    if num_tokens is None:
        num_tokens = rows
    student_log_probs = student - mx.logsumexp(student, axis=-1, keepdims=True)
    teacher_log_probs = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
    kl = mx.sum(mx.exp(student_log_probs) * (student_log_probs - teacher_log_probs))
    return kl / max(num_tokens, 1)


def cross_entropy(
    student_logits: mx.array,
    targets: mx.array,
    num_tokens: int | None = None,
) -> mx.array:
    """Hard-target CE against `targets` (data labels or the teacher's argmax)."""
    vocab = student_logits.shape[-1]
    student = student_logits.reshape(-1, vocab).astype(mx.float32)
    labels = targets.reshape(-1)
    rows = student.shape[0]
    if rows == 0:
        return mx.zeros(())
    if labels.shape[0] != rows:
        raise ValueError(f"targets ({labels.shape[0]}) do not match rows ({rows})")
    if num_tokens is None:
        num_tokens = rows
    log_probs = student - mx.logsumexp(student, axis=-1, keepdims=True)
    picked = mx.take_along_axis(log_probs, labels[:, None].astype(mx.int32), axis=-1)
    return -mx.sum(picked) / max(num_tokens, 1)


def combine(
    ce_loss: mx.array,
    kl_loss: mx.array,
    tv_loss: mx.array,
    ce_weight: float,
    kl_weight: float,
    tv_weight: float,
) -> mx.array:
    """`α·CE + β·KL + γ·TV`, matching `trainer.py:combine_objective_losses`."""
    if min(ce_weight, kl_weight, tv_weight) < 0:
        raise ValueError("objective weights must be non-negative")
    if max(ce_weight, kl_weight, tv_weight) <= 0:
        raise ValueError("at least one objective weight must be positive")
    return ce_weight * ce_loss + kl_weight * kl_loss + tv_weight * tv_loss


# ------------------------------------------------------------------ diagnostics


def agreement_metrics(
    student_logits: mx.array, teacher_logits: mx.array
) -> dict[str, float]:
    """Argmax agreement and teacher entropy — the quantities acceptance depends on.

    Kept separate from the loss so that it can be run under `mx.stop_gradient` on the
    evaluation path without touching the training graph.
    """
    vocab = student_logits.shape[-1]
    student = student_logits.reshape(-1, vocab).astype(mx.float32)
    teacher = teacher_logits.reshape(-1, vocab).astype(mx.float32)
    if student.shape[0] == 0:
        return {"top1_agreement": 0.0, "teacher_entropy": 0.0, "teacher_top1_prob": 0.0}
    teacher_log_probs = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
    teacher_probs = mx.exp(teacher_log_probs)
    agreement = mx.mean(
        (mx.argmax(student, axis=-1) == mx.argmax(teacher, axis=-1)).astype(mx.float32)
    )
    entropy = mx.mean(-mx.sum(teacher_probs * teacher_log_probs, axis=-1))
    top1 = mx.mean(mx.max(teacher_probs, axis=-1))
    return {
        "top1_agreement": float(agreement.item()),
        "teacher_entropy": float(entropy.item()),
        "teacher_top1_prob": float(top1.item()),
    }
