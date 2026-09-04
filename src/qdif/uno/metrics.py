"""Block-prediction metrics.

## Two different questions, deliberately kept apart

**Teacher-forced agreement** (this module) asks: given a real corpus prefix, does the
student's distribution at future slot `i` have the same argmax as the frozen model's?
It is cheap, batched, and is the right training signal to watch.

**Free-running acceptance** (`decode.py`) asks: when the model generates its own
continuation, how many proposals survive verification? That is what determines the
speedup, and it is *not* the same number — teacher-forced metrics condition on corpus
text, while decoding conditions on the model's own greedy output. A model can look
good on the first and mediocre on the second.

Results must report both, and must never quote one as if it were the other.
"""

from __future__ import annotations

import mlx.core as mx


def slot_metrics(
    student_logits: mx.array,
    teacher_logits: mx.array,
    targets: mx.array,
) -> dict:
    """Per-future-slot agreement, accuracy and divergence.

    All arrays are `[B, L, V]` / `[B, L]`. Slot 0 is the seed, which runs with the
    adapter off and is therefore trivially in agreement — it is reported so that a
    regression there (which would mean the gate is leaking) is visible, but it is
    excluded from the aggregate `accept_*` figures.
    """
    batch, length, vocab = student_logits.shape
    student = student_logits.astype(mx.float32)
    teacher = teacher_logits.astype(mx.float32)

    student_argmax = mx.argmax(student, axis=-1)
    teacher_argmax = mx.argmax(teacher, axis=-1)
    agree = (student_argmax == teacher_argmax).astype(mx.float32)  # [B, L]
    gold = (student_argmax == targets.astype(student_argmax.dtype)).astype(mx.float32)

    student_log_probs = student - mx.logsumexp(student, axis=-1, keepdims=True)
    teacher_log_probs = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
    teacher_probs = mx.exp(teacher_log_probs)

    gold_index = targets.astype(mx.int32)[..., None]
    student_ce = -mx.take_along_axis(student_log_probs, gold_index, axis=-1)[..., 0]
    teacher_ce = -mx.take_along_axis(teacher_log_probs, gold_index, axis=-1)[..., 0]
    kl = mx.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    teacher_entropy = -mx.sum(teacher_probs * teacher_log_probs, axis=-1)

    # Teacher-forced accepted-prefix length over the speculative slots 1 … L-1.
    if length > 1:
        spec_agree = agree[:, 1:]
        cumulative = mx.cumprod(spec_agree, axis=1)
        accepted_prefix = mx.sum(cumulative, axis=1)  # [B]
    else:
        accepted_prefix = mx.zeros((batch,))

    def per_slot(values: mx.array) -> list[float]:
        return [round(float(v), 6) for v in mx.mean(values, axis=0).tolist()]

    offered = max(length - 1, 1)
    prefix_at_least = {
        f"prefix_ge_{k}": round(
            float(mx.mean((accepted_prefix >= k).astype(mx.float32)).item()), 6
        )
        for k in (1, 2, 4, 8)
        if k <= length - 1
    }

    return {
        "block_size": length,
        "rows": batch,
        # agreement with the frozen model — this is what acceptance depends on
        "agree_per_slot": per_slot(agree),
        "agree_specs": round(float(mx.mean(agree[:, 1:]).item()), 6) if length > 1 else 0.0,
        # agreement with the corpus token — the harder, less relevant question
        "gold_per_slot": per_slot(gold),
        "gold_specs": round(float(mx.mean(gold[:, 1:]).item()), 6) if length > 1 else 0.0,
        "student_ce_per_slot": per_slot(student_ce),
        "teacher_ce_per_slot": per_slot(teacher_ce),
        "kl_per_slot": per_slot(kl),
        "teacher_entropy_per_slot": per_slot(teacher_entropy),
        "mean_accepted_prefix": round(float(mx.mean(accepted_prefix).item()), 6),
        "full_block_rate": round(
            float(mx.mean((accepted_prefix >= length - 1).astype(mx.float32)).item()), 6
        )
        if length > 1
        else 1.0,
        "offered_specs": offered,
        **prefix_at_least,
    }


def entropy_buckets(
    student_logits: mx.array,
    teacher_logits: mx.array,
    num_buckets: int = 5,
) -> list[dict]:
    """Agreement as a function of teacher entropy.

    Tests the hypothesis in the brief's §26: parallel prediction should succeed where
    the continuation is already nearly determined, and fail where it is open. If that
    holds, a fixed block size is leaving throughput on the table and an entropy-routed
    block size is worth building.
    """
    student = student_logits.astype(mx.float32).reshape(-1, student_logits.shape[-1])
    teacher = teacher_logits.astype(mx.float32).reshape(-1, teacher_logits.shape[-1])
    if student.shape[0] == 0:
        return []
    teacher_log_probs = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
    entropy = -mx.sum(mx.exp(teacher_log_probs) * teacher_log_probs, axis=-1)
    agree = (mx.argmax(student, axis=-1) == mx.argmax(teacher, axis=-1)).astype(mx.float32)

    values = entropy.tolist()
    order = sorted(range(len(values)), key=lambda i: values[i])
    agree_list = agree.tolist()
    per_bucket = max(1, len(order) // num_buckets)

    buckets = []
    for index in range(num_buckets):
        lo = index * per_bucket
        hi = len(order) if index == num_buckets - 1 else min(len(order), (index + 1) * per_bucket)
        chunk = order[lo:hi]
        if not chunk:
            continue
        buckets.append(
            {
                "bucket": index,
                "n": len(chunk),
                "entropy_lo": round(values[chunk[0]], 4),
                "entropy_hi": round(values[chunk[-1]], 4),
                "mean_entropy": round(sum(values[i] for i in chunk) / len(chunk), 4),
                "agreement": round(sum(agree_list[i] for i in chunk) / len(chunk), 4),
            }
        )
    return buckets


def pearson(xs: list[float], ys: list[float]) -> float:
    """Plain Pearson correlation; returns 0.0 when either series is constant."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = (sum(v * v for v in dx) ** 0.5) * (sum(v * v for v in dy) ** 0.5)
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / denominator
