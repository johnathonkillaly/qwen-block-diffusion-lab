"""Training-batch construction: the frozen model teaching its own future.

## The layout, and why it is not IFM's

Uno noises every block of a sequence at once and separates the clean teacher rows from
the noisy student rows with a custom `block_diff_mask` over a doubled sequence, so one
forward yields both. **That mask cannot be expressed on Qwen3.5's 24 Gated DeltaNet
layers**, which are causal by construction rather than by masking
(`FINDINGS.md`). Interleaving clean and noisy rows in one recurrence would leak the
answer into the student in a way we could not gate, and the experiment would be void.

So we use one noisy block per window, at the suffix, and two plain causal forwards.
For a window `x[0 … W-1]` with block size `L`, let `s = W - L - 1`:

```
teacher :  x[0] … x[W-2]                          adapter OFF
student :  x[0] … x[s]  ν₁ … ν_{L-1}              adapter ON only on ν rows
                     ↑ seed
supervised positions: s … s+L-1   (predicting x[s+1] … x[s+L])
```

Both forwards are `W-1` wide and agree on their first `s+1` tokens. This is *exactly*
the computation the draft pass performs at decode time — arguably a tighter train/test
match than Uno's own layout, since no mask approximation is involved. The cost is two
forwards per step instead of one, and `L-1` supervised positions per window instead of
every block. At our scale that is the right trade.

## The seed-row invariant

Position `s` carries the seed with the adapter switched **off**, so the student and
teacher computations there are bit-identical and its TV contribution must be exactly
zero. That is a free, continuous check that gating is actually working: if it ever
becomes non-zero, the adapter is leaking onto clean rows and every result is suspect.
`stage0.py` asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .noise import make_noise


@dataclass
class UnoBatch:
    """One training batch. All arrays are `[B, W-1]` unless noted."""

    teacher_ids: mx.array
    student_ids: mx.array
    lora_mask: mx.array
    #: Index of the seed position; supervised span is `block_start … block_start+L-1`.
    block_start: int
    block_size: int
    #: `[B, L]` — the true tokens at the supervised positions, i.e. x[s+1 … s+L].
    targets: mx.array
    #: `[B, L]` — 1.0 where the student row actually holds noise.
    corrupted: mx.array
    corruption_rate: mx.array

    @property
    def batch_size(self) -> int:
        return int(self.teacher_ids.shape[0])

    @property
    def supervised_slice(self) -> slice:
        return slice(self.block_start, self.block_start + self.block_size)


def build_uno_batch(
    windows: mx.array,
    block_size: int,
    mask_token_id: int,
    vocab_size: int,
    noise_mode: str = "random_uniform",
    corruption: str = "uniform",
    shuffle_targets: bool = False,
    rng_key: mx.array | None = None,
    key: mx.array | None = None,
) -> UnoBatch:
    """Build one batch from `[B, W]` token windows.

    `corruption`:
      * `"uniform"` — Uno/SDAR faithful: a per-sequence rate `t ~ U(0,1)` decides how
        many of the `L-1` draft rows hold noise. Trains the adapter across the whole
        noise schedule.
      * `"full"` — every draft row holds noise. This is the *only* state that occurs at
        inference, so it is the tighter train/test match and is worth measuring
        against the faithful setting rather than assuming.

    `shuffle_targets` is Control 2: the **teacher's** context is rolled by one sequence
    while the student's is left alone, so the adapter is asked to match a future that
    belongs to somebody else's context. It must fail. If it does not, the adapter is
    not using the context and any apparent learning is a context-free marginal.
    """
    if windows.ndim != 2:
        raise ValueError(f"windows must be [B, W], got {tuple(windows.shape)}")
    batch, width = windows.shape
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if width < block_size + 2:
        raise ValueError(
            f"window width {width} is too small for block_size {block_size}; "
            f"need at least {block_size + 2}"
        )

    block_start = width - block_size - 1
    true_ids = windows[:, : width - 1].astype(mx.int32)
    targets = windows[:, block_start + 1 : block_start + 1 + block_size].astype(mx.int32)

    # The control must break the *student -> teacher* correspondence, which means
    # rolling the teacher's context while leaving the student's alone. Rolling both
    # (the obvious-looking implementation) keeps them perfectly aligned and, under a
    # TV-only objective that never reads `targets`, makes the control bit-identical to
    # the real arm — a no-op control that silently reports "no separation".
    teacher_ids = mx.roll(true_ids, shift=1, axis=0) if shuffle_targets else true_ids
    student_prefix = true_ids[:, : block_start + 1]
    num_draft = block_size - 1
    if num_draft == 0:
        student_ids = student_prefix
        corrupted = mx.zeros((batch, 0), dtype=mx.float32)
        rate = mx.zeros((batch,), dtype=mx.float32)
    else:
        noise_key, rate_key, draw_key = (
            (None, None, None) if key is None else tuple(mx.random.split(key, 3))
        )
        noise = make_noise(
            (batch, num_draft), noise_mode, mask_token_id, vocab_size,
            seed=None if rng_key is None else int(rng_key.item()),
            key=noise_key,
        )
        clean_tail = true_ids[:, block_start + 1 : block_start + 1 + num_draft]
        if corruption == "full":
            keep_noise = mx.ones((batch, num_draft), dtype=mx.float32)
            rate = mx.ones((batch,), dtype=mx.float32)
        elif corruption == "uniform":
            rate = mx.random.uniform(shape=(batch, 1), key=rate_key)
            keep_noise = (
                mx.random.uniform(shape=(batch, num_draft), key=draw_key) < rate
            ).astype(mx.float32)
            rate = rate.reshape(batch)
        else:
            raise ValueError(f"unknown corruption schedule {corruption!r}")
        student_tail = mx.where(keep_noise.astype(mx.bool_), noise, clean_tail)
        student_ids = mx.concatenate([student_prefix, student_tail], axis=1)
        corrupted = mx.concatenate(
            [mx.zeros((batch, 1), dtype=mx.float32), keep_noise], axis=1
        )

    if num_draft == 0:
        corrupted = mx.zeros((batch, 1), dtype=mx.float32)

    # Adapter off on the whole clean prefix *and* the seed; on for every draft row,
    # corrupted or not. This mirrors Uno, where the router mask covers the entire
    # noisy region rather than only the positions that happened to be corrupted.
    lora_mask = mx.concatenate(
        [
            mx.zeros((batch, block_start + 1), dtype=mx.float32),
            mx.ones((batch, num_draft), dtype=mx.float32),
        ],
        axis=1,
    )

    return UnoBatch(
        teacher_ids=teacher_ids,
        student_ids=student_ids,
        lora_mask=lora_mask,
        block_start=block_start,
        block_size=block_size,
        targets=targets,
        corrupted=corrupted,
        corruption_rate=rate,
    )


def teacher_logits_for(model, batch: UnoBatch) -> mx.array:
    """`[B, L, V]` frozen-AR next-token distributions at the supervised positions.

    Always under `stop_gradient`: the teacher is the frozen model and must never
    receive a gradient, not even through a shared adapter that happens to be off.
    """
    logits = model.ar_logits(batch.teacher_ids)
    return mx.stop_gradient(logits[:, batch.supervised_slice, :])


def student_logits_for(model, batch: UnoBatch) -> mx.array:
    """`[B, L, V]` adapter-conditioned distributions at the supervised positions."""
    logits = model.draft_logits(batch.student_ids, batch.lora_mask)
    return logits[:, batch.supervised_slice, :]
