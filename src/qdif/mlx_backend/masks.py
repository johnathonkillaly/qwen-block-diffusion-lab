"""Canvas attention masks in MLX.

Mirrors `qdif.models.masks` (the torch v0.1 implementation) so the two backends
express the same thing: a causal prefix plus a fully-connected canvas block, applied
to full-attention layers only.

The DeltaNet layers still cannot take a bidirectional mask -- in v0.2 their
bidirectionality comes from the reverse recurrence in `deltanet.py`, not from here.
"""

from __future__ import annotations

import mlx.core as mx


def build_canvas_allow_matrix(
    total_length: int, canvas_spans: list[tuple[int, int]], bidirectional: bool = True
) -> mx.array:
    """Boolean [T, T]; entry [q, k] is True if query q may attend to key k."""
    idx = mx.arange(total_length)
    allow = idx[:, None] >= idx[None, :]  # causal
    if bidirectional:
        for start, end in canvas_spans:
            if not (0 <= start < end <= total_length):
                raise ValueError(
                    f"canvas span [{start}, {end}) out of range for length {total_length}"
                )
            block = (idx >= start) & (idx < end)
            allow = allow | (block[:, None] & block[None, :])
    return allow


def build_attention_mask(
    total_length: int,
    canvas_start: int,
    dtype: mx.Dtype = mx.bfloat16,
    bidirectional: bool = True,
) -> mx.array:
    """Additive [1, 1, T, T] mask for mlx-lm's scaled_dot_product_attention."""
    allow = build_canvas_allow_matrix(
        total_length, [(canvas_start, total_length)] if canvas_start >= 0 else [], bidirectional
    )
    neg = mx.finfo(dtype).min if hasattr(mx, "finfo") else -3.0e38
    mask = mx.where(allow, mx.array(0.0, dtype=dtype), mx.array(neg, dtype=dtype))
    return mask[None, None]
