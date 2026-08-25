"""Directionality probe: does information actually travel backwards?

The v0.1 probe measured the whole stack and could only show that *something* was
bidirectional -- which, in v0.1, was necessarily the full-attention layers, because
the DeltaNet layers are causal by construction.

The v0.2 probe is stronger and per-layer. For a selected DeltaNet layer, in
isolation:

    1. build a canvas,
    2. perturb the FINAL canvas token,
    3. measure how the layer's representation changes at EARLIER positions.

Two conditions, same weights:

    native causal DeltaNet   -> earlier positions must not change at all
    bidirectional DeltaNet   -> earlier positions must change, because the reverse
                                recurrence carries information backwards

Reported per condition: number of affected earlier positions, mean L2 delta, max L2
delta, and mean cosine change. A causal layer that shows any earlier-position change
is a bug; a bidirectional layer that shows none means the fusion is not admitting
the reverse direction.

The probe also measures the prefix, because the leakage boundary matters as much as
the bidirectionality: a canvas perturbation must never move a prefix position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .deltanet import BidirectionalGatedDeltaNet


@dataclass
class DirectionStats:
    condition: str
    layer: int
    canvas_positions: int
    earlier_positions_affected: int
    mean_l2_delta: float
    max_l2_delta: float
    mean_cosine_change: float
    prefix_positions_affected: int
    prefix_max_l2_delta: float
    per_position_l2: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"  {self.condition:<24} layer {self.layer:>2} | affected "
            f"{self.earlier_positions_affected:>3}/{self.canvas_positions - 1} | "
            f"meanL2 {self.mean_l2_delta:.4e} | maxL2 {self.max_l2_delta:.4e} | "
            f"cos-change {self.mean_cosine_change:.4e} | prefix affected "
            f"{self.prefix_positions_affected}"
        )


def to_numpy(a: mx.array) -> np.ndarray:
    """numpy has no bfloat16 dtype, so cast through float32 before crossing over."""
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.astype(mx.float32)
    return np.asarray(a)


def _l2_and_cosine(a: mx.array, b: mx.array):
    """Per-position L2 delta and cosine change between two [1, S, D] tensors."""
    a32 = a.astype(mx.float32)[0]
    b32 = b.astype(mx.float32)[0]
    delta = mx.sqrt(mx.sum((a32 - b32) ** 2, axis=-1))
    na = mx.sqrt(mx.sum(a32**2, axis=-1))
    nb = mx.sqrt(mx.sum(b32**2, axis=-1))
    cos = mx.sum(a32 * b32, axis=-1) / mx.maximum(na * nb, 1e-9)
    return np.asarray(delta), np.asarray(1.0 - cos)


def probe_layer(
    model,
    hidden: mx.array,
    layer_index: int,
    canvas_start: int,
    condition: str,
    bidirectional: bool,
    threshold: float = 1e-4,
) -> DirectionStats:
    """Run one DeltaNet layer twice (clean vs perturbed final canvas token)."""
    layer = model.layers[layer_index]
    mod = layer.linear_attn
    if not isinstance(mod, BidirectionalGatedDeltaNet):
        raise TypeError(
            f"layer {layer_index} is not a BidirectionalGatedDeltaNet; wrap the model first"
        )

    previous = mod.bidirectional
    mod.bidirectional = bidirectional
    mod.set_canvas(canvas_start)
    try:
        base_out = mod(hidden)
        # Perturb the final canvas position only.
        pert = to_numpy(hidden).copy()
        rng = np.random.default_rng(0)
        pert[:, -1, :] += rng.normal(0.0, 1.0, size=pert.shape[-1]).astype(pert.dtype)
        perturbed = mx.array(pert).astype(hidden.dtype)
        pert_out = mod(perturbed)
        mx.eval(base_out, pert_out)
    finally:
        mod.bidirectional = previous

    l2, cos_change = _l2_and_cosine(base_out, pert_out)

    canvas_l2 = l2[canvas_start:]
    earlier = canvas_l2[:-1]  # exclude the perturbed position itself
    earlier_cos = cos_change[canvas_start:][:-1]
    prefix_l2 = l2[:canvas_start]

    return DirectionStats(
        condition=condition,
        layer=layer_index,
        canvas_positions=int(canvas_l2.size),
        earlier_positions_affected=int((earlier > threshold).sum()),
        mean_l2_delta=float(earlier.mean()) if earlier.size else 0.0,
        max_l2_delta=float(earlier.max()) if earlier.size else 0.0,
        mean_cosine_change=float(earlier_cos.mean()) if earlier_cos.size else 0.0,
        prefix_positions_affected=int((prefix_l2 > threshold).sum()),
        prefix_max_l2_delta=float(prefix_l2.max()) if prefix_l2.size else 0.0,
        per_position_l2=[round(float(v), 8) for v in canvas_l2],
    )


def probe_directionality(
    setup,
    text: str,
    canvas_length: int = 16,
    layer_index: int | None = None,
    threshold: float = 1e-4,
) -> dict:
    """Full probe: one early DeltaNet layer, both conditions, plus a stack-level check."""
    model = setup.model
    tokenizer = setup.tokenizer
    ids = mx.array([tokenizer.encode(text)])
    if ids.shape[1] <= canvas_length:
        raise ValueError(
            f"probe text is {ids.shape[1]} tokens; need more than canvas_length={canvas_length}"
        )
    canvas_start = ids.shape[1] - canvas_length

    linear_layers = [i for i, l in enumerate(model.layers) if l.is_linear]
    if layer_index is None:
        layer_index = linear_layers[0]

    hidden = model.embed(ids)
    hidden = model.layers[layer_index].input_layernorm(hidden)
    mx.eval(hidden)

    causal = probe_layer(
        model, hidden, layer_index, canvas_start, "native causal DeltaNet", False, threshold
    )
    bidi = probe_layer(
        model, hidden, layer_index, canvas_start, "bidirectional DeltaNet", True, threshold
    )

    verdict_ok = (
        causal.earlier_positions_affected == 0
        and bidi.earlier_positions_affected > 0
        and bidi.prefix_positions_affected == 0
    )

    return {
        "layer_probed": layer_index,
        "deltanet_layers": linear_layers,
        "full_attention_layers": [i for i, l in enumerate(model.layers) if not l.is_linear],
        "canvas_start": canvas_start,
        "canvas_length": canvas_length,
        "causal": causal.__dict__,
        "bidirectional": bidi.__dict__,
        "pass": bool(verdict_ok),
        "interpretation": (
            "PASS requires all three: the causal condition moves ZERO earlier positions "
            "(the recurrence really is causal by construction), the bidirectional "
            "condition moves at least one (the reverse recurrence really does carry "
            "information backwards), and NO prefix position moves in either condition "
            "(the leakage boundary holds)."
        ),
    }
