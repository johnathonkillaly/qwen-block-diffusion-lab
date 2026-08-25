"""Bidirectional Gated DeltaNet on shared pretrained weights.

THE PROBLEM
-----------
Qwen3.5's Gated DeltaNet is causal *by construction*: a causal depthwise conv1d
feeds a sequential recurrence (`gated_delta_update` loops over t). No mask makes it
bidirectional. v0.1 therefore left all 24 DeltaNet layers causal and got backward
information flow only from the 8 full-attention layers.

THE v0.2 HYPOTHESIS
-------------------
A pretrained causal recurrence can still supply useful *bidirectional* features if
the SAME weights are evaluated a second time on the reversed canvas and the two
directional representations are fused:

    H_f     = D(x_1 ... x_C ; W)          forward, as pretrained
    H_r_rev = D(x_C ... x_1 ; W)          reverse, same W
    H_r     = reverse(H_r_rev)            realign to original positions
    H_bi    = Fuse(H_f, H_r)

`W` is shared, not copied. A 4B backbone must not become an 8B backbone.

WHERE THE FUSION HAPPENS -- and why
-----------------------------------
mlx-lm's `GatedDeltaNet.__call__` is, in order:

    1  in_proj_qkv / in_proj_z / in_proj_b / in_proj_a   (position-local)
    2  causal depthwise conv1d over qkv                  (mixes across positions)
    3  reshape to heads; rms_norm + scaling on q, k
    4  gated_delta_update  ->  y [B, S, Hv, Dv]          (the recurrence)
    5  norm(y, z)          RMSNormGated, z from step 1   (nonlinear, gated)
    6  out_proj            -> hidden_size
    7  (in DecoderLayer)   h = x + r                     residual

We fuse at the output of **step 4**, the raw recurrent output, before the gated norm
and before the output projection. Reasons:

  * Step 4 is the only stage that carries directional information at all. Steps 1
    and 5-6 are position-local; step 2 is directional but is *part of* what we want
    to run in reverse, not a place to combine.
  * `z` (the output gate) is position-local, so the forward pass's `z` is already
    correctly aligned for the fused representation. Gating is applied ONCE, with
    pretrained semantics intact.
  * `out_proj` is applied ONCE, so exactly one contribution enters the residual
    stream. Fusing after `out_proj` and averaging would be equivalent for `mean`
    (out_proj is linear and bias-free) but would double-count for any gated fusion,
    and would add a second nonlinearity's worth of drift for `norm`.
  * The residual `h = x + r` stays a single addition. Running a whole layer twice
    and averaging post-residual outputs would silently halve the residual branch and
    change the layer's effective depth semantics -- the failure mode this design
    exists to avoid.

LEAKAGE BOUNDARY -- the critical correctness constraint
-------------------------------------------------------
Sequence layout is `[ fixed causal history | editable canvas ]`. The reverse pass
runs over the **canvas only**, starting from a zero recurrent state:

  * Forward pass covers the full sequence, unchanged, so canvas position i sees the
    prefix and canvas positions <= i.
  * Reverse pass covers positions [P, T) only, so canvas position i additionally
    sees canvas positions >= i -- and nothing outside the canvas.
  * Prefix positions are never fused; they keep the pure forward representation.

Consequences, both required:
  * no canvas information ever reaches a prefix position (the prefix stays causal
    and stable, which is what makes block-autoregressive commitment sound), and
  * the reverse pass cannot "see" the prompt, so committed history is never
    re-entered from the future direction.

Running the reverse pass over the full sequence instead would give prefix positions
information about the canvas -- that is the leak this module is built to prevent, and
`tests/test_bidirectional_deltanet.py` tests exactly that boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

FUSION_STRATEGIES = ("mean", "scalar_gate", "token_gate", "concat_proj")


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


# --------------------------------------------------------------------- front end


def deltanet_front_end(base, inputs: mx.array, mask: mx.array | None = None):
    """Run steps 1-3 of `GatedDeltaNet.__call__` using the base module's own weights.

    Mirrors mlx_lm/models/qwen3_5.py exactly. It is deliberately a re-expression
    rather than a copy of the weights: every tensor used here belongs to `base`.

    `tests/test_bidirectional_deltanet.py::test_forward_path_matches_upstream`
    asserts that this decomposition reproduces `base(x)` bit-for-bit, so an upstream
    change to mlx-lm fails loudly instead of silently altering the science.

    Returns:
        q, k, v, a, b, z  -- everything `gated_delta_update` and the gated norm need.
    """
    B, S, _ = inputs.shape

    qkv = base.in_proj_qkv(inputs)
    z = base.in_proj_z(inputs).reshape(B, S, base.num_v_heads, base.head_v_dim)
    b = base.in_proj_b(inputs)
    a = base.in_proj_a(inputs)

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)

    # Causal depthwise conv: left-pad with zeros, no cache. Because the caller may
    # hand us a reversed sequence, this becomes a reverse-time causal conv for free.
    conv_state = mx.zeros((B, base.conv_kernel_size - 1, base.conv_dim), dtype=inputs.dtype)
    conv_out = nn.silu(base.conv1d(mx.concatenate([conv_state, qkv], axis=1)))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [base.key_dim, 2 * base.key_dim], -1),
            [base.num_k_heads, base.num_k_heads, base.num_v_heads],
            [base.head_k_dim, base.head_k_dim, base.head_v_dim],
        )
    ]

    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    return q, k, v, a, b, z


def deltanet_recurrence(base, q, k, v, a, b, mask=None, use_kernel=False):
    """Step 4: the recurrence itself. Returns raw `y` of shape [B, S, Hv, Dv].

    `use_kernel=False` selects the ops path, which is the differentiable one; the
    fused Metal kernel is inference-only in mlx-lm (`use_kernel=not self.training`).
    """
    from mlx_lm.models.gated_delta import gated_delta_update

    y, _ = gated_delta_update(
        q, k, v, a, b, base.A_log, base.dt_bias, None, mask, use_kernel=use_kernel
    )
    return y


# ----------------------------------------------------------------------- fusions


class Fusion(nn.Module):
    """Combine forward and reverse recurrent outputs, both [B, C, Hv, Dv]."""

    def __call__(self, h_f: mx.array, h_r: mx.array) -> mx.array:  # pragma: no cover
        raise NotImplementedError

    @property
    def num_parameters(self) -> int:
        from mlx.utils import tree_flatten

        return sum(v.size for _, v in tree_flatten(self.parameters()))


class MeanFusion(Fusion):
    """H = (H_f + H_r) / 2. No trainable parameters. The control condition.

    Note this is NOT a no-op at initialisation: it immediately halves the pretrained
    forward representation and mixes in an untrained-for direction. That is exactly
    why it is the control -- any gain has to come from the reverse information
    itself, not from a learned gate that could simply have turned the reverse path
    off.
    """

    def __call__(self, h_f, h_r):
        return (h_f + h_r) * 0.5


class ScalarGateFusion(Fusion):
    """H = g * H_f + (1 - g) * H_r with a single learned scalar per layer.

    Initialised at g = `gate_init` (default 0.95) so the layer starts close to the
    pretrained causal computation and has to *learn* to admit the reverse direction.
    """

    def __init__(self, gate_init: float = 0.95):
        super().__init__()
        self.alpha = mx.array([_logit(gate_init)], dtype=mx.float32)

    def gate(self) -> mx.array:
        return mx.sigmoid(self.alpha)

    def __call__(self, h_f, h_r):
        g = mx.sigmoid(self.alpha).astype(h_f.dtype)
        return g * h_f + (1.0 - g) * h_r


class TokenGateFusion(Fusion):
    """Per-token, per-head gate: g_i = sigmoid(W_g [feat(H_f_i) ; feat(H_r_i)] + b_g).

    A gate over the full [Hv, Dv] value tensor would need a 2*Hv*Dv -> Hv*Dv matrix
    (33M parameters per layer at 4B). Instead the gate features are the per-head
    means over the value dimension, giving a 2*Hv -> Hv map: 2048 parameters per
    layer, and the resulting gate is per (token, head). This keeps the module small,
    as specified, at the cost of not being able to gate individual value channels --
    documented rather than hidden.

    `W_g` is zero-initialised and `b_g = logit(gate_init)`, so at step 0 the gate is
    a uniform `gate_init` everywhere and the pretrained forward path dominates.
    """

    def __init__(self, num_v_heads: int, gate_init: float = 0.95):
        super().__init__()
        self.num_v_heads = num_v_heads
        self.w_g = mx.zeros((2 * num_v_heads, num_v_heads), dtype=mx.float32)
        self.b_g = mx.full((num_v_heads,), _logit(gate_init), dtype=mx.float32)

    def __call__(self, h_f, h_r):
        feats = mx.concatenate(
            [h_f.mean(axis=-1).astype(mx.float32), h_r.mean(axis=-1).astype(mx.float32)],
            axis=-1,
        )  # [B, C, 2*Hv]
        g = mx.sigmoid(feats @ self.w_g + self.b_g)  # [B, C, Hv]
        g = g[..., None].astype(h_f.dtype)  # broadcast over Dv
        return g * h_f + (1.0 - g) * h_r


class ConcatProjFusion(Fusion):
    """H_i = W_p [H_f_i ; H_r_i], applied per head over the value dimension.

    A full [2*Hv*Dv -> Hv*Dv] projection is 33M parameters per layer at 4B. This
    shares one [2*Dv -> Dv] projection across heads: 65,536 parameters per layer.
    Initialised to [I ; 0] so it starts as an exact identity on the forward path.
    """

    def __init__(self, head_v_dim: int):
        super().__init__()
        eye = mx.eye(head_v_dim, dtype=mx.float32)
        self.w_p = mx.concatenate([eye, mx.zeros((head_v_dim, head_v_dim))], axis=0)

    def __call__(self, h_f, h_r):
        cat = mx.concatenate([h_f, h_r], axis=-1).astype(mx.float32)
        return (cat @ self.w_p).astype(h_f.dtype)


def build_fusion(strategy: str, num_v_heads: int, head_v_dim: int, gate_init: float) -> Fusion:
    if strategy == "mean":
        return MeanFusion()
    if strategy == "scalar_gate":
        return ScalarGateFusion(gate_init)
    if strategy == "token_gate":
        return TokenGateFusion(num_v_heads, gate_init)
    if strategy == "concat_proj":
        return ConcatProjFusion(head_v_dim)
    raise ValueError(f"unknown fusion {strategy!r}; expected one of {FUSION_STRATEGIES}")


# ------------------------------------------------------------------- the wrapper


@dataclass
class DirectionalOutputs:
    """Kept for the probe: the two directional representations before fusion."""

    h_forward: mx.array
    h_reverse: mx.array | None
    h_fused: mx.array


class BidirectionalGatedDeltaNet(nn.Module):
    """Drop-in replacement for a Qwen3.5 `GatedDeltaNet` that shares its weights.

    `self.base` holds the *same module object* that was already in the layer, so all
    ~4B pretrained parameters are referenced, never duplicated. The only new
    parameters are the fusion module's (0 to 65k per layer).
    """

    def __init__(self, base, fusion: Fusion, enabled: bool = True):
        super().__init__()
        self.base = base
        self.fusion = fusion
        self.bidirectional = enabled
        # Set per forward by the diffusion model; -1 means "no canvas, pure causal".
        self._canvas_start = -1
        self._capture: DirectionalOutputs | None = None
        self._capture_enabled = False

    # -- properties forwarded so the wrapper looks like the module it replaces ----
    @property
    def num_v_heads(self):
        return self.base.num_v_heads

    @property
    def head_v_dim(self):
        return self.base.head_v_dim

    def set_canvas(self, canvas_start: int) -> None:
        """Declare where the editable canvas begins. Everything before it is fixed
        causal history and is never reversed."""
        self._canvas_start = canvas_start

    def capture(self, enabled: bool = True) -> None:
        self._capture_enabled = enabled

    @property
    def last_directional(self) -> DirectionalOutputs | None:
        return self._capture

    def __call__(self, inputs: mx.array, mask=None, cache=None) -> mx.array:
        if cache is not None:
            raise NotImplementedError(
                "BidirectionalGatedDeltaNet does not support incremental cache. v0.2 "
                "recomputes the canvas recurrence every denoising step on purpose: a "
                "cached recurrent state goes stale the moment an earlier canvas token "
                "is rewritten, and silently-stale state is the failure mode this "
                "milestone refuses to risk. See docs/BIDIRECTIONAL_DELTANET.md."
            )

        base = self.base
        B, S, _ = inputs.shape
        use_kernel = not self.training

        # ---- forward direction: the pretrained computation, untouched -----------
        q, k, v, a, b, z = deltanet_front_end(base, inputs, mask)
        h_f = deltanet_recurrence(base, q, k, v, a, b, mask, use_kernel=use_kernel)

        h_r = None
        h = h_f
        start = self._canvas_start
        if self.bidirectional and 0 <= start < S:
            # ---- reverse direction: canvas only, zero initial state -------------
            canvas = inputs[:, start:]
            canvas_rev = mx.flip(canvas, axis=1)
            mask_rev = None
            if mask is not None:
                mask_rev = mx.flip(mask[:, start:], axis=1)

            qr, kr, vr, ar, br, _ = deltanet_front_end(base, canvas_rev, mask_rev)
            h_r_rev = deltanet_recurrence(base, qr, kr, vr, ar, br, mask_rev, use_kernel=use_kernel)
            h_r = mx.flip(h_r_rev, axis=1)  # realign to original positions

            fused = self.fusion(h_f[:, start:], h_r)
            h = mx.concatenate([h_f[:, :start], fused], axis=1)

        if self._capture_enabled:
            self._capture = DirectionalOutputs(h_forward=h_f, h_reverse=h_r, h_fused=h)

        # ---- gated norm and output projection, applied exactly once -------------
        out = base.norm(h, z)
        return base.out_proj(out.reshape(B, S, -1))


def wrap_deltanet_layers(
    model,
    fusion: str = "mean",
    gate_init: float = 0.95,
    layer_indices: list[int] | None = None,
    enabled: bool = True,
) -> dict:
    """Replace every (selected) `GatedDeltaNet` with a shared-weight bidirectional one.

    Returns a report including the identity of the shared tensors, so weight sharing
    can be asserted rather than assumed.
    """
    layers = model.layers
    selected = set(layer_indices) if layer_indices else None

    wrapped: list[int] = []
    fusion_params = 0
    shared_ids: dict[int, int] = {}

    for i, layer in enumerate(layers):
        if not getattr(layer, "is_linear", False):
            continue
        if selected is not None and i not in selected:
            continue
        base = layer.linear_attn
        if isinstance(base, BidirectionalGatedDeltaNet):
            continue
        shared_ids[i] = id(base.in_proj_qkv.weight)
        module = BidirectionalGatedDeltaNet(
            base,
            build_fusion(fusion, base.num_v_heads, base.head_v_dim, gate_init),
            enabled=enabled,
        )
        layer.linear_attn = module
        wrapped.append(i)
        fusion_params += module.fusion.num_parameters

    return {
        "wrapped_layers": wrapped,
        "num_wrapped": len(wrapped),
        "fusion": fusion,
        "fusion_params": fusion_params,
        "gate_init": gate_init,
        "shared_weight_ids": shared_ids,
        "enabled": enabled,
    }


def set_canvas_start(model, canvas_start: int) -> int:
    """Tell every bidirectional DeltaNet where the editable canvas begins."""
    n = 0
    for layer in model.layers:
        mod = getattr(layer, "linear_attn", None)
        if isinstance(mod, BidirectionalGatedDeltaNet):
            mod.set_canvas(canvas_start)
            n += 1
    return n


def set_bidirectional(model, enabled: bool) -> int:
    """Runtime switch between the v0.2 bidirectional path and the v0.1 causal path.

    This is what makes ablation A vs B a matched comparison: identical weights,
    identical framework, one boolean.
    """
    n = 0
    for layer in model.layers:
        mod = getattr(layer, "linear_attn", None)
        if isinstance(mod, BidirectionalGatedDeltaNet):
            mod.bidirectional = enabled
            n += 1
    return n


def deltanet_modules(model) -> list[tuple[int, BidirectionalGatedDeltaNet]]:
    out = []
    for i, layer in enumerate(model.layers):
        mod = getattr(layer, "linear_attn", None)
        if isinstance(mod, BidirectionalGatedDeltaNet):
            out.append((i, mod))
    return out
