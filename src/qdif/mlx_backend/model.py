"""`MLXDiffusionQwen` -- the v0.2 diffusion model.

Same contract as the v0.1 torch `DiffusionQwen`:

    sequence layout   [ prefix (clean, causal) | canvas (noisy) ]
    readout           logits[:, i] predicts x0[i]  -- NO shift
    AR control        adapters toggle off to restore the base model exactly

What is new in v0.2:

  * full-attention layers get the v0.1 bidirectional canvas mask (unchanged), and
  * DeltaNet layers additionally run the *same* pretrained recurrence in reverse over
    the canvas and fuse the two directions (`deltanet.py`).

We re-implement the text-model forward loop rather than calling
`Qwen3_5TextModel.__call__`, because we need to supply our own per-layer-type masks
and tell each DeltaNet where the canvas begins. `tests/test_mlx_diffusion_model.py`
asserts that this loop reproduces upstream's forward exactly when the canvas is
disabled, so drift in mlx-lm fails loudly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..config import DiffusionConfig
from .deltanet import BidirectionalGatedDeltaNet, set_canvas_start
from .masks import build_attention_mask


@dataclass
class DiffusionForwardOutput:
    logits: mx.array  # [B, C, V] over canvas positions only
    hidden_states: mx.array  # [B, C, H]
    canvas_start: int
    total_length: int


def sinusoidal_features(t: mx.array, dim: int, max_period: float = 1.0e4) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None] * 1000.0
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1))], axis=-1)
    return emb


class TimestepConditioner(nn.Module):
    """Additive noise-level bias on canvas hidden states. Zero-initialised output,
    so at init the diffusion path is numerically identical to the AR path.

    NORM BOUND -- added in Act III after this module caused a real failure.

    In the first Act III run the bias grew to norm ~13-15 while Qwen3.5-4B token
    embeddings have norm ~0.66, a 20x ratio. Because the bias is *added* to
    `inputs_embeds` at canvas positions, it swamped the token embeddings and the
    canvas contents became irrelevant: the canvas-conditioning probe collapsed from
    1.517 to 0.016 within 125 steps. The learned bias was also nearly identical at
    t=0.1/0.5/0.9, i.e. it was not encoding the timestep at all -- it had found a
    degenerate "add a large constant to canvas positions" solution.

    `max_relative_norm` hard-clips the bias norm to a fraction of the mean token
    embedding norm, so this failure mode cannot recur silently. Set it to 0 to
    disable the bound (the Acts I/II behaviour).

    Note that under absorbing-state (mask) corruption this module is arguably
    unnecessary: the noise level is directly observable from the number of [MASK]
    tokens in the canvas, and FLARE's formulation carries no timestep embedding.
    Act III therefore runs with `timestep_conditioning: false`.
    """

    def __init__(self, hidden_size: int, feature_dim: int = 256,
                 max_relative_norm: float = 0.5, reference_norm: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_relative_norm = max_relative_norm
        self.reference_norm = reference_norm
        self.fc1 = nn.Linear(feature_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2.weight = mx.zeros_like(self.fc2.weight)
        self.fc2.bias = mx.zeros_like(self.fc2.bias)

    def __call__(self, t: mx.array) -> mx.array:
        raw = self.fc2(nn.silu(self.fc1(sinusoidal_features(t, self.feature_dim))))[:, None, :]
        if self.max_relative_norm and self.max_relative_norm > 0:
            budget = self.max_relative_norm * self.reference_norm
            norm = mx.sqrt(mx.sum(raw.astype(mx.float32) ** 2, axis=-1, keepdims=True))
            scale = mx.minimum(1.0, budget / mx.maximum(norm, 1e-6))
            raw = raw * scale.astype(raw.dtype)
        return raw


class MLXDiffusionQwen(nn.Module):
    def __init__(self, base_model, cfg: DiffusionConfig, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg
        # The norm bound is relative to the model's own mean token-embedding norm,
        # measured once at construction -- see TimestepConditioner's docstring.
        ref = 1.0
        try:
            emb = base_model.language_model.model.embed_tokens.weight
            ref = float(mx.linalg.norm(emb.astype(mx.float32), axis=-1).mean())
        except (AttributeError, RuntimeError):
            pass
        self.embedding_reference_norm = ref
        self.timestep_conditioner = (
            TimestepConditioner(hidden_size, reference_norm=ref)
            if cfg.timestep_conditioning else None
        )

    # ------------------------------------------------------------------ helpers
    @property
    def layers(self):
        return self.base_model.layers

    @property
    def text_model(self):
        return self.base_model.language_model.model

    @property
    def args(self):
        return self.base_model.language_model.args

    def embed(self, ids: mx.array) -> mx.array:
        return self.text_model.embed_tokens(ids)

    def readout(self, hidden: mx.array) -> mx.array:
        lm = self.base_model.language_model
        if self.args.tie_word_embeddings:
            return lm.model.embed_tokens.as_linear(hidden)
        return lm.lm_head(hidden)

    # ------------------------------------------------------------------ forward
    def _run_layers(
        self, hidden: mx.array, canvas_start: int, bidirectional_attention: bool
    ) -> mx.array:
        """The text-decoder loop, with our masks and canvas boundary."""
        T = hidden.shape[1]
        fa_mask = (
            build_attention_mask(T, canvas_start, hidden.dtype, bidirectional_attention)
            if T > 1
            else None
        )
        set_canvas_start(self.base_model, canvas_start)

        for layer in self.layers:
            if layer.is_linear:
                # DeltaNet: no attention mask (padding only, and we have no padding).
                hidden = layer(hidden, mask=None, cache=None)
            else:
                hidden = layer(hidden, mask=fa_mask, cache=None)
        return self.text_model.norm(hidden)

    def __call__(
        self,
        canvas_ids: mx.array,
        t: mx.array,
        prefix_ids: mx.array | None = None,
        bidirectional_attention: bool | None = None,
    ) -> DiffusionForwardOutput:
        if canvas_ids.ndim != 2:
            raise ValueError(f"canvas_ids must be [B, C], got {canvas_ids.shape}")
        B, C = canvas_ids.shape

        if prefix_ids is not None and prefix_ids.size:
            input_ids = mx.concatenate([prefix_ids, canvas_ids], axis=1)
            P = prefix_ids.shape[1]
        else:
            input_ids = canvas_ids
            P = 0
        T = input_ids.shape[1]

        hidden = self.embed(input_ids)
        if self.timestep_conditioner is not None:
            bias = self.timestep_conditioner(t.reshape(-1)).astype(hidden.dtype)
            hidden = mx.concatenate(
                [hidden[:, :P], hidden[:, P:] + bias], axis=1
            ) if P else hidden + bias

        bidi_attn = (
            self.cfg.bidirectional_canvas
            if bidirectional_attention is None
            else bidirectional_attention
        )
        out = self._run_layers(hidden, canvas_start=P, bidirectional_attention=bidi_attn)
        canvas_hidden = out[:, P:]
        return DiffusionForwardOutput(
            logits=self.readout(canvas_hidden),
            hidden_states=canvas_hidden,
            canvas_start=P,
            total_length=T,
        )

    def ar_forward(self, input_ids: mx.array) -> mx.array:
        """The untouched autoregressive path: causal mask, no canvas, next-token logits."""
        from .deltanet import set_bidirectional

        previous = [m.bidirectional for _, m in _bidi(self)]
        set_bidirectional(self.base_model, False)
        try:
            hidden = self.embed(input_ids)
            out = self._run_layers(hidden, canvas_start=-1, bidirectional_attention=False)
            return self.readout(out)
        finally:
            for (_, m), p in zip(_bidi(self), previous):
                m.bidirectional = p


def _bidi(model) -> list[tuple[int, BidirectionalGatedDeltaNet]]:
    out = []
    for i, layer in enumerate(model.layers):
        mod = getattr(layer, "linear_attn", None)
        if isinstance(mod, BidirectionalGatedDeltaNet):
            out.append((i, mod))
    return out


def parameter_report(model) -> dict:
    """Total / trainable / by-family parameter counts and byte sizes."""
    from mlx.utils import tree_flatten

    all_params = tree_flatten(model.parameters())
    trainable = tree_flatten(model.trainable_parameters())

    total = sum(v.size for _, v in all_params)
    nbytes = sum(v.nbytes for _, v in all_params)
    train_n = sum(v.size for _, v in trainable)
    train_bytes = sum(v.nbytes for _, v in trainable)

    def family(path: str) -> str:
        if path.endswith(("lora_a", "lora_b")):
            return "lora"
        if ".fusion." in path:
            return "fusion"
        if path.startswith("timestep_conditioner"):
            return "timestep_conditioner"
        return "base"

    by_family: dict[str, int] = {}
    for path, v in trainable:
        by_family[family(path)] = by_family.get(family(path), 0) + v.size

    return {
        "total_params": total,
        "trainable_params": train_n,
        "frozen_params": total - train_n,
        "percent_trainable": 100.0 * train_n / total if total else 0.0,
        "trainable_by_family": by_family,
        "param_bytes": nbytes,
        "param_gb": nbytes / 1e9,
        "trainable_bytes": train_bytes,
        "trainable_mb": train_bytes / 1e6,
    }
