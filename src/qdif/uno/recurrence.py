"""A Gated DeltaNet forward that exposes per-token state, inside one model pass.

## Why this file exists

`mlx_lm`'s `gated_delta_update` runs the whole `T`-step recurrence inside a single
fused Metal kernel and returns only the *final* state. Speculative decoding needs the
state at an arbitrary accepted prefix, so Act IV-U had to rewind to the last committed
frontier and **replay the accepted tokens through the entire model** — a third forward
per rejected block, which is precisely what capped Act IV-U at TPF 0.980.

The fix is not to avoid the recurrence but to *unroll* it. Everything expensive — the
input projections, the depthwise convolution, attention, every MLP — still runs
**once** for the whole block. Only the recurrence itself is stepped token by token, and
the state is captured at each boundary.

`docs/act4u2_transactional_state.md` has the measured state anatomy and the derivation.

## Fidelity

`gated_delta_update` is called per step rather than reimplemented, so `β = sigmoid(b)`,
`g = exp(-exp(A_log)·softplus(a+dt_bias))`, the `Hv/Hk` head repeat and the kernel
itself are upstream's, unchanged. The state round-trips through fp32 between steps,
which is the same dtype the fused kernel keeps it in.
`tests/test_uno_recurrence.py` asserts this path reproduces the stock layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.gated_delta import gated_delta_update


@dataclass
class LayerRecord:
    """Per-token state captured from one DeltaNet layer during a block forward."""

    #: `states[j]` is the recurrent state after consuming `j` of the block's tokens,
    #: so `states[0]` is the pre-block state and `len(states) == S + 1`.
    states: list[mx.array] = field(default_factory=list)
    #: `[conv_state_before ; qkv]`, `[B, (kernel-1)+S, conv_dim]`. The conv state after
    #: `j` tokens is the slice `[:, j : j + kernel - 1, :]`.
    conv_input: mx.array | None = None
    #: Per-token factors for algebraic rewind. Only populated when requested.
    factors: list[dict] = field(default_factory=list)

    def state_at(self, step: int) -> mx.array:
        return self.states[step]

    def conv_state_at(self, step: int, keep: int) -> mx.array:
        if self.conv_input is None:
            raise RuntimeError("conv_input was not recorded")
        return mx.contiguous(self.conv_input[:, step : step + keep, :])


@dataclass
class BlockRecord:
    """Everything captured from one block forward, keyed by layer index."""

    layers: dict[int, LayerRecord] = field(default_factory=dict)
    #: KV-cache offset before the block, keyed by layer index.
    kv_offset_before: dict[int, int] = field(default_factory=dict)
    block_length: int = 0

    @property
    def nbytes(self) -> int:
        total = 0
        for record in self.layers.values():
            total += sum(s.nbytes for s in record.states)
            if record.conv_input is not None:
                total += record.conv_input.nbytes
            for factor in record.factors:
                total += sum(v.nbytes for v in factor.values())
        return total


def deltanet_block_forward(attn, inputs: mx.array, cache, record: LayerRecord,
                           keep_factors: bool = False) -> mx.array:
    """`GatedDeltaNet.__call__`, with the recurrence unrolled and recorded.

    Mirrors `mlx_lm.models.qwen3_5.GatedDeltaNet.__call__` step for step. `mask` is
    omitted deliberately: this path is only used for unpadded single-sequence
    speculative blocks, and a padding mask would change what the recorded states mean.
    """
    B, S, _ = inputs.shape

    qkv = attn.in_proj_qkv(inputs)
    z = attn.in_proj_z(inputs).reshape(B, S, attn.num_v_heads, attn.head_v_dim)
    b = attn.in_proj_b(inputs)
    a = attn.in_proj_a(inputs)

    keep = attn.conv_kernel_size - 1
    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, keep, attn.conv_dim), dtype=inputs.dtype)

    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    record.conv_input = conv_input
    if cache is not None:
        cache[0] = mx.contiguous(conv_input[:, -keep:, :])
    conv_out = nn.silu(attn.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [attn.key_dim, 2 * attn.key_dim], -1),
            [attn.num_k_heads, attn.num_k_heads, attn.num_v_heads],
            [attn.head_k_dim, attn.head_k_dim, attn.head_v_dim],
        )
    ]

    state = cache[1] if cache else None
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    if state is None:
        state = mx.zeros(
            (B, attn.num_v_heads, attn.head_v_dim, attn.head_k_dim), dtype=mx.float32
        )

    record.states = [state]
    outputs = []
    for step in range(S):
        out_t, state = gated_delta_update(
            q[:, step : step + 1],
            k[:, step : step + 1],
            v[:, step : step + 1],
            a[:, step : step + 1],
            b[:, step : step + 1],
            attn.A_log,
            attn.dt_bias,
            state,
            None,
            use_kernel=not attn.training,
        )
        outputs.append(out_t)
        record.states.append(state)
        if keep_factors:
            record.factors.append(
                {
                    "k": k[:, step],
                    "v": v[:, step],
                    "a": a[:, step],
                    "b": b[:, step],
                }
            )

    out = mx.concatenate(outputs, axis=1)
    if cache is not None:
        cache[1] = state
        cache.advance(S)

    out = attn.norm(out, z)
    return attn.out_proj(out.reshape(B, S, -1))


def text_model_block_forward(
    model, input_ids: mx.array, caches, keep_factors: bool = False
) -> tuple[mx.array, BlockRecord]:
    """One full model forward that records every DeltaNet layer's per-token state.

    Replicates `Qwen3_5TextModel.__call__`. The attention layers are untouched — their
    cache is a plain append and needs only an offset rewind (§8 of the state doc).
    """
    text = model.language_model.model
    layers = text.layers
    record = BlockRecord(block_length=int(input_ids.shape[1]))

    hidden = text.embed_tokens(input_ids)
    attention_mask = create_attention_mask(hidden, caches[text.fa_idx])

    for index, (layer, cache) in enumerate(zip(layers, caches)):
        if layer.is_linear:
            # `create_ssm_mask` returns None unless the cache carries padding metadata.
            # This path records per-token state, and a padding mask would silently
            # change what those states mean, so refuse rather than mis-record.
            if getattr(cache, "lengths", None) is not None or (
                getattr(cache, "left_padding", None) is not None
            ):
                raise NotImplementedError(
                    "transactional recording does not support padded batches"
                )
            layer_record = LayerRecord()
            record.layers[index] = layer_record
            residual = hidden
            normed = layer.input_layernorm(hidden)
            attn_out = deltanet_block_forward(
                layer.linear_attn, normed, cache, layer_record, keep_factors
            )
            hidden = residual + attn_out
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        else:
            record.kv_offset_before[index] = int(cache.offset)
            hidden = layer(hidden, mask=attention_mask, cache=cache)

    hidden = text.norm(hidden)
    args = model.language_model.args
    if args.tie_word_embeddings:
        logits = text.embed_tokens.as_linear(hidden)
    else:
        logits = model.language_model.lm_head(hidden)
    return logits, record
