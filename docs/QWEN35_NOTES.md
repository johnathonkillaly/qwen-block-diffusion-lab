# Qwen3.5 architecture findings

Everything here was read off the installed `transformers` implementation and the
local `Qwen/Qwen3.5-0.8B` checkpoint, not from documentation or assumption.
Regenerate with `qdif arch-report`.

Sources: `transformers/models/qwen3_5/modular_qwen3_5.py`,
`transformers/models/qwen3_next/modeling_qwen3_next.py` (Qwen3.5 inherits its token
mixers from Qwen3-Next), `transformers 5.15.1`.

## Class hierarchy

```
Qwen3_5ForConditionalGeneration      multimodal wrapper (what the checkpoint declares)
└── Qwen3_5Model                     (Qwen3VLModel)
    ├── Qwen3_5VisionModel           vision tower  <-- never instantiated by us
    └── Qwen3_5TextModel             (Qwen3NextModel)
        └── Qwen3_5DecoderLayer x N
            ├── .linear_attn  Qwen3_5GatedDeltaNet   (Qwen3NextGatedDeltaNet)
            │   or
            │   .self_attn    Qwen3_5Attention       (Qwen3NextAttention)
            ├── .mlp          Qwen3_5MLP
            ├── .input_layernorm / .post_attention_layernorm   Qwen3_5RMSNorm

Qwen3_5ForCausalLM                   text-only head  <-- what this project loads
└── model = Qwen3_5TextModel
```

`Qwen3_5ForCausalLM` declares
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`, and
transformers remaps `model.language_model.*` -> `model.*` on load. **Text-only
execution therefore requires no surgery at all**: the vision tower and the MTP head
are simply not instantiated. Verified by
`tests/test_model_integration.py::test_vision_tower_is_not_loaded`.

## Qwen3.5-0.8B measured shape

| | |
|---|---|
| Layers | 24 |
| Hidden / intermediate | 1024 / 3584 |
| Vocabulary | 248,320 |
| Text-decoder parameters | 752,393,024 |
| Attention heads / KV heads | 8 / 2, `head_dim` 256 |
| `attn_output_gate` | **True** — `q_proj` emits `2 * n_heads * head_dim` (query + gate) |
| DeltaNet heads | 16 key / 16 value, k-dim 128, v-dim 128, conv kernel 4 |
| Tied embeddings | **True** (`lm_head.weight` shares storage with `embed_tokens.weight`) |
| RoPE | mRoPE interleaved, sections [11,11,10], theta 1e7, `partial_rotary_factor` 0.25 |
| Max positions | 262,144 |
| MTP | 1 layer in config; weights present in the checkpoint, not loaded |

## The hybrid layer pattern

```
layer:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
type:   L  L  L  F  L  L  L  F  L  L  L  F  L  L  L  F  L  L  L  F  L  L  L  F
                 ^           ^           ^           ^           ^           ^
L = linear_attention (Gated DeltaNet), 18 layers
F = full_attention, 6 layers   -> indices [3, 7, 11, 15, 19, 23]
```

`full_attention_interval = 4`; full attention lands on the **last** layer of each
group of four. The first three layers of the network are purely recurrent.

## The finding that shapes the whole project

> **Gated DeltaNet layers are causal by construction, not by masking. No attention
> mask can make them bidirectional.**

Two independent mechanisms enforce it, both inside `Qwen3NextGatedDeltaNet`:

1. **Causal depthwise convolution.** The mixed QKV stream is passed through
   `causal_conv1d_fn` (or, without the fused kernel,
   `F.silu(self.conv1d(x)[:, :, :seq_len])`, i.e. a left-padded conv with the right
   tail discarded). Position *i* sees only positions *i-3 … i*.
2. **A triangular chunked recurrence.** `torch_chunk_gated_delta_rule` builds its
   intra-chunk interaction with hard-coded `torch.tril` / `torch.triu`, and carries
   state forward across chunks in one direction only.

The `attention_mask` these layers receive is produced by
`create_recurrent_attention_mask` and is a **padding mask** — its only job is zeroing
padded positions (`apply_mask_to_padding_states`). Handing it a bidirectional mask
does nothing at all. It is not a hook.

By contrast, `Qwen3_5Attention` is ordinary gated GQA and consumes whatever additive
4-D mask it is given.

### Consequence for a diffusion canvas

The honest description of what this repository builds is:

> **causal DeltaNet + bidirectional full-attention canvas**

Every canvas position can influence every other canvas position, but only through
the 6 full-attention layers. Information cannot travel backwards through the 18
DeltaNet layers. Layers 0–2 are purely causal even inside the canvas.

This is a real compromise and it is not hidden. `qdif probe-bidir` measures it:

```
[bidirectional]   earlier canvas positions affected: 15/15   max delta 6.09e-01
[causal_control]  earlier canvas positions affected:  0/15   max delta 0.00e+00
```

Perturbing the last canvas token moves every earlier canvas hidden state when the
full-attention mask is relaxed, and moves **nothing** when it is not — which
simultaneously proves the bidirectionality is real and that it comes from the
full-attention layers alone.

### Options that were considered and not taken

| Strategy | Why not (yet) |
|---|---|
| **Adapt only full-attention layers, keep DeltaNet causal** | **Chosen.** Scientifically clean, zero risk of a fake result, and it isolates the variable in research question 4. |
| Two-direction DeltaNet (run forward + reversed, merge) | Doubles recurrent compute, needs a learned merge that does not exist in the checkpoint, and destroys state caching. Worth trying *after* the causal version has a baseline. Would answer research question 3 properly. |
| Separate diffusion adapters wrapping the recurrent blocks | Plausible: a small bidirectional mixer inserted around `linear_attn`. Adds parameters outside the base checkpoint, so it is a different (larger) claim than "LoRA can do it". |
| Rewriting `chunk_gated_delta_rule` to be non-causal | Not a mask change but a different operator; the pretrained weights were never trained for it. Would be faking bidirectionality, not achieving it. |

## The integration point

`Qwen3_5TextModel.forward` accepts `attention_mask` **as a dict keyed by layer
type**:

```python
causal_mask_mapping = {
    "full_attention":   create_causal_mask(**mask_kwargs),
    "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
}
...
attention_mask=causal_mask_mapping[self.config.layer_types[i]]
```

and it uses `attention_mask` directly when it is already a dict. So passing

```python
{"full_attention": <4-D additive mask>, "linear_attention": None}
```

injects a bidirectional canvas with **no monkeypatching, no subclassing, and no
vendored copy of transformers**. This is what `qdif.models.masks.build_diffusion_masks`
produces. It is the single cleanest thing about this architecture for our purposes.

## Tokenizer

- `Qwen2TokenizerFast`, vocabulary 248,320, byte-level BPE.
- `eos_token_id = 248044`. No dedicated mask token, so `mask`-mode corruption borrows
  a token id and is offered only as an experimental alternative.
- Reserved multimodal ids: image 248056, video 248057, vision start/end 248053/248054.
  Uniform corruption samples over the full range and can therefore emit these; at
  248k vocabulary the rate is negligible, and text-only execution ignores them.

## KV / recurrent state behaviour

- Full-attention layers use a standard growing KV cache.
- DeltaNet layers keep a **fixed-size** state: a `[k_dim*2 + v_dim, conv_kernel]` conv
  window plus a `[v_heads, k_head_dim, v_head_dim]` recurrent matrix, held in fp32
  (`mamba_ssm_dtype: float32`). It does not grow with sequence length.
- The practical effect: a 3:1 hybrid has roughly a quarter of the long-context cache
  of a same-depth pure-attention model. `qdif.env.memory.runtime_state_bytes` accounts
  for both.
- **This is why the sampler does not cache.** A committed prefix is causal and could
  legitimately be cached, but rewriting a canvas requires rolling the recurrent state
  back to the block boundary, and getting that subtly wrong corrupts results silently
  rather than loudly. The harness recomputes; optimising this is deferred until the
  numbers are trustworthy.

## MTP

The checkpoint ships `mtp.fc`, one full-attention decoder layer, `pre_fc_norm_embedding`,
`pre_fc_norm_hidden` and `mtp.norm` — an Eagle-style speculative head that consumes
`concat(embed(next_token), hidden)` and reuses the shared output head. It is not
loaded by `Qwen3_5ForCausalLM`.

Whether it is useful for diffusion (research question 11) is genuinely open. The
appealing hypothesis: MTP is already trained to predict a token at a position from a
*hidden state plus a token embedding at that position*, which is structurally much
closer to the diffusion readout (`logits[i] -> x0[i]`) than the base LM head is. That
makes it a candidate initialisation for a denoising head rather than a decoding
trick. Untested.
