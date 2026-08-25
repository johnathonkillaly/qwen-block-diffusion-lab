# Migrating from Qwen3.5-0.8B/4B to Qwen3.8-27B

Written after reading the actual local Qwen3.8-27B config
(`models--lmstudio-community--Qwen3.8-27B-MLX-8bit`, a local MLX export) side by side
with `Qwen/Qwen3.5-0.8B`. Nothing here is extrapolated from a model card.

## The headline

**Qwen3.8-27B declares `model_type: qwen3_5` and `architectures:
["Qwen3_5ForConditionalGeneration"]`.** It is the *same architecture family*, dense
(not MoE), scaled up. Almost all of this harness carries over unchanged.

## Side by side

| | Qwen3.5-0.8B | Qwen3.8-27B | Impact |
|---|---|---|---|
| `model_type` | `qwen3_5` | `qwen3_5` | none — same code path |
| Architecture | `Qwen3_5ForConditionalGeneration` | same | none |
| Layers | 24 | 64 | none (all layer logic is index-driven) |
| Hidden / intermediate | 1024 / 3584 | 5120 / 17408 | none |
| `layer_types` | 18 L / 6 F | **48 L / 16 F** | none — same 3:1, `full_attention_interval` 4 |
| Full-attention indices | 3,7,…,23 | 3,7,…,63 | `lora.layer_indices` values change |
| Vocabulary | 248,320 | **248,320** | none — identical tokenizer |
| Attention heads / KV | 8 / 2 | 24 / 4 | none |
| `head_dim` | 256 | 256 | none |
| `attn_output_gate` | True | True | none |
| DeltaNet key heads | 16 | 16 | none |
| DeltaNet **value** heads | **16** | **48** | **see below — a real gap in test coverage** |
| DeltaNet k/v head dim | 128 / 128 | 128 / 128 | none |
| `linear_conv_kernel_dim` | 4 | 4 | none |
| **`tie_word_embeddings`** | **True** | **False** | **behavioural difference, see below** |
| MTP layers | 1 | 1 | none |
| RoPE | mRoPE [11,11,10], theta 1e7, partial 0.25 | identical | none |
| Max positions | 262,144 | 262,144 | none |
| Vision tower depth | 12 | 27 | none — never loaded |

## Code that carries over unchanged

- `models/masks.py` — the mask dict is keyed by layer *type*, not index or count.
- `models/qwen35_adapter.py` — `Qwen3_5ForCausalLM` + text-only loading is identical.
- All of `diffusion/` — no architecture dependency at all.
- `training/lora.py` — module *names* are identical (`self_attn.q_proj`,
  `linear_attn.in_proj_qkv`, …), so every target preset resolves.
- `data/`, `eval/`, `inference/` — unaffected.

`lora.layer_indices` is the only config field whose *values* are model-specific, and
leaving it empty (adapt every matching module) is model-agnostic.

## Two real differences

### 1. Untied embeddings

Qwen3.5-0.8B ties `lm_head.weight` to `embed_tokens.weight`. **Qwen3.8-27B does not** —
the checkpoint carries a separate `lm_head.weight`.

Consequences:

- **Self-conditioning's justification changes.** `SelfConditioner` computes an expected
  token embedding against the *input* embedding matrix. At 0.8B that matrix is also
  the readout basis, which is why the representation was chosen. At 27B the two are
  different spaces. The module still runs and is still reasonable, but the argument for
  it is weaker, and using `lm_head.weight` instead becomes a real alternative worth
  testing. `DiffusionQwen.embedding_matrix` is the single place to change.
- **`lm_head` becomes a legitimate LoRA target.** At 0.8B, adapting it would also
  perturb the input embeddings, so every preset avoids it. At 27B that coupling is
  gone, and since the readout convention is exactly what we are asking the model to
  relearn, a LoRA on the (now independent) `lm_head` is arguably the *most* targeted
  intervention available. It is not in any preset today.
- `inspect_architecture` already reports `tie_word_embeddings` and
  `lm_head_shares_embedding` separately, so this shows up in `qdif arch-report` rather
  than surprising someone.

### 2. DeltaNet value-head replication is untested at 0.8B

At 0.8B, `linear_num_value_heads == linear_num_key_heads == 16`, so
`num_v_heads // num_k_heads == 1` and this branch in `Qwen3_5GatedDeltaNet.forward` is
**dead**:

```python
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key   = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
```

At 27B the ratio is **3**, so it runs. Nothing in this harness depends on it, but it
means the 0.8B development loop does not exercise that code path at all. This is the
clearest example of the general risk: *0.8B does not cover everything 27B does*. The
mitigation is cheap — `qdif smoke-forward` and `qdif probe-bidir` on the first 27B load,
before any training.

Qwen3.5-4B sits in between; check its `linear_num_value_heads` when the checkpoint is
fetched.

## Migration blockers

1. **No HF-format bf16 27B checkpoint on this machine.** Every local Qwen3.8-27B copy
   is an MLX or GGUF *export* (4-bit / 8-bit / mxfp8 / NVFP4). Their tensor keys differ
   (`language_model.lm_head.weight`, no `mtp.*`), and quantized exports are not torch
   training backbones. Migration therefore requires either a ~55 GB bf16 download or
   the MLX training backend.
2. **QLoRA is unavailable on Apple silicon** (bitsandbytes is CUDA-only, see
   [MEMORY.md](MEMORY.md)). At 27B this stops being an inconvenience and becomes the
   deciding constraint — which makes the MLX backend a *prerequisite* for the 27B
   experiment on this hardware, not an optimisation.
3. **The scientific blocker is upstream of all of this.** Experiment 001 shows the
   objective converging to copying at 0.8B. Scaling that to 27B would buy a more
   expensive copier. Fix the objective first.

## Memory at 27B (estimates, `qdif memory`)

Roughly, with ~27e9 parameters:

| Configuration | Parameters | Optimizer | Approx total |
|---|---|---|---|
| BF16 inference | ~54 GB | — | ~57 GB |
| 8-bit inference | ~30 GB | — | ~33 GB |
| 4-bit inference | ~16 GB | — | ~19 GB |
| BF16 full training | ~54 GB | ~216 GB | **infeasible here** |
| BF16 base + LoRA | ~54 GB | ~0.5 GB | ~60 GB — **fits in 103 GB working set** |
| 4-bit base + MLX LoRA | ~16 GB | ~0.5 GB | ~22 GB — comfortable |

So a bf16 backbone + diffusion LoRA is feasible on this machine, and a 4-bit MLX
backbone + LoRA is comfortable. Full-weight conversion at 27B is not, which is exactly
why `training.mode: full` is scoped at the small checkpoints (0.8B, and 4B if
affordable).

## Recommended order

1. Fix the objective at 0.8B (experiments 004+ in [EXPERIMENTS.md](EXPERIMENTS.md)).
2. Reproduce whatever works at Qwen3.5-4B-Base. Confirms it is not a 0.8B artefact.
3. Implement the MLX training backend with the torch-equivalence test.
4. `qdif arch-report` + `smoke-forward` + `probe-bidir` on 27B **before** any training —
   this is where the value-head replication path and untied embeddings first execute.
5. Only then, LoRA at 27B.

## Things to avoid that would only work at small scale

- Do not hard-code layer indices anywhere outside a config file.
- Do not assume tied embeddings (already avoided; `parameter_report` and the
  checkpoint writer both handle either case).
- Do not assume the whole model fits in memory twice; the AR control is implemented as
  an adapter *toggle* on one model instance, not a second copy, and should stay that way.
- Do not add a `full`-mode code path that only works because gradients for a 0.8B model
  are cheap.
