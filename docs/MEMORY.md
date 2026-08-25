# Memory

Run `qdif inspect` or `qdif memory` for live numbers. This document explains what the
numbers mean and what is actually implemented.

## Unified memory is not VRAM

This machine is an **Apple M4 Max with 128 GB of unified memory**, shared between CPU
and GPU. The harness reports it as unified memory everywhere
(`training/diagnostics.py::memory_label`), because calling it VRAM produces two
specific mistakes:

1. It is shared with the OS and every other process, so the whole 128 GB is never
   available.
2. Metal caps a process at `recommendedMaxWorkingSetSize`, typically **~75% of
   installed RAM** (~103 GB here). `qdif inspect` prints this as the practical GPU
   ceiling, separately from the installed total.

`torch.mps.driver_allocated_memory()` is what "peak memory" means on MPS; on CUDA it
is `torch.cuda.max_memory_allocated()`; on CPU it is process RSS. The unit is always
labelled in the output.

## The five consumers

Lumping these together is how people conclude a 27B LoRA run fits in 32 GB.

| Consumer | Scales with | Notes |
|---|---|---|
| **Parameters** | model size x weight dtype | Quantized weights carry group scales/zero-points; the estimator adds 10% (int8) / 15% (int4). |
| **Gradients** | *trainable* parameters x compute dtype | LoRA makes this ~0. This is the whole point. |
| **Optimizer state** | *trainable* parameters x 8 bytes | AdamW keeps two fp32 moments. At full-weight training this is the single largest line. |
| **Activations** | batch x seq x hidden x layers | ~2 hidden-sized buffers/layer at inference, ~12 for backward without checkpointing. Plus the logits tensor, which is **not** negligible at a 248,320 vocabulary. |
| **Runtime state** | KV cache + DeltaNet state | Inference only; see below. |

### Hybrid runtime state

The 3:1 hybrid changes the shape of the cache in a way worth stating explicitly:

- **Full-attention layers** hold a KV cache that grows linearly with sequence length —
  but there are only 6 of 24 (0.8B) or 16 of 64 (27B).
- **DeltaNet layers** hold a *fixed-size* state regardless of sequence length: a
  `[2*k_dim + v_dim, conv_kernel]` convolution window plus a
  `[v_heads, k_head_dim, v_head_dim]` recurrent matrix, in fp32
  (`mamba_ssm_dtype: float32`).

So a hybrid has roughly a quarter of the long-context cache of a same-depth pure
attention model, and the DeltaNet portion is a constant. Both are modelled in
`env/memory.py::runtime_state_bytes`.

## Measured: Qwen3.5-0.8B, bf16 base + LoRA r=16, canvas 32, batch 2, MPS

| | |
|---|---|
| Total parameters | 754,787,136 |
| Trainable | 2,394,112 (**0.3172%**) |
| — LoRA (24 modules, q/k/v/o of 6 full-attention layers) | 1,081,344 |
| — Auxiliary (timestep conditioner) | 1,312,768 |
| Parameter storage | 1.51 GB |
| **Peak training memory** | **3.48 GB unified** (3.63 GB with `full_attn_mlp`, r=32) |
| Throughput | ~200 tok/s (r=16), ~155 tok/s (r=32 + MLP) |

The estimator predicted 2.58 GB for `BF16 base + LoRA` at seq 512 / batch 1; the
measured 3.48 GB at batch 2 is consistent given the activation rule of thumb. The
estimator is a planning tool, not a substitute for measurement.

## Estimated profiles

`qdif memory` reports six configurations. What is actually implemented:

| Configuration | Status |
|---|---|
| BF16 inference | works |
| 8-bit inference | works via MLX exports (`mlx-community/Qwen3.5-0.8B-MLX-4bit` etc. are on SHUTTLE); not a torch training path |
| 4-bit inference | same |
| BF16 full training (`training.mode: full`) | implemented, **not** the default; optimizer state dominates |
| **BF16 base + LoRA** | **implemented and measured** — the default |
| 4-bit base + LoRA (QLoRA) | **not implemented on Apple silicon** |

## Why QLoRA is not available here

`bitsandbytes` — the usual 4-bit NF4 + paged-optimizer path, and what Unsloth builds
on — is **CUDA-only**. There is no Apple-silicon build. `training.quantization` other
than `none` therefore raises with that explanation rather than silently ignoring the
setting.

The Apple-native route is MLX quantization (`mx.quantize`, affine, group size 64),
which is why the MLX backend matters more for the 27B target than for the 0.8B one.
That is a real dependency of the migration plan, not a nice-to-have — see
[QWEN38_MIGRATION.md](QWEN38_MIGRATION.md).

On CUDA hardware, bitsandbytes/Unsloth-style QLoRA is the intended path and the
`training.quantization` seam exists for it.

## NVFP4 (future, NVIDIA Blackwell)

Documented as a **future backend**, deliberately excluded from this implementation.

- NVFP4 is an NVIDIA Transformer Engine format for Blackwell-class hardware. It has no
  bearing on Apple silicon and none of this machine's work depends on it.
- The constraint that matters: **the diffusion algorithm must remain
  precision-agnostic.** Nothing under `src/qdif/diffusion/` may import a precision
  backend or branch on dtype beyond the fp32 loss upcast. If NVFP4 support is added it
  belongs behind the `training.backend` / `training.quantization` seams, alongside a
  numerical-equivalence test against the bf16 reference on the same seeded batch.
- Local `Qwen3.8-27B-NVFP4-MTP-GGUF` exports exist on SHUTTLE; they are inference
  artefacts and not training backbones.

## Rules

- Never commit weights, checkpoints, datasets, caches or virtualenvs. `.gitignore`
  covers `*.safetensors`, `*.gguf`, `runs/`, `checkpoints/`, `.venv/`.
- Adapter checkpoints contain **only** LoRA factors and auxiliary modules — a few MB.
  `tests/test_model_integration.py::test_checkpoint_save_and_reload` asserts no base
  weight ever lands in one.
- Model files on `SHUTTLE` are read-only to this project. Nothing here writes to or
  deletes an existing model directory.
