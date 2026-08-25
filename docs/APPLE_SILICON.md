# Running a diffusion-LLM transfer experiment on Apple Silicon

What it actually takes, measured rather than estimated. Numbers below are from an
M4 Max / 128 GB, but the guidance is written so you can size it for a different Mac.

## The headline

Converting a 4B hybrid Gated-DeltaNet model toward block diffusion, with LoRA on
1.3% of parameters, runs on a single Apple Silicon Mac in **~21–23 GB of unified
memory** at **~80 canvas tokens/sec**, with three forward passes per step. A 2,500-step
transfer run takes **under two hours**.

That is a real experiment on real hardware, not a toy. It is also nothing like a
CUDA recipe, and the differences matter.

## Unified memory is not VRAM

Every memory figure in this repository is **unified memory**, shared between CPU and
GPU. Two consequences people get wrong:

1. It is shared with the OS and everything else running, so the whole installed amount
   is never available.
2. Metal caps a process at `recommendedMaxWorkingSetSize`, typically **~75% of
   installed RAM** (~103 GB on a 128 GB machine). `qdif inspect` prints this separately
   from the installed total.

`mx.get_peak_memory()` is what we report. It is not `nvidia-smi`.

## Measured requirements (Qwen3.5-4B-Base, BF16)

| configuration | peak unified | notes |
|---|---|---|
| Load + forward only | 8.4 GB | weights are 8.41 GB bf16 |
| Act II single-forward diffusion training, canvas 32, batch 4 | 11.7–12.2 GB | one forward/step |
| **Act III transfer, canvas 128 + prefix 64, batch 2, 3 forwards/step** | **21–23 GB** | the main run |
| Act III milestone check (extra probe allocations) | 23.1 GB | transient |

### Sizing for a smaller Mac

| machine | what fits |
|---|---|
| 16 GB | 4B inference only, or training a 0.8B model. The Act III recipe will not fit. |
| 24 GB | Act III at `canvas_length: 48`, `batch_size: 1`, `complementary_views: false` (2 forwards/step). Tight. |
| 32 GB | Act III at `canvas_length: 64`, `batch_size: 1`. Comfortable. |
| 36–48 GB | Act III at `canvas_length: 128`, `batch_size: 1–2`. |
| 64 GB+ | the shipped `configs/act3_main.yaml` unchanged. |

The knobs that move memory most, in order: `canvas_length` (sequence length dominates
activation memory), `batch_size`, `complementary_views` (drops one of three forwards),
and `lora.rank`. Reducing `lora.mlp` to `false` saves parameters but was part of what
made Act III healthy — cut sequence length first.

## Software stack

| component | version used | notes |
|---|---|---|
| macOS | Darwin 27.0 (arm64) | |
| Python | 3.13 | |
| `mlx` / `mlx-metal` | 0.32.1 | MIT |
| `mlx-lm` | 0.31.3 | ships the Qwen3.5 model definition |
| `unsloth` | 2026.8.21 | Apache-2.0 |
| `unsloth_zoo` | 2026.8.15 | licence discrepancy — see [THIRD_PARTY.md](../THIRD_PARTY.md) |
| `transformers` | 5.5.0 | pinned *down* by unsloth; only Acts I/II use it |
| `torch` | 2.11.0 | pinned down by unsloth; Acts I/II only |

**Two virtual environments are required**, and this is an upstream constraint rather
than a choice: Unsloth pins `transformers <= 5.5.0`, while the Act I/II torch backend
needs `transformers >= 5.8` for the dict-keyed attention mask. See
[UNSLOTH_BACKEND.md](UNSLOTH_BACKEND.md).

```
.venv           torch 2.13 + transformers 5.15   ->  Acts I/II (torch/MPS)
.venv-unsloth   unsloth + mlx                    ->  Act II v0.2 / Act III (MLX)
```

## What Unsloth actually does here

On Apple Silicon, **Unsloth is an MLX stack**: `unsloth.device_type.DEVICE_TYPE`
reports `"mlx"`, and `unsloth_zoo`'s darwin-arm64 dependency set is mlx / mlx-lm /
mlx-vlm, explicitly *excluding* torch, peft, trl and accelerate.

The component that matters for this project is
`unsloth_zoo.gated_delta_vjp.patch_gated_delta()` — a memory-efficient custom VJP for
**Qwen3.5's** Gated DeltaNet with hand-written Metal chunk kernels, which recomputes
recurrent states during backward instead of holding all T intermediates in the graph.
Every run header prints whether it activated:

```
Unsloth: Patched GatedDeltaNet with memory-efficient custom VJP.
[setup] unsloth gated-delta VJP: active (gated_delta_ops now from mlx_lm.models.gated_delta)
```

It is **optional**. Without Unsloth installed the harness reports
`unavailable (ImportError: ...)` and falls back to mlx-lm's own `gated_delta_update`.
Verified, not assumed.

### What never activates on Apple Silicon

| | why |
|---|---|
| Triton kernels | no macOS wheels |
| xformers | no macOS wheels |
| bitsandbytes 4-bit / 8-bit / paged optimizers | CUDA-only library |
| Flash Attention | CUDA |
| Unsloth's torch-autograd gradient checkpointing | it is a torch construct; MLX has no equivalent hook |

**This is why Act III is BF16 base + BF16 LoRA and not QLoRA.** On CUDA you would
reach for 4-bit NF4 and paged AdamW; here you cannot, so you buy headroom with
sequence length and batch size instead. On a 128 GB machine that is not a hardship;
on 16–24 GB it is the binding constraint.

Also note `mlx_lm` selects `use_kernel = not self.training`, so the fused inference
kernel is bypassed during training in favour of the differentiable ops path — which is
precisely the path Unsloth's custom VJP makes efficient.

## Throughput

| | value |
|---|---|
| Seconds per step (batch 2, canvas 128, 3 forwards) | ~2.7 s |
| Canvas tokens/sec | ~82 |
| Sequence tokens/sec | ~135 |
| 2,500-step run | ~110 min |
| Act II single-forward, canvas 32 | 191 tok/s (causal), 151 (bidirectional) |

Three forward passes per step is the price of FLARE's construction: one clean causal
pass for `L_AR`, and two noisy passes for the complementary mask views. Setting
`complementary_views: false` drops to two and roughly quarters the diffusion cost, at
the price of supervising only half the canvas per step.

## How this differs from a CUDA / Blackwell recipe

| | CUDA (e.g. FLARE's own setup) | here |
|---|---|---|
| Precision | 4-bit NF4 QLoRA or full BF16 with ZeRO | BF16 base + BF16 LoRA only |
| Attention kernels | FlashAttention / FlashInfer | MLX SDPA |
| GDN kernels | fused Route I/II chunk kernels | mlx-lm ops path + Unsloth custom VJP |
| Optimizer | paged / 8-bit AdamW | plain MLX AdamW |
| Memory model | discrete VRAM, explicit transfers | unified, no transfers |
| Scaling | multi-GPU, TorchTitan/SGLang | one machine |
| Gradient checkpointing | standard | not available in MLX the same way |

The practical upshot: **Apple Silicon is a fine place to *understand* this class of
experiment and a poor place to *scale* it.** You get a single large-memory device with
no host↔device transfers, which is genuinely pleasant for a 4B model, but you give up
quantized training, fused attention kernels, and any multi-device story.

## Reproducing

```bash
./scripts/reproduce_smoke.sh
```

~5 minutes: capability report, the 15 milestone checks, a 30-step transfer run, and a
diffusion trace. The full run is one command and is documented in the README.

Every run directory contains `run_metadata.json` with package versions, git commit and
dirtiness, config, dataset revision, seeds and hardware — so a `runs/` folder found
later is self-describing.
