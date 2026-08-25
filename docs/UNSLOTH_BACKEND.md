# The Unsloth backend

Everything here is **probed at runtime**, not inferred from a package being
importable. Regenerate with `qdif unsloth-report`.

## Versions (this machine, 2026-08-25)

| | |
|---|---|
| Platform | Darwin 27.0.0 arm64, Apple M4 Max, 137 GB unified memory |
| unsloth | 2026.8.21 |
| unsloth_zoo | 2026.8.15 |
| Unsloth Studio | running on `http://127.0.0.1:8888`, 356 endpoints |
| mlx / mlx-metal | 0.32.1 |
| mlx-lm | 0.31.3 |
| torch | 2.11.0 (pinned down by unsloth from 2.13.0) |
| transformers | 5.5.0 (pinned down by unsloth from 5.15.1) |
| peft / trl | 0.20.0 / 0.24.0 (installed, unused on this path) |
| bitsandbytes | 0.50.1 (installed, **cannot run** — CUDA library) |
| triton / xformers | **not installed, no macOS wheels** |

## The headline finding

> **Unsloth on Apple silicon is an MLX stack, not a torch stack.**

Evidence, all verified:

* `unsloth.device_type.DEVICE_TYPE == "mlx"`, `is_mlx_available() == True`.
* `unsloth/device_type.py` reads `_IS_MLX = is_mlx_available()` and then
  `if not _IS_MLX: import torch` — torch is the *fallback*, not the path.
* `unsloth_zoo`'s dependency metadata for `sys_platform == "darwin" and
  platform_machine == "arm64"` requires `mlx==0.32.1`, `mlx-lm==0.31.3`,
  `mlx-vlm>=0.4.4`, and **excludes** `torch`, `torchao`, `accelerate`, `trl`, `peft`
  and `cut_cross_entropy` (each carries a `sys_platform != "darwin" or
  platform_machine != "arm64"` marker).
* `unsloth_zoo/mlx/` contains a full training stack: `trainer.py` (407 KB),
  `loader.py` (330 KB), `utils.py` (631 KB), `optimizers_quantized.py`, `cce/`.

So "use Unsloth as the training engine on this machine" means "use MLX". This is why
v0.2 moved off the v0.1 torch/MPS path — not a preference, a fact about the library.

## What actually activates for us

| Unsloth component | Status | Evidence |
|---|---|---|
| `unsloth_zoo.gated_delta_vjp.patch_gated_delta()` | **ACTIVE** | prints `Unsloth: Patched GatedDeltaNet with memory-efficient custom VJP.` at load; applied in `mlx_backend/loader.py` |
| Metal chunk kernels for GDN (`_make_gd_chunk_states_kernel`, `_make_gd_chunk_backward_kernel`) | present, used by the patched op | `gated_delta_kernel_supported()` gate in `unsloth_zoo` |
| MLX runtime (Metal GPU, unified memory) | **ACTIVE** | `mx.metal.is_available() == True`, `mx.default_device() == Device(gpu, 0)` |
| `unsloth.FastLanguageModel` loader | **not used** | dispatches into `unsloth_zoo.mlx` SFT loader; we call `mlx_lm.load` underneath it (same materialisation) so we can drive a custom forward |
| `unsloth_zoo.mlx.trainer` | **not used** | it is an SFT trainer; its loss is next-token CE over a chat-formatted batch, which cannot express x0-prediction over a corrupted canvas |
| Unsloth LoRA / `peft` | **not used** | `peft` is excluded on darwin-arm64; we use the hand-written MLX LoRA in `mlx_backend/lora.py`, which we need anyway for layer-index targeting and a runtime enable/disable switch |
| Unsloth gradient checkpointing | **not applicable** | `unsloth_zoo/gradient_checkpointing.py` is a torch-autograd construct. MLX has no equivalent hook, and the GDN custom VJP already does the recompute-in-backward job for the expensive operator |
| Triton kernels, xformers, Flash Attention | **NEVER ACTIVE** | no macOS wheels; not installed |
| bitsandbytes 4-bit / 8-bit / paged optimizers | **NEVER ACTIVE** | CUDA-only. This is why v0.2 is BF16 base + BF16 LoRA |

### The one that matters

`unsloth_zoo/gated_delta_vjp.py` is a memory-efficient custom VJP for **Qwen3.5's**
GatedDeltaNet, written in MLX with hand-written Metal chunk kernels. Its docstring:
it "[replaces] the T-step Python loop … with an `mx.custom_function` that recomputes
states during backward instead of keeping all T intermediate states in the autograd
graph."

That is precisely the operator v0.2 evaluates twice per layer, so it is the single
most relevant Unsloth component for this experiment, and it is Apple-native. It is
applied at load time and the status string is reported in every run header — never
claimed on faith.

## Does the custom forward preserve Unsloth's optimisations?

**Yes, for the GDN VJP.** The patch replaces the `gated_delta` ops *module-level*, and
`mlx_backend/deltanet.py` calls `gated_delta_update` from that module for both the
forward and the reverse pass. Both directions therefore go through the patched,
memory-efficient implementation.

**No, for anything torch-shaped**, because nothing torch-shaped runs here.

Note one interaction worth knowing: mlx-lm selects `use_kernel=not self.training`, so
the fused inference kernel is bypassed during training in favour of the differentiable
ops path — which is the path Unsloth's custom VJP makes efficient. Our
`deltanet_recurrence()` passes `use_kernel=False` when training for the same reason.

## Can Unsloth Studio own this training loop?

**No.** Verified against the running instance's own OpenAPI spec:

* `POST /api/train/start` takes a `TrainingStartRequest` with **72 fields** and a
  **closed** `training_type` enum: `["LoRA/QLoRA", "Full Finetuning", "Continued
  Pretraining"]`.
* There is **no** custom-script, entry-point, hook, callback or custom-loss field.
  The only field matching `custom` is `custom_format_mapping`, which maps dataset
  columns.
* Every knob describes a standard SFT run: dataset format, packing, target modules,
  `train_on_completions`, `finetune_vision_layers`, and so on.

A diffusion forward pass — corrupt a canvas, apply a per-layer-type bidirectional
mask, run the DeltaNet recurrence twice, read out unshifted logits — is not
expressible in that request. Claiming otherwise would be exactly the "stock no-code
SFT workflow" pretence the brief warned against.

**Careful with the name:** Studio's `/api/train/diffusion/*` endpoints *are* real
diffusion training — **image** diffusion LoRA. The dataset endpoints take images and
captions (`/api/train/diffusion/dataset/{name}/image/{filename}`,
`.../caption/{filename}`). Unrelated to discrete text diffusion.

### What Studio *can* own

| Capability | Endpoint | Usable for us? |
|---|---|---|
| Model/GGUF management, download | `/api/hub/*`, `/api/models/*` | yes — inventory and GGUF fetch |
| AR inference (OpenAI-compatible) | `/v1/models`, `/v1/completions` | yes — AR reference baselines only |
| LoRA listing | `/api/models/loras` | partially — it expects Unsloth-format adapters |
| LoRA / merged / GGUF export | `/api/export/export/{lora,merged,gguf}` | not for our adapters: our checkpoint contains fusion parameters and a timestep conditioner that have no Unsloth adapter schema |
| Hardware telemetry | `/api/train/hardware` | yes |
| Training | `/api/train/start` | **no** — closed SFT schema |

So the division of labour is: **the Unsloth Python engine (MLX + the GDN custom VJP)
runs the training; Studio owns model management and AR inference.** That split is a
finding, not a workaround, and it is stated in the README rather than glossed.

## Environment split (an upstream constraint, not a choice)

Unsloth pins `transformers <= 5.5.0` and `torch < 2.12`. The v0.1 torch backend needs
`transformers >= 5.8` for the dict-keyed `attention_mask` that makes bidirectional
canvas masks possible without monkeypatching. **The two backends cannot share one
environment.**

```
.venv           torch 2.13.0, transformers 5.15.1   -> v0.1 backend, 128 tests
.venv-unsloth   unsloth 2026.8.21, mlx 0.32.1       -> v0.2 backend, 24 tests
```

Consequence for the science: a v0.1-vs-v0.2 number is cross-framework **and**
cross-model, so it is not a controlled comparison. This is why ablation **A**
(causal DeltaNet + bidirectional full attention) is reproduced *inside* the MLX
backend and is bit-exact with upstream when `bidirectional_deltanet.enabled: false`.
A and B differ by one boolean, one framework, one checkpoint. The v0.1 torch numbers
in `docs/EXPERIMENTS.md` are historical reference only.

## Upstream limitations encountered

1. `mlx_lm.tokenizer_utils.TokenizerWrapper` is not callable and does not accept
   `return_tensors`. `qdif.data.datasets.encode()` now normalises both tokenizer
   flavours so the two backends produce identical token ids.
2. numpy has no `bfloat16`, so every MLX→numpy boundary must cast through float32
   (`to_numpy()` in `mlx_backend/{smoke,probe}.py`).
3. `mlx_lm`'s `Qwen3_5TextModel.__call__` builds its masks internally, so v0.2
   re-implements the decoder loop (`MLXDiffusionQwen._run_layers`) to inject per-layer
   masks and the canvas boundary. Guarded by an exactness test against upstream.
4. `unsloth` installs `bitsandbytes` unconditionally on macOS even though it cannot
   run there — harmless, but not evidence that quantized training is available.
