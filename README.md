# Qwen Diffusion Lab (`qdif`)

**Experimental research.** This repository investigates converting an autoregressive
Qwen model into a block-discrete diffusion language model using parameter-efficient
adaptation. It is not an official Qwen, Google, MLX, or Unsloth project.

The question is narrow and falsifiable: **can LoRA plus a few small auxiliary modules
teach a pretrained autoregressive Qwen3.5 a second generation mode — iterative
discrete denoising over a token canvas — without disturbing the first?**

Milestone 1 is not model quality. It is a harness that can load the model, corrupt a
canvas, run a diffusion forward pass, compute a valid loss, push gradients into
adapters only, iteratively denoise, compare against an autoregressive control, and
record enough diagnostics to tell whether the idea is working.

---

## Status

| Component | State |
|---|---|
| Qwen3.5 text-only loading, architecture introspection | working |
| Bidirectional canvas mask injection (no monkeypatching) | working, empirically verified |
| Uniform discrete corruption + schedules | working, exactly reproducible |
| x0-prediction objective + anti-self-deception metrics | working |
| LoRA (layer-index targeting, runtime enable/disable) | working |
| torch / Apple MPS trainer | working |
| Iterative denoising sampler, block-autoregressive generation | working |
| AR control on identical weights | working |
| Checkpoint save/reload (adapters only) | working |
| MLX **training** backend | **not implemented** — deliberate, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Quantized (8-bit / 4-bit) training | **not implemented** on Apple silicon, see [docs/MEMORY.md](docs/MEMORY.md) |
| **Does it actually learn to denoise?** | **No, not yet.** See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |

The headline experimental result so far is a **negative** one, and it is documented
rather than buried: under uniform random-token corruption with the loss on all canvas
positions, LoRA adaptation converges to *copying its input*, not denoising. The loss
falls 5x and identity accuracy climbs to 96%, and both numbers are meaningless —
identity accuracy tracks the copy baseline to the decimal. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), experiment 001.

## Install

```bash
uv venv --python 3.13 .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev,mlx]"
```

## Quick start

```bash
.venv/bin/qdif inspect
```

```bash
.venv/bin/qdif corrupt --text "The cat sat on the mat." --t 0.5
```

```bash
.venv/bin/qdif smoke-forward
```

```bash
.venv/bin/qdif smoke-grad
```

```bash
.venv/bin/qdif probe-bidir
```

```bash
.venv/bin/qdif overfit-one-batch
```

## Commands

| Command | Purpose |
|---|---|
| `qdif inspect` | Hardware, backends, local checkpoints, baseline servers, memory estimates |
| `qdif arch-report` | Live introspection of the loaded Qwen3.5 module tree |
| `qdif corrupt --text ... --t 0.5` | Per-token view of the forward noising process |
| `qdif smoke-forward` | One diffusion forward pass, loss, and metrics |
| `qdif smoke-grad` | Verify gradients reach LoRA and *only* LoRA |
| `qdif probe-bidir` | Measure real information flow across the canvas |
| `qdif overfit-one-batch` | The mandatory first experiment, with an honest verdict |
| `qdif train -c configs/...` | Run a training experiment |
| `qdif reconstruct --noise 0.5` | Reconstruction accuracy sweep across noise levels |
| `qdif generate --prompt ... --steps 16` | Block-diffusion generation |
| `qdif compare --prompt ...` | AR baseline vs diffusion on identical weights |
| `qdif memory` | Detailed memory breakdown |

## Model

Development default is **`Qwen/Qwen3.5-0.8B`**, because it is already on this machine
and is architecturally identical to Qwen3.5-4B in every respect this experiment
depends on: the same 3:1 Gated-DeltaNet / full-attention hybrid, the same MTP head,
the same 248,320-token vocabulary, the same separable vision tower. The stated 4B
target lives in `configs/qwen35_4b_lora_torch.yaml` and needs a download.

Nothing in the harness is tuned to 0.8B. The one thing 0.8B does **not** exercise is
DeltaNet head-group replication (`linear_num_value_heads > linear_num_key_heads`),
which is active at 27B — see [docs/QWEN38_MIGRATION.md](docs/QWEN38_MIGRATION.md).

## Tests

```bash
.venv/bin/python -m pytest tests -q -m "not model"
```

```bash
.venv/bin/python -m pytest tests -q -m model
```

95 checkpoint-free tests, 28 integration tests that skip automatically when no local
Qwen3.5 checkpoint is present.

## Documentation

- [RESEARCH.md](RESEARCH.md) — the tracked research questions and what is known so far
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — code layout, backend strategy, seams
- [docs/DIFFUSION_OBJECTIVE.md](docs/DIFFUSION_OBJECTIVE.md) — the math, and the readout convention
- [docs/QWEN35_NOTES.md](docs/QWEN35_NOTES.md) — architecture findings, especially Gated DeltaNet
- [docs/QWEN38_MIGRATION.md](docs/QWEN38_MIGRATION.md) — what changes at 27B
- [docs/MEMORY.md](docs/MEMORY.md) — memory accounting on unified memory
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — the experiment log, including failures

## Licence and provenance

The development corpus in `src/qdif/data/dev_corpus.py` is public-domain text
(Austen 1813, Carroll 1865, Melville 1851) plus synthetic sentences written for this
repository. No model weights, datasets, checkpoints or caches are committed.
