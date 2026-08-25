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

## v0.2 — bidirectional Gated DeltaNet (branch `experiment/bidirectional-deltanet`)

v0.2 moves to **Qwen3.5-4B-Base** on the **Unsloth** engine and asks whether the
pretrained causal DeltaNet recurrence can be run in *both* directions over the canvas
with shared weights, fusing the two directional representations.

Architecture smoke test: **12/12 pass.** Weights shared by object identity (4B stays
4B), the reverse recurrence demonstrably carries information backwards (0/15 earlier
positions move under causal, 15/15 under bidirectional), and the leakage boundary
holds exactly (0 prefix positions move in either condition). Cost: **1.23×** forward,
**1.19×** forward+backward.

**Phase 3 (held-out wikitext-2) settled it: the hypothesis is not supported.** The
aligned dual recurrence loses to the causal v0.1 baseline at every noise level while
costing 1.26x; the aligned-minus-shuffled gap stays at zero throughout training instead
of becoming positive; and a learned gate moves *away* from the reverse path in all 24
layers. The pre-registered continue criterion failed and two kill criteria fired. Full
numbers in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

An honest caveat travels with that: every arm sat in a low-power regime where none
learned generalising denoising (training plateaued by step 250, lift over copy ~-83%
at t=0.10). The *relative* comparison the criteria were written against is valid; the
*absolute* numbers say this capacity/step budget cannot learn the task.

**The earlier zero-training measurement was also negative.** With zero training, a
position-**shuffled** reverse control matches or beats correctly-aligned reverse
fusion at every gate value. So the untrained loss improvement is not backward
positional information — it is an unstructured perturbation of the autoregressive
readout. The one-batch overfit test cannot arbitrate: at 4B it saturates, with the
causal and bidirectional configs both hitting loss ~1e-4 and 100% corrupted-position
accuracy.

Held-out training is the only thing that can decide, and the kill/continue criteria
are recorded **before** that run in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

Two Unsloth findings worth knowing: on Apple silicon Unsloth **is** an MLX stack
(`DEVICE_TYPE == "mlx"`), and it ships a Qwen3.5-specific GatedDeltaNet custom VJP
that v0.2 runs on. Unsloth **Studio** cannot own this training loop — its
`/api/train/start` has a closed SFT schema with no custom-loss hook. Details:
[docs/UNSLOTH_BACKEND.md](docs/UNSLOTH_BACKEND.md),
[docs/BIDIRECTIONAL_DELTANET.md](docs/BIDIRECTIONAL_DELTANET.md).

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/qdif smoke-bidir -c configs/v02_B_mean_fusion.yaml
```

## Status (v0.1, Qwen3.5-0.8B / torch)

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
| **Does it learn to denoise?** | **Yes — on a one-batch memorisation test only.** See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| **Does it generalise?** | **Unknown. Never trained on a corpus.** |

### What is actually established

LoRA on the full-attention `q/k/v/o` projections — 2.4M trainable parameters, 0.32% of
the model — takes Qwen3.5-0.8B from 0% to 95% accuracy on *corrupted* canvas positions
in 150 steps, 31 points above what copying the input would score. With MLP targets and
rank 32 it reaches 100% and loss 0.006. Meanwhile `next_token_accuracy` collapses from
48% to ~0%: the model genuinely abandons autoregressive prediction.

**This is memorisation of two examples.** It proves the optimisation path exists and the
harness is wired correctly. It says nothing about unseen text.

### Two findings worth reading before trusting any number here

1. **A falling loss and a rising accuracy are compatible with learning nothing.** The
   first run "succeeded" at 60 steps with identity accuracy tracking the copy baseline
   to the decimal — the model had learned to echo its input. `lift_over_copy` and
   `corrupted_accuracy` exist because of it.
2. **The AR path does not survive with adapters active.** Reference perplexity degrades
   14.4x and greedy decoding repeats a single token — the exact consequence of teaching
   the model an unshifted readout. AR and diffusion are a *toggle*, not coexistence.

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
