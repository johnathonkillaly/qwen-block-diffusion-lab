# Qwen Diffusion Lab

A local research lab exploring how an autoregressive **Qwen3.5** model can be adapted
into a **block-diffusion** language model on **Apple Silicon**, using MLX and Unsloth.

It is an honest implementation notebook, not a benchmark paper. Failed hypotheses,
aborted runs and a mis-specified success criterion are all kept in, because the point
is to understand the transition:

> autoregressive Qwen  ⟶  Qwen capable of block diffusion

**Not affiliated with** Qwen/Alibaba, Unsloth, Apple, Google, or the FLARE authors. No
novel architecture is claimed. Act III is a local, small-scale reproduction of ideas
from a published paper — **not** a reproduction of its benchmark numbers.

---

## What happened

### Act I — Can Qwen denoise?
Discrete-denoising plumbing works, and LoRA can overfit a reconstruction task. But
several apparently successful runs were **copy collapse**: loss fell 5×, identity
accuracy hit 96.9%, and the model had learned only to echo its input — identity
accuracy tracked the copy baseline to the decimal. The instrumentation was rewritten
(`lift_over_copy`, `corrupted_accuracy`) so that could not happen again. A later run
then **corrected an earlier conclusion**: copy collapse is a transient the optimiser
passes through, not a terminal state. With no AR term in the objective, autoregressive
ability was destroyed (perplexity 8.06 → 115.87).

### Act II — Bidirectional recurrent diffusion
Qwen3.5's Gated DeltaNet is **causal by construction**, not by masking: a causal
depthwise conv feeds a recurrence with `tril`/`triu` baked in. We built a shared-weight
reverse recurrence — same 4B parameters evaluated twice — verified backward information
flow (0/15 earlier positions move under the native recurrence, 15/15 with ours) and
proved no prefix leakage.

Then it lost. Against pre-registered criteria it failed to beat the causal baseline, it
never beat its own **position-shuffled** control, and when given a learned gate all 24
layers moved *away* from the reverse path. Hypothesis rejected and retired.
[Details](docs/BIDIRECTIONAL_DELTANET.md).

### Act III — Reproducing a known-good method
A FLARE-inspired transfer: combined `L_AR + λ·L_diff`, absorbing-state `[MASK]`
corruption instead of random-token corruption, complementary mask views, and Gated
DeltaNet block-end state scheduling. Run 1 aborted at step 125 when the
canvas-conditioning probe collapsed — the timestep conditioner's bias had grown to 20×
the token-embedding norm and swamped the canvas. Fixed, and run 2 worked.

---

## Findings

Held-out WikiText-103 validation, Qwen3.5-4B-Base, 2500 steps, 148 min, 21.1 GB unified:

| | step 0 | step 2500 |
|---|---|---|
| masked-position accuracy @ t=0.10 | 1.4% | **58.2%** |
| masked-position accuracy @ t=0.50 | 1.6% | **41.8%** |
| masked-position accuracy @ t=0.90 | 0.8% | **13.0%** |
| canvas-conditioning probe (L1) | 1.62 | **1.86** |
| AR reference perplexity | 8.91 | **5.11** |

- **It denoises unseen text**, with strong noise dependence. Act II's pathology was a
  flat ~6–7% regardless of `t`; this is 58% → 13% across the noise range.
- **The AR path got better, not worse** (perplexity 8.91 → 5.11) — the combined
  objective fixes the Act I/II collapse.
- **It is an infiller, not yet a generator.** From a *fully masked* canvas it produces
  high-frequency repetition. Masked accuracy at t=1.00 is only 5.37%.
- **6 of 7 pre-registered criteria passed**, so by its own written definition Act III is
  recorded as a **failure**. The one that failed — visible-token preservation — turned
  out to be mis-specified for absorbing-state diffusion. We did not rewrite it.

Full picture, including everything we **cannot** claim: **[FINDINGS.md](FINDINGS.md)**.

---

## Reproduce

Requires an Apple Silicon Mac (see [Hardware](#hardware)) and ~9 GB for the checkpoint.

```bash
uv venv --python 3.13 .venv-unsloth && VIRTUAL_ENV=.venv-unsloth uv pip install -e ".[mlx,dev]" unsloth
```

Five-minute end-to-end smoke reproduction — capability report, 15 milestone checks, a
short transfer run, and a diffusion trace:

```bash
./scripts/reproduce_smoke.sh
```

The full Act III experiment (~2.5 h on an M4 Max):

```bash
HF_HOME=/path/to/model/cache .venv-unsloth/bin/qdif act3-train -c configs/act3_main.yaml
```

Watch a masked canvas resolve, one denoising step at a time:

```bash
.venv-unsloth/bin/qdif generate-trace --checkpoint runs/act3-main/checkpoint --prompt "The rain had stopped by morning" --steps 16
```

Verify what the backend is actually doing, rather than trusting that it is:

```bash
.venv-unsloth/bin/qdif unsloth-report
```

Other commands: `qdif inspect`, `arch-report`, `corrupt`, `act3-check`, `probe-bidir`,
`act3-compare`, `p3-report`, `memory`.

---

## Hardware

| | |
|---|---|
| Developed on | Apple M4 Max, 128 GB unified memory |
| Act III peak memory | **21.1 GB unified** (not VRAM — see below) |
| Throughput | ~83 canvas tok/s, 3 forwards per step |
| Full run | ~2.5 hours |

Runs on smaller Macs with adjustment — 32 GB at `canvas_length: 64`, 24 GB at
`canvas_length: 48` with `complementary_views: false`. Sizing table, MLX/Unsloth
specifics, and how this differs from a CUDA recipe (no QLoRA, no Triton, no
FlashAttention, no bitsandbytes): **[docs/APPLE_SILICON.md](docs/APPLE_SILICON.md)**.

Apple unified memory is shared with the CPU and the OS and is **not** VRAM; we report
it as unified memory throughout, and note that Metal caps a process at roughly 75% of
installed RAM.

---

## Research journal

Chronological notes, including the wrong turns:

- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — Acts I & II, run by run
- [docs/ACT3_JOURNAL.md](docs/ACT3_JOURNAL.md) — Act III, including the aborted run 1
- [RESEARCH.md](RESEARCH.md) — the tracked research questions and their current status
- [docs/ACT3_OBJECTIVE.md](docs/ACT3_OBJECTIVE.md) — the exact objective, equations and tensor shapes
- [docs/ACT3_CRITERIA.md](docs/ACT3_CRITERIA.md) — success criteria, committed before the run
- [docs/QWEN35_NOTES.md](docs/QWEN35_NOTES.md) — architecture findings
- [docs/BIDIRECTIONAL_DELTANET.md](docs/BIDIRECTIONAL_DELTANET.md) — the Act II hypothesis and its rejection
- [docs/FLARE_COMPARISON.md](docs/FLARE_COMPARISON.md) — mechanism-by-mechanism comparison
- [docs/UNSLOTH_BACKEND.md](docs/UNSLOTH_BACKEND.md) — what Unsloth does on Apple Silicon, verified
- [docs/QWEN38_MIGRATION.md](docs/QWEN38_MIGRATION.md) — what changes at 27B

---

## Prior work

- **FLARE: Diffusion for Hybrid Language Model** — Zhu, Shi, Ge, Tan, Xu, Zhu, Kuen,
  Goswami, Jain, Chen, Tao, Gu. [arXiv:2606.01774](https://arxiv.org/abs/2606.01774).
  Act III implements methodological ideas from this paper, independently, from the
  published specification. Their reference implementation is PolyForm Noncommercial and
  **no code was copied from it**.
- **Qwen3.5** — the Qwen team, for the hybrid Gated-DeltaNet backbone.
- **Gated DeltaNet** and the DeltaNet line of work.
- **Unsloth** — Apple/MLX path and the Qwen3.5-specific Gated DeltaNet custom VJP.
- **MLX / mlx-lm** — Apple.
- **DiffusionGemma** — the original inspiration for attempting AR→diffusion conversion.

Attribution and the full licence audit: **[THIRD_PARTY.md](THIRD_PARTY.md)**.

---

## Limitations

Read these before drawing conclusions.

- **Not a FLARE reproduction in the benchmark sense.** Much smaller scale, LoRA instead
  of full-weight conversion, one 128-token block instead of `K` blocks of 4, WikiText-103
  instead of their transfer mix, and **no logit shift** (the paper mentions one without
  giving the formula in the sections we could read). FLARE itself reports that *data mix
  dominates algorithmic recipe*, so absolute quality here says little about their method.
- **Free generation does not work yet.** The Act III model infills; it does not generate
  coherent blocks from scratch.
- **No fluency evaluation.** Masked-position accuracy is not fluency, and no human or
  benchmark evaluation was run.
- **No speed claims.** The sampler is unoptimised research Python/MLX. Comparing it to
  GGUF, llama.cpp or LM Studio would be meaningless, and we do not.
- **Single seed, single machine.** No variance estimates across seeds.
- **Nothing verified at 27B.** Analysis only.

## Licence

Code in this repository: **Apache-2.0** ([LICENSE](LICENSE)), chosen after the
dependency audit in [THIRD_PARTY.md](THIRD_PARTY.md) rather than by default. Model
weights, datasets and checkpoints are **not** included and carry their own licences.
