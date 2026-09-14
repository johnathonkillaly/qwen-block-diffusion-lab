# Qwen Diffusion Lab

A local research lab on Apple Silicon asking one question, arrived at the long way:

> Can a lightweight diffusion-style adapter predict several future tokens in parallel, so
> that an unmodified autoregressive model, which stays authoritative, decodes faster?

Everything runs on MLX on one M4 Max. This is a research notebook, not a production
inference engine. Failed hypotheses, aborted runs and a mis-specified success criterion
are kept in on purpose: the project is run to disprove itself, and most of what it has
learned is negative.

**Not affiliated with** Qwen/Alibaba, IFM, Unsloth, Apple, Google, or the FLARE authors. No
novel architecture is claimed. The drafter is a from-source reproduction of IFM's
[Uno](https://github.com/ifm-ai/uno), and Act III reproduces ideas from FLARE.

---

## Current headline result

| | |
|---|---|
| Target model | Qwen3.5-4B-Base, bfloat16, frozen (backbone digest checked before and after every run) |
| Drafter | 21.2 M-parameter token-conditional LoRA (0.50% of the model), checkpoint `u4b1/step-16000` |
| Hardware / framework | Apple M4 Max, 128 GB unified memory; MLX 0.32.1 |
| Decoding | greedy; the drafter proposes 4 positions, the target verifies them in one forward, the accepted prefix is committed |
| Workload | 27 hand-written prompts in 9 categories, 128 generated tokens, 3 repeats, all arms interleaved |
| **Speedup** | **1.245× end-to-end over native AR in the same harness** (60.32 vs 48.46 tok/s). **1.230×** on loop-free continuations. Two later sessions: 1.248× and 1.238× |
| Correctness | identical to native greedy output **except at bf16 ties**: 1.24% of decoded positions have an exact top-2 tie, and all 759 audited divergences are within one bf16 ULP of a tie, none a decoder defect |

**Scope, stated plainly.** One model, one machine, short prompts, greedy decoding, research
Python/MLX. The baseline is native AR decoding in the same harness, not an optimised
runtime such as llama.cpp, LM Studio or `mlx_lm`'s own generator, so nothing here is a
claim about speed against those. The speedup is not universal:
* on loop-free text it ranges from 1.08× (technical prose) to 1.34× (structured text);
* it shrinks at long context.

Source: [docs/RESULTS_SPECULATIVE.md](docs/RESULTS_SPECULATIVE.md).

---

## What did not work

- **Wider draft blocks.** K=8 gives 1.128× and K=16 0.528×, even though K=16 commits the most
  tokens per forward (1.580). At every width tested, only three speculative slots pay for
  their cost.
- **More denoising at inference.** Four refinement passes nearly triple the accepted prefix
  (1.07 → 2.78) and make decoding slower than no speculation at all (0.776× AR).
- **Adaptive K.** No scheduler beat fixed K=4 (best 1.230× vs 1.238×).
- **Long context.** Decode-only speedup falls from 1.44× at 512 tokens of context to 1.12× at
  16K. 24 of 32 layers keep constant-size recurrent state, so native decoding barely slows
  with context, while the verify pass does.
- **Training the drafter on longer horizons (Act IV-U5).** K=6, K=8 and a 4→6→8 curriculum
  did not beat a matched K=4 control at K=4 decoding. The earlier observation that
  motivated the experiment turned out to be noise.
- **Uncertainty-based early exit (RPRM).** Denoiser entropy predicts acceptance, but only
  +0.022 AUROC beyond a progress baseline, against a pre-registered bar of 0.05.
- **The first speculative decoder (Act IV-U).** It was slower than AR until Act IV-U2 removed
  a replay forward that the recurrent layers forced on every partial rejection.
- **Bidirectional recurrent diffusion (Act II).** It lost to the causal baseline and never
  beat its own position-shuffled control.

---

## Why the negative results matter

**Algorithmic metrics are not speed.** Tokens per forward, acceptance and accepted prefix
all *improved* in arms that were *slower*: K=16, four refinement passes, the entropy
scheduler. A speculative position costs about 1.5 ms of a ~50 ms cycle and has to return
enough accepted tokens to pay for itself. Only wall-clock committed tokens per second
decides whether a decoder is better.

**Greedy base-model output loops, and loops flatter everything.** At 512 generated tokens,
half of all 8-grams are repeats. Unstratified, that produced three false results:
* "speedup grows with generation length";
* "high-entropy prompts draft best";
* "a parameter-free n-gram drafter matches diffusion".

Every headline here is also reported on the loop-free subset.

**A training launch is one draw.** On this MLX/Metal setup, relaunching an identical
training run gives a materially different adapter. "Identical" covers checkpoint,
arguments, seed, data position, optimizer state, code and environment, all verified from
artifacts. K=4 accepted prefix ranged 0.976–1.029 across three identical launches, and two
K=8 launches gave 0.988 and 1.083. Evaluating a fixed adapter is deterministic, so this
spread is training variance; its root cause is not established. Comparing separately
trained adapters therefore needs replicate launches. Act IV-U4's "longer-horizon training
helps" rested on a single launch and did not survive Act IV-U5.

---

## Experiment map

| stage | question | verdict | report |
|---|---|---|---|
| Act I | Can Qwen3.5 learn token denoising with LoRA? | plumbing works; copy collapse found and instrumented; with no AR term the AR path is destroyed | [EXPERIMENTS](docs/EXPERIMENTS.md) |
| Act II | Does a bidirectional Gated DeltaNet improve denoising? | **rejected** by pre-registered criteria | [BIDIRECTIONAL_DELTANET](docs/BIDIRECTIONAL_DELTANET.md) |
| Act III | Does a FLARE-style AR + diffusion objective give a healthy denoiser? | healthy denoiser and better AR perplexity; 6/7 criteria, so **recorded as a failure** | [ACT3_JOURNAL](docs/ACT3_JOURNAL.md) |
| Act IV-N | Does a structured corruption alphabet help? | **paused** before Stage 2 | not in this tree (see `AGENTS.md`) |
| Act IV-U | Can a Uno-style adapter on a frozen 4B model draft blocks the model accepts? | `ALGORITHMIC SIGNAL, NO SPEEDUP` | [act4u_results](docs/act4u_results.md) |
| Act IV-U2 | Does transactional recurrent verification remove the replay cost? | `TRANSACTIONAL SPEEDUP ACHIEVED` (1.03× at K=4, 1.09× at K=2) | [act4u2_results](docs/act4u2_results.md) |
| Act IV-U3 | Does acceptance, and then speed, scale with training? | `ACCEPTANCE AND SPEED SCALE` (1.168×) | [act4u3_results](docs/act4u3_results.md) |
| Act IV-U4 | How far can the parallel horizon go? | `K4 IS THE PRACTICAL FRONTIER` (1.195×) | [act4u4_results](docs/act4u4_results.md) |
| RPRM Stage 1 | Does denoiser uncertainty justify early exit? | **STOP** | [RPRM results](RPRM_DIFFUSION_STAGE1_RESULTS.md) |
| Act IV-S | The speculative decoder: K sweep, correctness, controls, context, adaptive K | `K=4 STANDS. ADAPTIVE K FAILS. TWO OF THE APPARENT WINS WERE ARTIFACTS.` (1.245×) | [RESULTS_SPECULATIVE](docs/RESULTS_SPECULATIVE.md) |
| Act IV-U5 | Does longer-horizon training make a better K=4 drafter? | `U4 OBSERVATION WAS NOISE` (and no cross-horizon transfer) | [act4u5_results](docs/act4u5_results.md) |

Further reading:
* the story in a few pages: [docs/PROJECT_SUMMARY_THROUGH_U5.md](docs/PROJECT_SUMMARY_THROUGH_U5.md);
* every stage with its dates, raw data and checkpoints: [docs/RESULTS_INDEX.md](docs/RESULTS_INDEX.md);
* everything established, and everything we **cannot** claim: [FINDINGS.md](FINDINGS.md).

---

## Reproduction

Requires an Apple Silicon Mac and the `unsloth/Qwen3.5-4B-Base` checkpoint (~9 GB).

```bash
uv venv --python 3.13 .venv-unsloth && VIRTUAL_ENV=.venv-unsloth uv pip install -e ".[mlx,dev]" unsloth
```

Trained adapters are not in git. `results/checkpoint_manifest.json` lists their paths and
SHA-256 digests. The training commands are in each report's reproduction section and in
[AGENTS.md](AGENTS.md).

**From committed data, no model needed (seconds):**

```bash
.venv-unsloth/bin/python scripts/spec_report.py --tag main
```

Rebuilds the Act IV-S tables and plots from the raw rows in `results/speculative/`.

```bash
.venv-unsloth/bin/python scripts/u5_matched_pairing.py --eval results/act4u5/raw/eval_curve.json
```

Scores every Act IV-U5 checkpoint against its matched control.

```bash
.venv-unsloth/bin/python scripts/u5_replicate_report.py --eval results/act4u5/raw/eval_replicates.json
```

Rebuilds the U5 training-launch robustness section.

```bash
.venv-unsloth/bin/python scripts/rprm_stage1_analyze.py --tag primary
```

Rebuilds the RPRM analysis.

**Measurements (model and drafter adapter required):**

| what | script |
|---|---|
| Act IV-S baseline, K sweep, controls, context, adaptive K | `scripts/spec_decode.py` (subcommands) and `scripts/spec_run_all.sh` |
| Act IV-S divergence and loop audits | `scripts/spec_audit.py` |
| Act IV-U5, primary run (~8 h) | `scripts/u5_launch.sh` |
| Act IV-U5, replicate launches and supplementary session (~4.5 h) | `scripts/u5_replicates.sh` |
| Act IV-U5, cacheless losslessness | `scripts/u5_losslessness.py` |
| Act IV-U…U4 training and evaluation | `scripts/uno.py` (`train`, `u3-eval`, `u4-eval`, `u5-eval`, …) |
| Act III | `qdif act3-train -c configs/act3_main.yaml` |
| Five-minute smoke test | `scripts/reproduce_smoke.sh` |

Set `HF_HOME` to the directory holding the model cache. Launch multi-hour runs in their
own process session. As `AGENTS.md` explains, a run started from an agent's shell dies
with that shell.

Tests (2026-09-14):

| test group | result |
|---|---|
| `pytest -m "not model"` | 436 passed |
| MLX model tests (`tests/test_act3.py`, `tests/test_bidirectional_deltanet.py`) | 45 passed |
| legacy torch path (`tests/test_model_integration.py`) | 19 pre-existing failures, from a `transformers` version mismatch, documented in `AGENTS.md` |

---

## Status

**Experimental research, not a production inference engine.** Act IV-U5 is closed.
Planned next: **Act IV-U6**, which asks whether decoupling the drafter's width from the
verifier's width helps the actual decoder. It uses the same frozen drafter and trains
nothing.

## Hardware

| | |
|---|---|
| Developed on | Apple M4 Max, 128 GB unified memory |
| Speculative decoding peak memory | 8.59 GB (native AR), 8.89 GB (K=4) |
| Act III training peak memory | 21.1 GB unified |
| Act IV-U5 training | ~1.26 h per 3,200-step arm |

Apple unified memory is shared with the CPU and the OS and is **not** VRAM, so we report
it as unified memory throughout. Sizing for smaller Macs, and how this differs from a CUDA
recipe: [docs/APPLE_SILICON.md](docs/APPLE_SILICON.md).

## Prior work

- **Uno** by IFM ([github.com/ifm-ai/uno](https://github.com/ifm-ai/uno), Apache-2.0). The
  drafter objective and runtime are reproduced from its published source; provenance is in
  [docs/act4u_uno_source_notes.md](docs/act4u_uno_source_notes.md).
- **FLARE: Diffusion for Hybrid Language Model**, Zhu et al.,
  [arXiv:2606.01774](https://arxiv.org/abs/2606.01774). Act III implements its
  methodological ideas independently, from the published specification. Their reference
  implementation is PolyForm Noncommercial and **no code was copied from it**.
- **Qwen3.5** by the Qwen team; **Gated DeltaNet** and the DeltaNet line of work;
  **Unsloth**, for its Apple/MLX path and Qwen3.5 Gated DeltaNet VJP; **MLX / mlx-lm** by
  Apple.

Attribution and the full licence audit: [THIRD_PARTY.md](THIRD_PARTY.md).

## Limitations

- **Speed claims are relative to native AR in this harness only.** Nothing is claimed
  against optimised inference runtimes.
- **One model, one machine, greedy decoding, short hand-written prompts.** Sampling-preserving
  speculative decoding was not tested.
- **Training variance is only partly measured.** It was measured for Act IV-U5 (three
  identical K=4 launches and two identical K=8 launches). Earlier single-launch comparisons
  between separately trained adapters are qualified in their reports.
- **Exact greedy identity holds up to bf16 ties.** True byte-identity would need a
  width-invariant target argmax.
- **Act III is not a FLARE reproduction in the benchmark sense.** It uses much smaller
  scale, LoRA instead of full-weight conversion, WikiText-103 instead of their data mix,
  and no logit shift. The Act III model infills; it does not generate coherent blocks from
  scratch.
- **No fluency evaluation, and nothing verified at 27B.**

## Licence

Code in this repository is **Apache-2.0** ([LICENSE](LICENSE)). That licence was chosen
after the dependency audit in [THIRD_PARTY.md](THIRD_PARTY.md), not by default. Model
weights, datasets and checkpoints are **not** included and carry their own licences.
