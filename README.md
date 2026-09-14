# Diffusion Speculative Decoding on Qwen3.5

A research project, now **closed**, that tested one question on Apple Silicon:

> Can a small diffusion-trained adapter draft several future tokens in parallel, cheaply
> enough that an unmodified autoregressive model, which stays authoritative, decodes
> faster with the same greedy output?

On the tested setup, **yes, modestly**, and only in a narrow configuration. Most of what
the project learned is where that stops working. The negative results are kept as
prominently as the positive one.

**Not affiliated with** Qwen/Alibaba, IFM, Unsloth, Apple, Google, or the FLARE authors.
No novel architecture is claimed: the drafter reproduces IFM's
[Uno](https://github.com/ifm-ai/uno) from its published source.

---

## Headline result

| | |
|---|---|
| Target model | `unsloth/Qwen3.5-4B-Base`, snapshot `61541fe4…`, bfloat16, **frozen** (backbone digest checked before and after every run) |
| Drafter | 21.2 M-parameter token-conditional LoRA (0.50% of the model), frozen: [release asset](#adapter-weights) |
| Runtime | this repository's research Python harness on MLX 0.32.1 / mlx-lm 0.31.3 |
| Hardware | Apple M4 Max, 128 GB unified memory |
| Decoding | greedy; fixed **K=4** (the drafter proposes 4 positions, the target verifies them in one forward) |
| Workload | 27 hand-written prompts in 9 categories, 128 generated tokens, 3 repeats, arms interleaved |
| **Speedup vs native AR in the same harness** | **1.245×** (60.32 vs 48.46 tok/s); **1.230×** on loop-free continuations |
| Reproduced | 1.248× and 1.238× in two later Act IV-S sessions; 1.256× as the Act IV-U6 incumbent |
| Correctness | identical to native greedy output **except at bf16 ties** ([below](#correctness-caveat)) |

That is the whole claim. It is not a claim about optimised runtimes (llama.cpp, vLLM, LM
Studio, `mlx_lm`'s own generator), CUDA, other Qwen models, sampling, other hardware, or
speculative decoding in general.

Sources: [docs/RESULTS_SPECULATIVE.md](docs/RESULTS_SPECULATIVE.md) and
[docs/act4u6_results.md](docs/act4u6_results.md).

## The mechanism

```
committed context ──► drafter forward: [seed, noise, noise, noise]    (adapter on noise rows)
                      proposes p1 … p4          p1 is the target's own next token
                  ──► target verify forward: [seed, p1 … p4]          (adapter off)
                      accept the longest prefix the target agrees with,
                      commit it plus the target's own next token
                  ──► keep only the verified prefix in the target's cache
```

One draft forward plus one verify forward per cycle. The target decides every committed
token.

## What worked

- **Transactional recurrent state (Act IV-U2).** 24 of 32 layers are Gated DeltaNet
  recurrences, which cannot be rewound mid-forward, so the first decoder paid an extra
  forward on every partial rejection. U2 records per-token state inside the verify forward
  and commits a prefix by selecting it. That took tokens per forward from 0.862 to 1.279
  at K=4, and was the step from slower than AR to faster.
- **Short-horizon drafting.** Training the adapter longer raised acceptance and then real
  throughput (Act IV-U3, K=4 1.056× → 1.168×). The prefix-survival curve is about 0.60 at
  the first speculative slot and halves at each slot after, so **three speculative slots
  pay for themselves and a fourth does not**. K=4 is the frontier, confirmed independently
  in Acts IV-U4, IV-S and IV-U6.

## What did not work

Each of these was tested and did not improve wall-clock decoding in this setup.

| hypothesis | result | report |
|---|---|---|
| Wider decode blocks | K=8 1.128×, K=16 0.528× AR, even though K=16 has the highest tokens per forward (1.580) | [RESULTS_SPECULATIVE §3](docs/RESULTS_SPECULATIVE.md) |
| More diffusion refinement at inference | 4 passes: accepted prefix 1.07 → 2.78, throughput **0.776×** AR | [§7](docs/RESULTS_SPECULATIVE.md) |
| Adaptive K / confidence scheduling | best adaptive 1.230× vs fixed K=4 1.238×; the entropy policy won tokens per forward and not speed | [§9](docs/RESULTS_SPECULATIVE.md) |
| Bigger gains at long context | decode-only speedup 1.444× at 512 tokens of context, falling to 1.118× at 16K | [§6](docs/RESULTS_SPECULATIVE.md) |
| Longer-horizon training (K=6, K=8, curriculum 4→6→8) to make a better K=4 drafter | no arm beat a matched K=4 control at any checkpoint: **`U4 OBSERVATION WAS NOISE`**; `NO CROSS-HORIZON TRANSFER` also holds at the tested budget | [act4u5_results](docs/act4u5_results.md) |
| U4's curriculum-transfer interpretation | single-launch comparisons, inside measured launch-to-launch spread; not supported | [act4u4_results addendum](docs/act4u4_results.md) |
| A wider draft improving the verified slots | the draft pass is causal; slots 1–4 change with width no more than a noise re-draw changes them | [act4u6_results §3](docs/act4u6_results.md) |
| Decoupled draft/verify width, staged verification | every staged arm significantly slower than K=4 (−1.58% to −9.01%): **`VERIFY COST DOMINATES`** | [act4u6_results §7](docs/act4u6_results.md) |
| Denoiser entropy for early exit (RPRM) | +0.022 AUROC beyond progress, against a pre-registered 0.05: **STOP** | [RPRM results](RPRM_DIFFUSION_STAGE1_RESULTS.md) |
| Bidirectional recurrent diffusion (Act II) | lost to the causal baseline and to its own shuffled control | [BIDIRECTIONAL_DELTANET](docs/BIDIRECTIONAL_DELTANET.md) |

## What the project learned

**Algorithmic metrics are not wall-clock metrics.** Five times, a better number on paper
came with a slower decoder:
- acceptance improved without speed;
- tokens per forward improved without speed (K=16);
- refinement made predictions much better and decoding slower;
- adaptive scheduling was more algorithmically efficient and no faster;
- staged verification committed more tokens per cycle and still lost.

On this stack a verify forward costs about **20.1 ms fixed plus 0.80 ms per token**, so
anything that adds a forward pays far more than it looks on paper. Only committed tokens
per wall-clock second decides.

**A training launch is one draw.** Relaunching an identical training run gives a
materially different adapter, even with identical start checkpoint, arguments, seed, data
position, optimizer state, code and environment, all verified from artifacts. K=4 accepted
prefix spanned 0.976–1.029 across three identical launches, and two identical K=8 launches
gave 0.988 and 1.083. Evaluating a fixed adapter is deterministic, so this is training
variance. **The root cause is not established.** Comparing independently trained adapters
needs replicate launches.

**Timing needs pairing.** Two *identical* K=4 arms had medians 2.3% apart but a paired
mean 0.16% apart. Every decision here interleaves arms within each prompt and uses paired
statistics.

**Greedy base-model output loops, and loops flatter drafting.** At 512 tokens, half of all
8-grams repeat. Unstratified, that produced three false wins, including a parameter-free
n-gram drafter nearly matching diffusion: 1.199× vs 1.248×, which becomes 1.077× vs 1.234×
on loop-free text. Headlines are therefore also reported on the loop-free subset.

## Correctness caveat

Speculative output equals native greedy output **except where the target's top two logits
tie in bfloat16**. A width-1 AR forward and a width-5 verify forward can break such a tie
differently, and neither is wrong.
- **Frequency:** 1.24% of decoded positions carry an exact tie.
- **Audit:** every divergence was checked against the target's full-context logits. All
  759 in Act IV-S and all 210 in Act IV-U6 were within one bf16 ULP of a tie. None was a
  decoder defect.

True byte-identity would need a width-invariant target argmax, such as fp32 logits or
tie-breaking by token id. It was not attempted.

## Results and raw evidence

- **[docs/RESULTS_INDEX.md](docs/RESULTS_INDEX.md):** every stage with its question,
  verdict, report, raw data, plots and checkpoint, plus the artifact behind each headline
  number.
- **[docs/PROJECT_SUMMARY_THROUGH_U6.md](docs/PROJECT_SUMMARY_THROUGH_U6.md):** the story in
  a few pages.
- **[FINDINGS.md](FINDINGS.md):** what is established, what is not, and what was falsified.
- **Machine-readable results** are in `results/`: raw per-row measurements, divergence
  audits, provenance, and `checkpoint_manifest.json`. Plots are in `plots/` and
  `results/act4u*/`.

Every stage from Act IV-U on was pre-registered before measurement. The criteria documents
are frozen, and later amendments are dated.

## Reproduction

Requires an Apple Silicon Mac and the `unsloth/Qwen3.5-4B-Base` checkpoint (~9 GB).

```bash
uv venv --python 3.13 .venv-unsloth && VIRTUAL_ENV=.venv-unsloth uv pip install -e ".[mlx,dev]" unsloth
```

**Rebuild tables and plots from the committed data (no model, seconds):**

| script | what it rebuilds |
|---|---|
| `scripts/spec_report.py --tag main` | Act IV-S tables and plots |
| `scripts/u5_matched_pairing.py --eval results/act4u5/raw/eval_curve.json` | Act IV-U5 matched-control scoring |
| `scripts/u5_replicate_report.py --eval results/act4u5/raw/eval_replicates.json` | Act IV-U5 launch robustness |
| `scripts/u6_report.py final --floor 1.0` | Act IV-U6 gates, verdict and plots |
| `scripts/rprm_stage1_analyze.py --tag primary` | RPRM Stage 1 analysis |

**Re-measure (model plus the drafter adapter):**

| what | script |
|---|---|
| Speculative decoding: baseline, K sweep, controls, context, adaptive K | `scripts/spec_decode.py`, `scripts/spec_run_all.sh`, `scripts/spec_audit.py` |
| Decoupled draft/verify width | `scripts/u6_wide_draft_diagnostic.py`, `u6_cost_curve.py`, `u6_bench.py` |
| Training the adapter; Act IV-U…U5 evaluation | `scripts/uno.py`; `scripts/u5_launch.sh`, `u5_replicates.sh` |

Pass the adapter with `--adapter /path/to/<asset>.safetensors`. Set `HF_HOME` to the
directory holding the model cache. Launch multi-hour runs in their own process session
(see `AGENTS.md`).

**Tests (2026-09-14):**

| group | result |
|---|---|
| `pytest -m "not model"` | 492 passed |
| MLX model tests (`tests/test_act3.py`, `tests/test_bidirectional_deltanet.py`) | 45 passed |
| legacy torch path | 19 pre-existing failures from a `transformers` version mismatch, documented in `AGENTS.md` |

## Adapter weights

The one adapter behind every speed claim is published as a release asset. It is not in
git.

| | |
|---|---|
| Release | [`diffusion-specdecode-v0.1`](https://github.com/johnathonkillaly/qwen-block-diffusion-lab/releases/tag/diffusion-specdecode-v0.1) |
| File | `qwen3.5-4b-base-diffusion-drafter-u4b1-step16000.safetensors` (84,966,094 bytes) |
| SHA-256 | `8200c339d3d47363a3920fc4aca58f3535fc8bf75be431e35204f374700a4f42` |
| Provenance | Act IV-U4 arm B1, step 16000, trained at block size 8; frozen for Acts IV-S and IV-U6 |

Verify with `shasum -a 256 <file>`. Optimizer state and the other experimental checkpoints
are not published; their digests are in `results/checkpoint_manifest.json`. The base model
is not redistributed. Licensing notes are in [THIRD_PARTY.md](THIRD_PARTY.md).

## Scope and limitations

- **Research code, not an inference engine.** Speed is measured against native AR decoding
  in this Python/MLX harness only, with no comparison against optimised runtimes.
- **One model, one machine, greedy decoding, short hand-written prompts.** Other models,
  hardware, sampling and long generations were not tested.
- **Long context works against this backbone.** 24 of 32 layers keep constant-size state,
  so native decoding barely slows with context.
- **Training variance is only partly measured:** three K=4 and two K=8 launches in Act
  IV-U5. Earlier single-launch comparisons are qualified in their reports.
- **Earlier acts** (I–III: AR → diffusion conversion) produced an infilling denoiser, not a
  generator, and do not reproduce FLARE's benchmarks.

## Experiment map

| stage | question | verdict | report |
|---|---|---|---|
| Act I | Can Qwen3.5 learn token denoising with LoRA? | plumbing works; copy collapse found; with no AR term the AR path is destroyed | [EXPERIMENTS](docs/EXPERIMENTS.md) |
| Act II | Does a bidirectional Gated DeltaNet improve denoising? | **rejected** | [BIDIRECTIONAL_DELTANET](docs/BIDIRECTIONAL_DELTANET.md) |
| Act III | Does a FLARE-style AR + diffusion objective give a healthy denoiser? | healthy denoiser; 6/7 criteria, **recorded as a failure** | [ACT3_JOURNAL](docs/ACT3_JOURNAL.md) |
| Act IV-N | Does a structured corruption alphabet help? | **paused** before Stage 2, never resumed | not in this tree (see `AGENTS.md`) |
| Act IV-U | Can a Uno-style adapter on a frozen 4B model draft blocks the model accepts? | `ALGORITHMIC SIGNAL, NO SPEEDUP` | [act4u_results](docs/act4u_results.md) |
| Act IV-U2 | Does transactional recurrent verification remove the replay cost? | `TRANSACTIONAL SPEEDUP ACHIEVED` | [act4u2_results](docs/act4u2_results.md) |
| Act IV-U3 | Does acceptance, then speed, scale with training? | `ACCEPTANCE AND SPEED SCALE` | [act4u3_results](docs/act4u3_results.md) |
| Act IV-U4 | How far can the parallel horizon go? | `K4 IS THE PRACTICAL FRONTIER` | [act4u4_results](docs/act4u4_results.md) |
| RPRM Stage 1 | Does denoiser uncertainty justify early exit? | **STOP** | [RPRM results](RPRM_DIFFUSION_STAGE1_RESULTS.md) |
| Act IV-S | The speculative decoder: K sweep, correctness, controls, context, adaptive K | `K=4 STANDS. ADAPTIVE K FAILS. TWO OF THE APPARENT WINS WERE ARTIFACTS.` | [RESULTS_SPECULATIVE](docs/RESULTS_SPECULATIVE.md) |
| Act IV-U5 | Does longer-horizon training make a better K=4 drafter? | `U4 OBSERVATION WAS NOISE` | [act4u5_results](docs/act4u5_results.md) |
| Act IV-U6 | Does decoupling draft width from verify width speed up decoding? | `VERIFY COST DOMINATES` | [act4u6_results](docs/act4u6_results.md) |

## Status

**Research project closed (2026-09-14).** The result above and the negative results
around it are the final state. No further stage is planned. Future work would be a new
research question in a new branch or project.

## Prior work

- **Uno** by IFM ([github.com/ifm-ai/uno](https://github.com/ifm-ai/uno), Apache-2.0). The
  drafter objective and runtime are reproduced from its published source; provenance is in
  [docs/act4u_uno_source_notes.md](docs/act4u_uno_source_notes.md).
- **FLARE: Diffusion for Hybrid Language Model**, Zhu et al.,
  [arXiv:2606.01774](https://arxiv.org/abs/2606.01774). Acts I–III implement its ideas
  independently. No code was copied.
- **Qwen3.5** by the Qwen team; **Gated DeltaNet**; **Unsloth**, for its MLX path and
  Qwen3.5 DeltaNet VJP; **MLX / mlx-lm** by Apple.

Attribution and the full licence audit: [THIRD_PARTY.md](THIRD_PARTY.md). Hardware notes:
[docs/APPLE_SILICON.md](docs/APPLE_SILICON.md).

## Licence

Code is **Apache-2.0** ([LICENSE](LICENSE)). Model weights and datasets carry their own
licences. See [THIRD_PARTY.md](THIRD_PARTY.md), including its notes on the released
adapter.
