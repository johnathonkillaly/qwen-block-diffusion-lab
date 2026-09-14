# Project summary, through Act IV-U6 (closed)

This document tells the story of the diffusion-drafter work from the first Uno reproduction
to the close of the project. It is short by design. Every number comes from the report named
next to it, and [RESULTS_INDEX.md](RESULTS_INDEX.md) maps each report to its raw data and
checkpoints.

Renamed from `PROJECT_SUMMARY_THROUGH_U5.md` when Act IV-U6 closed the project
(2026-09-14). The release is tagged `diffusion-specdecode-v0.1` (§ Record).

---

## The question

Acts I–III (August 2026) asked whether Qwen3.5 could be *converted* into a block-diffusion
model. They produced a healthy infilling denoiser (58.2% masked accuracy at t = 0.10, AR
perplexity improving 8.91 → 5.11) that could not generate from scratch.

From September the question narrowed to something a decoder can use:

> Can a small diffusion-trained adapter propose several future tokens in parallel, cheaply
> enough that the frozen autoregressive model verifies them and decodes faster, with the
> target model staying authoritative?

The target throughout is **Qwen3.5-4B-Base**, bf16, frozen, on an Apple M4 Max under MLX.
Decoding is greedy.

## 1. Reproducing Uno (Act IV-U, 2026-09-03)

A from-source reproduction of IFM's Uno: a 0.50% token-conditional LoRA, active only on
noise rows, drafts a block of K tokens, and the same frozen weights verify it in one
forward.

The adapter learned a real, control-separated part of the target's trajectory: K=2
agreement 0.391, against 0.063 untrained and 0.039 for a shuffled-target control. It still
could not beat AR (TPF 0.980, 0.63–0.83× wall clock). Verdict:
`ALGORITHMIC SIGNAL, NO SPEEDUP`. [act4u_results.md](act4u_results.md)

## 2. Removing the recurrent tax (Act IV-U2, 2026-09-04)

24 of 32 layers are Gated DeltaNet recurrences, which cannot be rewound mid-forward. So a
partially accepted block cost a third forward to replay the accepted tokens. U2 recorded
per-token recurrent state inside the verify forward, so committing a prefix became state
selection. With no retraining, TPF went 0.862 → 1.279 at K=4, and wall clock went 1.03× AR
at K=4 and 1.09× at K=2. [act4u2_results.md](act4u2_results.md)

## 3. Useful short-horizon drafting (Acts IV-U3 and U4, 2026-09-04/05)

**U3:** training 8× longer took K=4 from 1.056× to 1.168× AR, with every link from
validation loss to acceptance to wall clock moving together.

**U4:** pushing the horizon found K=4 to be the frontier (59.08 tok/s, 1.195× AR). K=8
committed more tokens per forward but ran at 0.910× K=4, because only three speculative
slots ever pay for their ~1.5 ms.

U4 also made one unregistered observation: 3,200 steps of K=8 training improved *K=4*
decoding more than further K=4 training had. That observation motivated Act IV-U5.
[act4u3_results.md](act4u3_results.md), [act4u4_results.md](act4u4_results.md)

A separate test, **RPRM Stage 1** (2026-09-08), asked whether the denoiser's entropy could
drive early exit. The signal was real but worth only +0.022 AUROC against a pre-registered
0.05, so it stopped. [RPRM_DIFFUSION_STAGE1_RESULTS.md](../RPRM_DIFFUSION_STAGE1_RESULTS.md)

## 4. The speculative decoder, measured properly (Act IV-S, 2026-09-13)

A new harness ran on the U4 drafter `u4b1/step-16000`, left frozen:
* 27 prompts in 9 categories, 128 tokens, 3 repeats;
* arms interleaved within each prompt;
* medians, never best-of-N.

**The K sweep reproduced U4's frontier on new prompts:**

| decoder | ×AR |
|---|---|
| **K=4** | **1.245×** (60.32 vs 48.46 tok/s) |
| K=2 | 1.129× |
| K=8 | 1.128× |
| K=16 | 0.528× |
| K=1 control | 0.814× |

The K=1 control has no speculative slot at all, so 0.814× is the price of the draft→verify
cycle itself. The prefix-survival curve does not depend on K: about 0.60 at the first slot,
halving each slot after. [RESULTS_SPECULATIVE.md](RESULTS_SPECULATIVE.md) §3

## 5. Correctness, and a limit of bf16

Speculative output diverged from native greedy output in 759 positions across the
experiments. Every one was within one bf16 ULP of a top-2 tie (492 were exact ties); none
was a defect. 1.24% of decoded positions carry an exact tie, where a width-1 AR forward and
a width-(K+1) verify forward break it differently.

The accurate statement is *identical to native greedy output except at bf16 ties*. Act
IV-U5 later found the same in the cacheless regime (2 of 15 prompts diverged at ties), and
Act IV-U6 audited 210 more divergences, all within one ULP. Earlier "exactly lossless"
claims held only on narrower prompt sets, and now carry dated addenda.
[RESULTS_SPECULATIVE.md](RESULTS_SPECULATIVE.md) §2

## 6. Controls and artifacts

**Degeneration.** Greedy base-model output loops (mean 8-gram loop fraction 0.18 at 128
tokens, 0.50 at 512), and loops are trivially draftable. Three apparent wins were
artifacts of it:
* speedup growing with generation length;
* high-entropy prompts drafting best;
* a prompt-lookup n-gram drafter nearly matching diffusion (1.199× vs 1.248× pooled). On
  loop-free text that last one reverses: 1.077× vs 1.234×.

All headlines are therefore also reported on the loop-free subset, where K=4 gives 1.230×.

**Refinement.** Four passes raised accepted prefix from 1.07 to 2.78 and cut throughput to
0.776× AR.

**Context scaling.** Decode-only speedup fell from 1.444× at 512 tokens to 1.118× at 16K.
The constant-state recurrent layers leave native decoding almost flat, while the verify
pass grows. [RESULTS_SPECULATIVE.md](RESULTS_SPECULATIVE.md) §5–7

## 7. Adaptive K fails (Act IV-S, gate 5)

Entropy-threshold, survival and expected-value schedulers were all charged their own
cost. The best reached 1.230× against fixed K=4's 1.238× (1.213× vs 1.215× on loop-free
text). The entropy policy had the best tokens-per-forward of any arm and turned none of it
into speed.

The drafter's top-1 probability does predict prefix survival (a 7.5× spread across
deciles), but only after the draft forward that K has to be chosen before.
[RESULTS_SPECULATIVE.md](RESULTS_SPECULATIVE.md) §8–9

## 8. The cross-horizon hypothesis (Act IV-U5)

U5 tested U4's observation with the control U4 lacked. Four arms started from one
byte-identical checkpoint (`u4a/step-12800`):
* A_k4, the matched control, trained at K=4;
* B_k6 at K=6;
* C_k8 at K=8;
* D_curr on a 4→6→8 curriculum.

Each trained 3,200 compute-matched steps, and every arm decoded at K=4. The criteria were
frozen before the first training step.
[act4u5_preregistered_criteria.md](act4u5_preregistered_criteria.md)

## 9. Discovering training-launch variance

A controlling-session restart killed the first launch of A_k4 before its first
checkpoint. Comparing its log with the relaunch showed that two identical launches diverge
from the second logged step. The frozen noise floors measured *evaluation* noise only, so
an arm could pass them on training noise alone.

Dated amendments were added before any endpoint was evaluated:
* A1 recorded the observation;
* A2–A3 added two more launches of the control and a rule for reading them;
* A4 added one independent C_k8 launch.

Every replicate was proven identical to its primary arm, from start checkpoint to command
line, environment, code and end-state config, and every final adapter still differed. The
root cause is not established.

## 10. U5 fails, and U4 is qualified (2026-09-13/14)

**No arm beat the matched control at any checkpoint.** Endpoint accepted-prefix change:

| arm | change vs control [95% CI] |
|---|---|
| B_k6 | +0.009 [−0.108, +0.101] |
| C_k8 | −0.040 [−0.155, +0.071] |
| D_curr | +0.000 |

The only significant difference in 45 matched tests was an arm being *slower*. U5-1, U5-3
and U5-4 failed; U5-5 failed by the letter.

The replicates explain U4:

| run | accepted prefix | vs start |
|---|---|---|
| A_k4, three identical launches | 0.976–1.029 | +0.008 to +0.061 |
| C_k8, two identical launches | 0.988 and 1.083 | +0.020 and +0.115 |
| U4's single K=8 launch | — | +0.090 |

The matched control alone gained +0.060 over the start, and U4's +0.090 sits inside both
ranges. The two C launches had opposite-signed effects.

Verdict: `U4 OBSERVATION WAS NOISE`; `NO CROSS-HORIZON TRANSFER` also holds at this training
budget. U4's result and its curriculum comparison now carry dated addenda.
[act4u5_results.md](act4u5_results.md)

## 11. Decoupling draft width from verify width (Act IV-U6, 2026-09-14)

The last question was about scheduling, not training: could the drafter propose a wider
block while the target verifies a narrower window per forward? The drafter was the frozen
Act IV-S adapter, and **nothing was trained**. The preregistration, the calibrated floor
and the pilot were each committed before the next measurement.

* **A wider causal draft does not improve earlier slots.** The draft pass is causal. With
  identical noise, draft slots 1–4 match the D=4 draft 97–99% of the time across widths
  6, 8 and 16. With the decoder's own noise, they differ only as much as a noise re-draw.
  The truncate variant was killed before any wall-clock work.
* **The verify-forward cost curve** on the M4 Max is 20.1 ms fixed plus 0.80 ms per token
  (R² 0.976). Verifying 2 twice costs 45.1 ms; verifying 4 once costs 24.5 ms.
* **Every staged arm was significantly slower than K=4**, paired over 81 units:

  | arm | vs K=4 [95% CI] |
  |---|---|
  | S(6,4) | −1.58% [−2.89, −0.36] |
  | S(8,4) | −1.61% [−3.10, −0.17] |
  | S(6,3) | −3.03% [−4.28, −1.91] |
  | S(8,2) | −9.01% [−10.58, −7.51] |

  At a fixed draft width, staging commits exactly the same tokens every cycle for every
  verify width (81/81 units). It only adds ~20 ms verify forwards.
* **K=4 remains the practical frontier.** The incumbent reproduced at 1.256× native AR,
  and every preregistered prediction held within half a point.

Verdict: `VERIFY COST DOMINATES`. [act4u6_results.md](act4u6_results.md)

## 12. Project closure

The project had one real positive result and enough negative evidence to understand its
local frontier. It closed on 2026-09-14 with that result published and every closed line
recorded.

---

## Established

- A frozen diffusion drafter speeds up greedy decoding of Qwen3.5-4B-Base on an M4 Max,
  under this project's research MLX harness, at K=4: **1.245×** native AR end-to-end in
  the same harness (1.230× loop-free), reproduced at 1.248× and 1.238× (Act IV-S) and
  1.256× (Act IV-U6).
- Speculative output equals native greedy output except at bf16 ties: every audited
  divergence (759 in Act IV-S, 210 in Act IV-U6) was within one ULP.
- K=4 is the practical frontier on this setup. The survival curve does not depend on
  decode width, only three speculative slots pay for themselves, and Act IV-U6 confirmed
  that no draft/verify decoupling beats it.
- Transactional state capture is what made the recurrent backbone faster than AR (U2).
- On this setup, identical training launches produce materially different adapters, while
  evaluating a fixed adapter is deterministic.
- Tokens-per-forward, acceptance, accepted prefix and committed-tokens-per-cycle can all
  improve while wall-clock speed gets worse. This happened with K=16, refinement, the
  entropy scheduler and staged verification.

## Not established

- Any speedup against optimised inference runtimes (llama.cpp, vLLM, `mlx_lm`'s own
  generator), or on CUDA.
- Any result for sampling-based decoding, other models, other hardware or long generations.
- The root cause of training-launch nondeterminism.
- A horizon-training effect smaller than about ±0.10 accepted prefix, which U5 could not
  resolve with 15 prompts.
- Anything at IFM's training scale (U5 used about 0.8 M tokens per arm).

## Falsified or unsupported

- Longer-horizon training (K=6, K=8, curriculum) makes a better K=4 drafter: **not
  supported** (U5).
- U4's single-launch "K=8 training improves K=4" and "the curriculum stage helps": **not
  supported**, within launch noise.
- Adaptive K beats fixed K=4: **falsified** in this setup (Act IV-S gate 5).
- More refinement makes a faster decoder: **falsified**.
- Speculative decoding gains more at long context: **false** on this hybrid backbone.
- A wider causal draft improves the slots that get verified: **falsified** (U6-1).
- Decoupled draft/verify width or staged verification beats K=4: **falsified** on this
  stack (U6, `VERIFY COST DOMINATES`).
- Denoiser entropy is useful for early exit: **stopped** by its pre-registered bar (RPRM).
- Bidirectional Gated DeltaNet improves denoising: **rejected** (Act II).

## Open questions

- ~~Does decoupling draft width from verify width help the actual decoder?~~ Answered by
  Act IV-U6: no, `VERIFY COST DOMINATES`.
- ~~Does longer-horizon training make a better K=4 drafter?~~ Answered by Act IV-U5: no.

**Further investigation is intentionally deferred.** Questions this project raised but did
not pursue, such as why training launches diverge, whether a width-invariant argmax gives
true byte-identity, how optimised runtimes compare, and whether other backbones behave
differently, are recorded in [../FINDINGS.md](../FINDINGS.md) as observations, not as a
plan. Pursuing any of them would be a new project.

## Record

| | |
|---|---|
| Research branch | `rprm-diffusion-stage1` |
| Act IV-U5 closure | `7172917` |
| Act IV-U6 | `1640c42` (preregistration, before any measurement) · `df71b03` (floor, U6-1, U6-2, before the pilot) · `4567849` (pilot, before the decisive run) · `50394f7` (results) |
| Project closure and release | tag `diffusion-specdecode-v0.1` |
| Canonical adapter | `runs/u4b1/step-16000`, SHA-256 `8200c339d3d47363a3920fc4aca58f3535fc8bf75be431e35204f374700a4f42` |
| Tests at closure (2026-09-14) | 492 passed (`-m "not model"`); 45 passed (MLX model tests) |
