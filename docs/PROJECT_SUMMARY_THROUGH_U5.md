# Project summary, through Act IV-U5

This document tells the story of the diffusion-drafter work from the first Uno reproduction
to the close of Act IV-U5. It is short by design. Every number comes from the report named
next to it, and [RESULTS_INDEX.md](RESULTS_INDEX.md) maps each report to its raw data and
checkpoints.

Research checkpoint: commit `7172917` closes Act IV-U5. It was recorded here in the
following commit, because a commit cannot contain its own hash (§ Record).

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
halving each slot after.
[RESULTS_SPECULATIVE.md](RESULTS_SPECULATIVE.md) §3

## 5. Correctness, and a limit of bf16

Speculative output diverged from native greedy output in 759 positions across the
experiments. Every one was within one bf16 ULP of a top-2 tie (492 were exact ties); none
was a defect. 1.24% of decoded positions carry an exact tie, where a width-1 AR forward and
a width-(K+1) verify forward break it differently.

The accurate statement is *identical to native greedy output except at bf16 ties*. Act
IV-U5 later found the same in the cacheless regime: 2 of 15 prompts diverged at ties.
Earlier "exactly lossless" claims held only on narrower prompt sets, and now carry dated
addenda. [RESULTS_SPECULATIVE.md](RESULTS_SPECULATIVE.md) §2

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
frozen before the first training step. [act4u5_preregistered_criteria.md](act4u5_preregistered_criteria.md)

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
line, environment, code and end-state config, and every final adapter still differed.

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
ranges. The two C launches even had opposite-signed effects (A4 case 5).

Verdict: `U4 OBSERVATION WAS NOISE`; `NO CROSS-HORIZON TRANSFER` also holds at this training
budget. U4's result and its curriculum comparison now carry dated addenda.
[act4u5_results.md](act4u5_results.md)

## 11. What remains established

The decoder result does not depend on U5. It is a decode-time measurement of one fixed
adapter, and U5 found no problem with it. What U5 changes is the *reason* that adapter was
chosen and the evidence behind claims about training recipes.

---

## Established

- A frozen diffusion drafter speeds up greedy decoding of Qwen3.5-4B-Base on an M4 Max under
  MLX at K=4: **1.245×** native AR end-to-end in the same harness (1.230× loop-free),
  reproduced at 1.248× and 1.238×.
- Speculative output equals native greedy output except at bf16 ties (759/759 divergences
  within one ULP).
- K=4 is the practical static frontier. The survival curve does not depend on decode
  width, and only three speculative slots pay for themselves.
- Transactional state capture is what made the recurrent backbone faster than AR (U2).
- On this setup, identical training launches produce materially different adapters, while
  evaluating a fixed adapter is deterministic.
- Tokens-per-forward, acceptance and accepted prefix can all improve while wall-clock speed
  gets worse (K=16, refinement, the entropy scheduler).

## Not established

- Any speedup against optimised inference runtimes.
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
- Denoiser entropy is useful for early exit: **stopped** by its pre-registered bar (RPRM).
- Bidirectional Gated DeltaNet improves denoising: **rejected** (Act II).

## Open questions

- ~~Does decoupling draft width from verify width help the actual decoder?~~ Answered after
  this checkpoint by Act IV-U6: no, `VERIFY COST DOMINATES`
  ([act4u6_results.md](act4u6_results.md)).
- Would a width-invariant target argmax (fp32 logits, or tie-breaking by token id) give true
  byte-identity, and at what cost?
- What makes training launches diverge on MLX/Metal? Any future training comparison needs
  at least three launches per arm and more than 15 evaluation prompts.
- Does the long-context case for speculation return on an attention-only backbone?
- Act IV-N (structured corruption alphabet) is paused before its Stage 2.

## Record

| | |
|---|---|
| Branch | `rprm-diffusion-stage1` |
| Last commit before this checkpoint | `4063e4a` (Act IV-U5 PREP) |
| Closure commit | `7172917` (docs/results: close Act IV-U5 and qualify cross-horizon finding) |
| Tests at closure (2026-09-14) | 436 passed (`-m "not model"`); 45 passed (MLX model tests) |
