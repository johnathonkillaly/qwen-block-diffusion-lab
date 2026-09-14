# Act IV-U5 success criteria — pre-registered

**Written during U5-PREP, before any U5 training step has run and while heavy execution
is blocked by an unrelated BoothGPT pretraining job.** No U5 result exists at the time
of writing. Every threshold below is calibrated from Act IV-U4's harness calibration
(`runs/u4-calib/rep{1,2,3}.json`, three evaluations of one *frozen* adapter differing
only in noise seeds), which contains no U5 information.

Frozen from here; amendments go in §11 with a date and a reason.

Design: [`act4u5_design.md`](act4u5_design.md).

---

## 0. Hypothesis

> `K_train > K_decode` can act as richer supervision: training the diffusion adapter
> against longer future blocks produces a better **K=4** decoder than training directly
> at K=4, under an equal training budget.

Provisionally: **cross-horizon distillation transfer**. Named as a label, not claimed
as established literature.

The defining rule (brief §12): **every arm decodes at `K_decode = 4`.** K=6/K=8
decoding is a diagnostic and never determines the verdict.

## 1. The prediction being tested

Act IV-U4 observed, unmatched and un-pre-registered, at `K_decode = 4`:

| vs `step-12800` | Δ accepted prefix | Δ TPF | Δ tok/s (median) | Δ tok/s (mean) |
|---|---|---|---|---|
| +3200 steps at K=8 | **+0.0895** | +0.0279 | +5.2% | +2.8% |
| +1600 K=6 then +1600 K=8 | +0.0772 | +0.0279 | +4.3% | +3.0% |

U5's arm C and arm D should reproduce something like this **against a matched K=4
control**. Recording the numbers here makes U5 a replication test rather than a
re-description: an arm C that lands at +0.09 replicates, one at +0.01 does not, and the
difference is decided by thresholds fixed below.

**A prediction that follows from U4 and is recorded so it can fail:** U4's K=4 gain sat
almost entirely at offset `+2` (+0.0234), with `+3` at −0.0039 and `+4` at +0.0117. If
that pattern holds, U5-2 will **fail** and the effect must be described as generic
next-token sharpening rather than horizon transfer. We expect U5-1 to pass and U5-2 to
be the interesting one.

## 2. Measurement noise, from U4's calibration

Three evaluations of one frozen adapter, differing only in the corruption seed and the
draft-noise seed, at `K_decode = 4`:

| quantity | mean | sd | 2 sd |
|---|---|---|---|
| validation TV | 0.9751 | 0.0029 | 0.0057 |
| teacher-forced agreement (specs) | 0.3086 | 0.0057 | 0.0113 |
| free-running acceptance rate | 0.3170 | 0.0042 | 0.0084 |
| **mean accepted prefix** | 0.9510 | 0.0127 | **0.0254** |
| **TPF** | 1.3909 | 0.0031 | **0.0062** |
| median tok/s (between sessions) | 55.83 | 2.356 | 4.71 |
| within-session repeat noise, tok/s | — | — | 0.10–0.22 |
| pooled per-offset agreement sd | — | 0.0072 | — |

**Every U5 comparison is paired**: same session, same prompts, same draft-noise stream,
same held-out rows, arm against control. The between-session sd of 2.36 tok/s therefore
does not apply to a U5 contrast; it is why cross-session comparison is forbidden.

## 3. How significance is decided

Two conditions, both required. A paired bootstrap alone can resolve effects far smaller
than anything practically interesting; a fixed floor alone ignores the pairing that
makes U5 possible.

```
(a) statistical — the 95% paired-bootstrap CI on mean(arm − control) excludes zero
                  (10,000 resamples over the paired differences)
(b) practical   — the point estimate exceeds the pre-registered floor for that metric
```

Floors, taken as U4's replicate **2 sd** — i.e. "larger than the difference a different
noise draw would have produced on an unchanged adapter":

| metric | floor |
|---|---|
| mean accepted prefix (free-running, K=4) | **0.0254** |
| TPF (K=4) | **0.0062** |
| per-offset teacher-forced agreement | **0.022** (3 × pooled sd) |
| wall-clock tok/s | see U5-4 |

## 4. Gates

### U5-0 — infrastructure integrity (required)

* frozen backbone digest unchanged across every training run and every evaluation;
* seed-row agreement (offset `+1`) exactly 1.000 at every arm and every `K_decode`;
* all arms began from a byte-identical adapter and optimizer, evidenced by
  `results/act4u5/u5_start_check.json`;
* after training, no two arms share an adapter digest (they must have diverged);
* `tests/test_uno_rng.py`, `tests/test_uno_resume.py` and
  `tests/test_uno_horizon_pairing.py` pass — RNG streams isolated, resume bitwise
  identical, arms sharing corruption on shared positions;
* the BoothGPT resource gate passed (i.e. no protected process was running) at the
  moment each heavy phase began, recorded in each artifact's `provenance.resource_guard`.

A failure here voids the stage.

### U5-1 — transfer exists

At least one of **B_k6**, **C_k8**, **D_curr** beats **A_k4** on K=4 **mean accepted
prefix** by both conditions in §3 (CI excludes zero, and Δ ≥ 0.0254).

Reported per arm, not merely "at least one": the identity of the winning horizon is the
result.

### U5-2 — horizon transfer, not only next-token sharpening

For any arm passing U5-1, at least one of the **deeper** speculative offsets — `+3` or
`+4` — must improve over the control by ≥ 0.022.

`+2` is the *first* speculative slot; an improvement confined to it is a better
next-token predictor, which is a real gain but not evidence that a longer horizon
transferred. If only `+2` moves, the result is recorded as **Mechanism B (generic
sharpening)** rather than **Mechanism A (shared short-horizon representation)**, per
brief §18, and the verdict must say so.

*(Note on labels: this document uses U4's convention, `+j` = supervised slot `j-1`, so
`+1` is the adapter-off seed row. Under a labelling where `+1` is the first speculative
slot, this gate reads "at least one of +2 or +3". The substantive test is the same:
something deeper than the nearest speculative position must move.)*

### U5-3 — algorithmic benefit

The same arm's K=4 **TPF** exceeds the control's by both conditions in §3
(CI excludes zero, Δ ≥ 0.0062).

### U5-4 — wall-clock benefit

At least one longer-horizon arm's K=4 **tok/s** exceeds the control's, with the paired
95% CI excluding zero. Because the U4 median estimator proved fragile at 15 prompts,
this gate is scored on the **paired per-prompt mean difference**, and the median is
reported alongside without being decisive.

Additionally, the gain must exceed **0.22 tok/s** — the upper end of U4's measured
within-session repeat noise.

### U5-5 — no degradation (required)

* verified decoding remains bit-exact lossless at `K_decode = 4` in the cacheless
  reference regime for every arm;
* in the cached regime, AR agreement is at least the backbone's own chunking floor;
* no arm's K=4 accepted prefix falls below the control by more than the §3 floor.

## 5. Descriptive milestones

Not gates. Reported for scale, per brief §17, on K=4 wall clock vs the matched control:

```
+2%   meaningful transfer
+5%   strong
+10%  very strong
```

Training is not to be adjusted to reach any of these.

## 6. Mechanism analysis (reported, not gated)

**Offset profile.** Plot `agreement(arm) − agreement(A_k4)` against future offset. A
gain concentrated at `+2` is Mechanism B; a gain spread across `+3`/`+4` is Mechanism A.

**Draftability stratification.** K=4 acceptance by frozen-model draftability quintile,
per arm. The missing-predecessor-bridging hypothesis predicts that longer-horizon
training preferentially improves the **high-gap** (least draftable) quintiles, which is
the direction U3 found for longer training at fixed K. Recorded as support or
non-support, not as a gate — U4 measured these per-arm correlations moving by ±0.05
across three runs of an *identical* adapter.

**Transfer efficiency.** `ΔA_h` per training hour. The arms are compute-matched to
~2.5%, so this will rank identically to `ΔA_h`; it is reported because "per unit
compute" is the practical question and because the near-equality is itself the point.

## 7. What U5-A cannot conclude

Stated before running so it cannot be quietly dropped afterwards.

Longer horizons necessarily supervise more future positions per step (K=8 supervises 7
adapted positions, K=4 supervises 3). A positive result is therefore consistent with
**horizon** (the positions are further out) *or* **density** (there are simply more of
them). U5-A cannot separate these.

The experiment that does, named now: train at K=8 with the loss **masked to the first
four supervised positions** — same block, same noisy context, same supervised count as
K=4. That is U5-B and is out of scope here.

## 8. Stop conditions

An arm is stopped, and the stop reported, if:

* the backbone fingerprint changes (voids the stage);
* the saturation guard fires (vanishing gradient with saturated logits);
* the loss is non-finite;
* held-out TV rises for 5 consecutive in-training evaluations (500 steps).

Heavy execution stops immediately, cleanly and without touching anything if the
resource guard detects a protected process. **Falling training loss is not a reason to
continue.**

## 9. Verdict

One primary verdict, chosen only after the gates above are scored:

```
CROSS-HORIZON TRANSFER CONFIRMED
LONGER TRAINING HORIZON HELPS, BUT NOT SPEED
CURRICULUM TRANSFER ONLY
K6 OPTIMAL FOR K4
K8 OPTIMAL FOR K4
NO CROSS-HORIZON TRANSFER
U4 OBSERVATION WAS NOISE
```

## 10. Out of scope

No dynamic-K routing: no entropy router, no draftability router, no acceptance-history
router, no learned router. No mixed-horizon objective. U4 established K=4 as the best
static runtime; U5 asks only how good a K=4 drafter training can make.

Public phrasing, if it is confirmed, is conservative and names the setting:

> Cross-horizon transfer in Uno diffusion distillation: on Qwen3.5-4B under MLX on an
> Apple M4 Max, in our Uno reproduction, training on longer speculative blocks improved
> shorter-horizon lossless decoding.

Not: "longer diffusion training always improves short decoding."

## 11. Amendments

### A1 — 2026-09-13: training is not reproducible across launches

**Written during the primary run, before any U5 checkpoint was evaluated.** No U5
result exists at the time of writing. No threshold, gate or verdict rule in §0–§10 is
changed by this amendment.

**Observation.** Arm `A_k4` was launched twice from the identical start
(`runs/u4a/step-12800`), with identical arguments, seed and restored data position. The
first launch was killed at step 13090 by an unrelated restart of the controlling
session, before its first checkpoint (13200), so it wrote no adapter and contributes no
data. Its log is kept: `runs/u5-launch.attempt1-killed-at-13090.log`. The two logs
overlap for five logged steps:

| step | loss (launch 1) | loss (launch 2) | \|g\| (launch 1) | \|g\| (launch 2) |
|---|---|---|---|---|
| 12800 | 0.2775 | 0.2775 | 12.359 | 12.351 |
| 12810 | 0.4221 | 0.4168 | 8.470 | 8.098 |
| 12820 | 0.7120 | 0.7129 | 8.021 | 7.761 |
| 12830 | 0.6479 | 0.6475 | 7.732 | 7.782 |
| 12840 | 0.2433 | 0.2450 | 3.991 | 4.164 |

The loss is identical at step 12800, so the starting weights and the first batch match.
The gradient norm already differs there, and from step 12810 the trajectories diverge:
max |Δloss| 0.0053, gradient norm up to 4.4% apart within 40 steps.

**Interpretation.** On the 4B backbone under MLX on Metal, two launches of the same
training run do not follow the same trajectory. The cause is not established: five
steps are too few, and nondeterministic GPU kernels in the bf16 backward pass are the
likeliest explanation, not a demonstrated one. This does not contradict earlier
reproducibility claims. The bit-identical results in Act IV-U3/U4 concerned
*evaluation* (forward passes only). The bitwise-resume requirement in U5-0 is scored by
`tests/test_uno_resume.py`, which this amendment does not touch, and which does not
cover the 4B backbone across separate launches.

**Consequence.** The floors in §2 and §3 are the spread of *evaluations* of one frozen
adapter. They contain no training-trajectory variance. A U5 difference between an arm
and the control is therefore the horizon effect **plus** launch-to-launch training
noise, and the frozen floors cannot tell these apart. An arm can clear every §3
condition on training noise alone.

### A2 — 2026-09-13: replicate launches of the control

Added as a control, not as a change to the protocol.

* The primary run (`scripts/u5_launch.sh`) proceeds **unmodified**, and U5-0…U5-5 and
  §9 are scored exactly as frozen.
* After it completes, `scripts/u5_replicates.sh` trains two more launches of `A_k4`
  (`A_k4_r2`, `A_k4_r3`) with byte-identical arguments to the primary `A_k4`, including
  the checkpoint schedule, because U3-0b showed that schedule changes can perturb
  training.
* All three `A_k4` launches and the `B_k6`, `C_k8`, `D_curr` endpoints are then evaluated
  in **one supplementary paired session** at `K_decode = 4`
  (`runs/u5/eval_replicates.json`). This is required, not a convenience: between-session
  evaluation noise (±4%) would otherwise swamp the quantity being measured.
* The replicate adapters are compared by tensor digest
  (`results/act4u5/u5_replicate_check.json`). If all three are identical, A1's
  observation did not generalise to a full run, and that is reported.

### A3 — 2026-09-13: how the replicates are read, fixed before they exist

For each of mean accepted prefix, TPF and tok/s at `K_decode = 4`, measured in the
supplementary session:

```
training-noise band  =  max − min across the three A_k4 launches
arm effect           =  arm − mean(three A_k4 launches)
```

A range is used rather than an sd because an sd from three launches is too poorly
estimated to be worth its apparent precision.

For every arm that passes U5-1, U5-3 or U5-4 **as frozen**:

* arm effect **> band** → reported as *passes, and exceeds measured training noise*;
* arm effect **≤ band** → reported as *passes the frozen criterion but is not
  distinguishable from launch-to-launch training noise*.

The §9 verdict is still chosen from the frozen list. If every passing arm falls in the
second case, the verdict line must carry that qualification in the same sentence.
`U4 OBSERVATION WAS NOISE` stays available, and this is exactly the case where it
should be considered.

**Stated limitation.** The band measures the control's training noise only. Reading it
as the noise of the longer-horizon arms assumes their training noise is similar. That is
plausible, since every arm uses the same optimizer, data and step count, but it is not
measured. Replicating `C_k8` would test it and is not scheduled.

### A4 — 2026-09-13: supplementary C_k8 launch-sensitivity check

**Written before any U5 endpoint was evaluated.** When this was written, the primary run
was still training its first arm (`A_k4`), and no U5 evaluation of any kind had run.

What this amendment does **not** do comes first:

* It does not modify U5-0 through U5-5, the §3 floors, or any threshold.
* It does not modify §9 verdict selection.
* It does not replace the original `C_k8`. The original `C_k8`, evaluated in the primary
  sessions, is the only C arm used to score the frozen gates.
* It does not permit averaging the C launches, choosing the better one, or feeding either
  replicate into U5-1, U5-3 or U5-4.
* It is a supplementary robustness check only. Its result may qualify the interpretation;
  it cannot change the scorecard.

**Rationale.** A3 estimates training noise from three launches of `A_k4` and applies that
band to `B_k6`, `C_k8` and `D_curr`. That assumes their launch-to-launch variability is
comparable to the control's, which A3 itself records as plausible but unmeasured.
`C_k8` is the arm closest to U4's unplanned K=8-training → K=4-decoding observation, so
it is where launch sensitivity matters most. One independent launch, `C_k8_r2`, tests
whether the C effect is visibly launch-sensitive. **Two C launches cannot estimate C's
variance**, and nothing here describes them as doing so. This supersedes A3's "not
scheduled" sentence; A3 is otherwise unchanged.

**Identity.** `C_k8_r2` resumes from the same start (`runs/u4a/step-12800`) with the same
arguments as the primary `C_k8` invocation in `u5_launch.sh`: horizon K=8, 16000 steps,
learning rate, optimizer, corruption, noise alignment, checkpoint schedule, seed, data,
code, environment and resource guard. Only the output directory differs. The only
intended difference is the launch-to-launch nondeterminism MLX/Metal already shows (A1).
The launches are not forced to differ. Provenance, recorded by
`scripts/u5_replicates.sh` and checked by `scripts/u5_replicate_provenance.py` into
`results/act4u5/u5_replicate_provenance.json`:

* the start checkpoint immediately before each replicate launch: step, data position,
  config hash, and adapter and optimizer file and tensor digests, compared with the
  record the primary run wrote at its own launch. That record is preserved as
  `results/act4u5/u5_start_check.at_primary_launch.json`, a byte-identical copy made
  during the primary run, because `u5_launch.sh` rewrites `u5_start_check.json` after
  training;
* the primary `C_k8` command line, captured read-only from the process table while it
  runs, compared token for token with the replicate's, excluding `--out`;
* digests of the training code and the Python/MLX environment at orchestrator start, at
  primary exit, and before each replicate launch;
* at step 16000: config hash, seed, block size, step, data position and RNG state, all of
  which must match; adapter and optimizer digests, reported as identical or diverged.

A replicate that fails any identity check is reported as **not a valid replicate** and
is not interpreted.

**Scheduling.** Supplementary phase only: `A_k4_r2`, `A_k4_r3`, then `C_k8_r2`. They start
after `u5_launch.sh` has exited, and only if all four primary arms and both primary
evaluations exist. `u5_launch.sh` is not modified. Nothing the primary run still executes
(`scripts/uno.py`, `u5_start_check.py`, `u5_report.py`, `src/qdif/uno/`) is modified
while it runs, which is why the supplementary analysis is a new script,
`scripts/u5_replicate_report.py`.

**Evaluation.** A2's supplementary session is extended to seven endpoints, all at step
16000 and `K_decode = 4`, in one `u5-eval` invocation with the same prompts, held-out
rows, draft-noise streams, evaluator and session: `A_k4`, `A_k4_r2`, `A_k4_r3`, `B_k6`,
`C_k8`, `C_k8_r2`, `D_curr`. `C1` below is the original `C_k8` **as re-evaluated in this
session**, not its primary-session number, so C1 and C2 are compared without
between-session noise.

A known harness property, not changed: `u5-eval` records each prompt's tok/s as the best
of three repeats. Repeats are keyed and decode identical tokens, so this selects on
timing jitter only (0.05–0.22 tok/s in U4), and it applies identically to every arm in
the session.

**Definitions.** All metrics are at `K_decode = 4` in the supplementary session:
free-running mean accepted prefix (`mean_accepted_specs`, U5-1's metric and the
**principal** metric), TPF (`tokens_per_forward`), tok/s (mean over prompts of per-prompt
tok/s, as U5-4 scores it), and teacher-forced agreement at offsets `+2`, `+3` and `+4`.

```
A_band     = max(A1, A2, A3) − min(A1, A2, A3)
C1_effect  = C_k8     − mean(A1, A2, A3)
C2_effect  = C_k8_r2  − mean(A1, A2, A3)
```

**Interpretation, fixed now.** "Exceeds the band" means strictly greater than `A_band`, in
the positive direction. On the principal metric each condition is evaluated
independently:

| case | condition | statement |
|---|---|---|
| 1 | C1 and C2 both positive and both exceed `A_band` | *The C_k8 effect replicated across two independent training launches and both exceeded the measured A_k4 launch spread.* |
| 2 | exactly one of C1, C2 exceeds `A_band` | *The C_k8 effect is launch-sensitive; the original frozen U5 result may be real but is not robust to an independent C training launch.* |
| 3 | \|C1 − C2\| > `A_band` | *The assumption that A_k4 launch spread is representative of C_k8 training variability is not supported by this check.* |
| 4 | neither exceeds `A_band` | *Independent C_k8 replication provides no evidence that the U4/U5 effect exceeds measured launch-to-launch training noise.* This materially strengthens `U4 OBSERVATION WAS NOISE`. |
| 5 | C1 and C2 have opposite signs | *C_k8 is highly launch-sensitive under this training setup; no stable positive horizon-training effect is established.* |

The headline is case 5 if it holds, else case 1, else case 2, else case 4. Case 3 is not a
headline: it is reported whenever it holds, including alongside case 1. Every other
condition that holds is reported as well, for example case 2 beneath a case-5 headline.
Case 1 is strong robustness evidence, not a variance estimate.

TPF and tok/s receive the same classification, reported beside the principal one. If
the metrics disagree, the disagreement is reported and not resolved in favour of any
metric.

Two degenerate situations are reported as such. If the three `A_k4` adapters are
tensor-identical, `A_band` is zero by construction and the classification is
uninformative. If `C_k8` and `C_k8_r2` are tensor-identical, the check says nothing
about C launch sensitivity.

**Mechanism profile.** For each C launch, at `+2` (the first speculative slot) and at
`+3` and `+4` (deeper):

```
Δ_j = agreement_j(C) − mean over A launches of agreement_j
```

An offset *moves* if `Δ_j ≥ 0.022` **and** `Δ_j` exceeds the `A_k4` launch band at that
offset. The 0.022 is U5-2's frozen floor, used here only as a descriptive reference.

A launch is labelled `deep` if `+3` or `+4` moves, `first-slot only` if only `+2`
moves, and `none` otherwise. The profile *replicates* if both launches get the same
label and that label is not `none`. "Performance replicates" means case 1 on the
principal metric. The replication outcome is exactly one of:

* performance and mechanism profile both replicate. The profile is named, and the phrase
  "deeper-offset profile replicates" is used only when the label is `deep`;
* performance replicates but the profile does not;
* the profile replicates but performance does not;
* neither replicates.

The frozen U5-2 rule is unchanged and is scored from the primary session only.

**Strength of language.** Two C launches do not establish C's standard deviation, a
confidence interval across training launches, heteroskedasticity, or a distribution of
K=8 outcomes. The check answers only whether the C effect survives one independent
rerun, and whether the difference between the two launches looks roughly compatible
with the A_k4 launch spread.

**Outputs.** `results/act4u5/u5_replicate_check.json`,
`results/act4u5/u5_replicate_provenance.json`, `runs/u5/eval_replicates.json` (with the
per-prompt and per-row data the harness records), and
`results/act4u5/u5_supplementary_robustness.{json,md}`. In the final U5 report these
appear under **Supplementary training-launch robustness**, separate from the
preregistered scorecard.
