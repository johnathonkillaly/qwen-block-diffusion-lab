# RPRM diffusion — Stage 1 results

**Date:** 2026-09-08
**Repo / commit:** `diffusion_project` @ `13e9d05`
**Pre-registration:** [`RPRM_DIFFUSION_PREREG.md`](RPRM_DIFFUSION_PREREG.md), frozen before these numbers existed
**Audit:** [`RPRM_DIFFUSION_STAGE0_AUDIT.md`](RPRM_DIFFUSION_STAGE0_AUDIT.md)

---

## Verdict

# FAIL

The verdict is FAIL on the pre-registered criteria, in **both** strata and under
**both** adapters, because of **criterion 3**: adding denoiser uncertainty to the
progress-only baseline improves held-out AUROC by **+0.022**, against a
pre-registered bar of **≥ 0.05**. Since a WEAK PASS also requires criterion 3, the
result does not reach WEAK PASS either.

This is not a null result. Five of the six criteria pass: the effect is
directionally real, monotone, survives conditioning on diffusion progress, clearly
beats the shuffled control, and adds information to the existing draftability
predictor. It is simply **less than half the size** the pre-registration demanded
of it. Per the brief, that is exactly the case the FAIL clause
*"entropy adds negligible information to simple baselines"* was written to catch,
and no threshold has been moved.

**A FAIL is a successful experiment.** What follows reports the effect at its
actual size, including the parts that look good.

---

## 1. Harness validation — the new code reproduces the published numbers

Before reading anything new, the `replication` run uses the **same adapter**
(`runs/uno-k4-true`) and **same block size** (K=8) as the published draftability
result, on different windows, different noise keys, and 3.3× the sample.

| Pearson vs acceptance | Published (n=2,688) | This harness (n=8,960) |
|---|---|---|
| existing draftability gap | −0.2114 | **−0.2339** |
| teacher entropy | −0.0968 | **−0.0832** |
| gap vs teacher entropy | −0.2535 | **−0.2617** |

Quintile tables agree just as closely (gap: published 0.263/0.149/0.155/0.123/0.074
vs here 0.255/0.169/0.144/0.113/0.080). **The prior work reproduces**, including
its signature anomaly: teacher entropy's non-monotone q0→q1 inversion
(published 0.149→0.207; here 0.129→0.205) appears again, independently.

The measurement path is therefore the existing one, unchanged. `P_full` and
`P_corrupt` come from the same code as `cmd_draftability`.

---

## 2. What was measured

| | |
|---|---|
| Positions | **17,920** per adapter (8,960 per stratum) |
| Windows | **1,280** (WikiText-103 validation split; never trained on) |
| Split | window level — FIT 640 / VAL 256 / **TEST 384** (all reported numbers are TEST, n=2,688 per stratum) |
| Block size | K=8, offsets +2 … +8, seed slot excluded |
| Vocab | 248,320 (`log|V|` = 12.42) |
| Primary adapter | `runs/u4b2-k8` (K=8, 16,000 steps) |
| Replication adapter | `runs/uno-k4-true` (K=4, 400 steps) |
| Seeds | analysis/split/shuffle/bootstrap all `20260908`; explicit MLX keys throughout |

Data: [`results/rprm_stage1/positions_primary.csv`](results/rprm_stage1/positions_primary.csv),
[`positions_replication.csv`](results/rprm_stage1/positions_replication.csv).
Code: [`scripts/rprm_stage1_extract.py`](scripts/rprm_stage1_extract.py),
[`scripts/rprm_stage1_analyze.py`](scripts/rprm_stage1_analyze.py).

Held-out acceptance: **0.175** (FULL, primary), 0.539 (SCHEDULE, primary).

---

## 3. The seven pre-registered models — held-out TEST AUROC

### FULL stratum (`t = 1`, the deployment condition) — carries the verdict

| # | Model | AUROC (primary) | log loss | AUROC (replication) |
|---|---|---|---|---|
| 1 | progress only (offset) | 0.7081 | 0.4184 | 0.6634 |
| 2 | **denoiser uncertainty only** | **0.6705** | 0.4301 | 0.6547 |
| 3 | existing draftability gap only | 0.5988 | 0.4561 | 0.6392 |
| 4 | progress + uncertainty | **0.7298** | 0.4026 | 0.6875 |
| 5 | progress + existing | 0.7207 | 0.4170 | 0.6941 |
| 6 | **progress + existing + uncertainty** | **0.7382** | **0.4016** | **0.7127** |
| 7 | shuffled control (mean, 20 perms) | 0.7078 `[0.7042, 0.7105]` | 0.4186 | 0.6631 `[0.6591, 0.6668]` |
| 8 | progress + *teacher* entropy (already tested) | 0.7180 | 0.4152 | 0.6751 |

**Key comparisons (primary, FULL):**

| Comparison | AUROC Δ | clustered 95% CI | bar | met? |
|---|---|---|---|---|
| **4 vs 1** — uncertainty beyond progress | **+0.0217** | `[+0.0010, +0.0414]` | ≥ 0.05 | **no** |
| **6 vs 5** — uncertainty beyond the existing predictor | +0.0175 | `[−0.0058, +0.0389]` | ≥ 0.02 | no |
| **6 vs 5** — log loss reduction | **0.0154 nats** | `[+0.0045, +0.0258]` | ≥ 0.01, CI > 0 | **yes** |

### SCHEDULE stratum (`t ~ U(0,1)`) — secondary

| # | Model | AUROC (primary) | AUROC (replication) |
|---|---|---|---|
| 1 | progress only (offset, `t`, corruption) | 0.8324 | 0.8541 |
| 4 | progress + uncertainty | 0.8522 | 0.8739 |
| 5 | progress + existing | 0.8450 | 0.8691 |
| 6 | progress + existing + uncertainty | 0.8660 | 0.8884 |
| 7 | shuffled control | 0.8326 `[0.8320, 0.8333]` | 0.8539 |

4 vs 1: **+0.0198** `[+0.0115, +0.0281]`. 6 vs 5: +0.0210 `[+0.0125, +0.0294]`,
log loss −0.0244 nats `[+0.0142, +0.0350]`.

**The strata agree.** Adding the timestep axis does not change the conclusion:
the increment is ~+0.02 AUROC everywhere.

---

## 4. Pre-registered criteria

| # | Criterion | Bar | FULL primary | FULL replication | SCHEDULE primary |
|---|---|---|---|---|---|
| 1 | monotone across held-out quintiles | ≤1 inversion, ≤2pp | ✅ **0 inversions** | ❌ +3.2pp at q2→q3 | ✅ |
| 2 | visible within individual cells | ≥4 of 7 offsets | ✅ **6/7** | ✅ 5/7 | ✅ 7/7 |
| 3 | **AUROC gain over progress-only** | **≥ 0.05** | ❌ **+0.0217** | ❌ +0.0242 | ❌ +0.0198 |
| 4 | beats within-cell shuffled control | > 97.5th pct | ✅ 0.7298 > 0.7105 | ✅ | ✅ |
| 5 | adds to existing predictor | ΔAUROC ≥0.02 **or** Δlogloss ≥0.01 | ✅ **via log loss** | ✅ | ✅ |
| 6 | bootstrap supports nontrivial effect | CI excludes 0 | ✅ `[+0.0010, +0.0414]` | ✅ | ✅ |

**Criterion 3 fails everywhere → FAIL.** WEAK PASS requires criterion 3, so it is
not reached.

---

## 5. Descriptive results

### 5a. Acceptance by denoiser-entropy quintile (FULL, primary, TEST)

| quintile | n | mean H^norm | acceptance | 95% CI |
|---|---|---|---|---|
| q0 (lowest) | 469 | 0.142 | **0.3945** | 0.351 – 0.439 |
| q1 | 597 | 0.274 | 0.1625 | 0.135 – 0.194 |
| q2 | 591 | 0.335 | 0.1235 | 0.099 – 0.153 |
| q3 | 522 | 0.381 | 0.1169 | 0.092 – 0.147 |
| q4 (highest) | 509 | 0.456 | **0.1081** | 0.084 – 0.138 |

Monotone, no inversion, and a **3.6× spread** between extremes. Compare the two
incumbents on the same TEST positions:

| quintile | denoiser H | existing gap | **teacher H** (already tested) |
|---|---|---|---|
| q0 | **0.3945** | 0.2438 | 0.1843 |
| q1 | 0.1625 | 0.1845 | **0.2246** ← inversion |
| q2 | 0.1235 | 0.1969 | 0.1945 |
| q3 | 0.1169 | 0.1411 | 0.1504 |
| q4 | 0.1081 | 0.1107 | 0.1161 |

**This is the clearest new finding.** The low-end pathology that made *teacher*
entropy unusable as a routing signal — documented in `docs/act4u_results.md` and
`draftability_gap.md`, and reproduced here — **does not appear for denoiser
entropy.** The denoiser's own confidence is monotone where the teacher's is not,
and its lowest-uncertainty bucket is its *best* bucket by a wide margin. The prior
work's diagnosis predicts this: teacher entropy is low exactly when a token is
determined by predecessors the draft cannot see, whereas the denoiser's entropy is
computed *from what the draft can actually see*.

See [`plot1_quintiles_primary.png`](results/rprm_stage1/plot1_quintiles_primary.png).

### 5b. Denoiser entropy and the draftability gap are nearly independent

Pearson over all 8,960 FULL positions:

| | primary | replication |
|---|---|---|
| denoiser entropy vs acceptance | **−0.2778** | −0.2383 |
| existing gap vs acceptance | −0.1769 | −0.2339 |
| teacher entropy vs acceptance | −0.0760 | −0.0832 |
| **gap vs denoiser entropy** | **+0.0475** | +0.0973 |

On the well-trained adapter, denoiser entropy is the **stronger** univariate
predictor (−0.278 vs −0.177), and it is nearly **orthogonal** to the gap
(r ≈ +0.05). That orthogonality is why model 6 beats model 5 on log loss even
though each signal alone is modest.

### 5c. Within-offset — where each signal lives

FULL, primary, TEST; AUROC of *negated* predictor vs acceptance (0.5 = no signal):

| offset | n | acceptance | AUROC(−denoiser H) | AUROC(−gap) |
|---|---|---|---|---|
| +2 | 384 | 0.4609 | **0.6686** | 0.5675 |
| +3 | 384 | 0.2344 | **0.6475** | 0.5126 |
| +4 | 384 | 0.1354 | **0.6339** | 0.5908 |
| +5 | 384 | 0.1146 | **0.6801** | 0.5094 |
| +6 | 384 | 0.1198 | 0.5164 | **0.6142** |
| +7 | 384 | 0.0911 | 0.4935 | **0.6958** |
| +8 | 384 | 0.0703 | 0.5869 | 0.5124 |

The signal is genuinely *within*-offset — it is not the progress variable in
disguise — but it is **concentrated at short horizons (+2 … +5) and disappears at
+6/+7**, precisely where the gap takes over. The two predictors are complementary
in horizon as well as in correlation. This is also why the pooled increment is
small: the offsets where entropy works are a minority of positions, and the
offsets where it fails are the ones with the least acceptance to predict.

See [`plot2_within_offset_primary.png`](results/rprm_stage1/plot2_within_offset_primary.png).

### 5d. Shuffled control

Permuting entropy within offset, 20 times, collapses model 4 to
**0.7078 `[0.7042, 0.7105]`** — indistinguishable from progress-only (0.7081).
Real entropy scores 0.7298, well outside that range. **The signal is
token-specific, not a repackaging of offset.**
See [`plot3_shuffled_control_primary.png`](results/rprm_stage1/plot3_shuffled_control_primary.png).

### 5e. Diagnostic alternatives — reported, not used to change the verdict

Pre-registered as diagnostics only (prereg §4). Increment over progress-only,
FULL primary:

| measure | AUROC | Δ vs progress |
|---|---|---|
| normalized entropy *(primary measure)* | 0.7298 | +0.0217 |
| **top-1 probability** | **0.7439** | **+0.0358** |
| top-1 − top-2 margin | 0.7436 | +0.0355 |
| effective support `e^H` | 0.7210 | +0.0129 |

Top-1 probability is ~65% better than entropy as an increment — but **still below
the 0.05 bar**, and in the replication it is *worse* than entropy (+0.0263 vs
+0.0242). It does not change the verdict and is recorded as a hypothesis for
future work, not as a result.

### 5f. Other artefacts

* [`plot4_roc_pr_primary.png`](results/rprm_stage1/plot4_roc_pr_primary.png) — ROC/PR for models 1, 4, 5, 6
* [`plot5_calibration_primary.png`](results/rprm_stage1/plot5_calibration_primary.png) — calibration for 1, 5, 6
* [`quintiles_primary.csv`](results/rprm_stage1/quintiles_primary.csv), [`within_offset_primary.csv`](results/rprm_stage1/within_offset_primary.csv), [`analysis_primary.json`](results/rprm_stage1/analysis_primary.json) (and `_replication` twins)

---

## 6. What this does and does not show

**Does show.** On this model, corpus, adapter and block size, the denoiser's own
per-token uncertainty carries **real, token-specific, held-out** information about
whether the frozen target will accept the proposal. It survives conditioning on
diffusion progress, beats a within-cell shuffled control decisively, is monotone
where the previously-tested teacher entropy is not, and is close to orthogonal to
the existing draftability gap.

**Does not show.** That the information is **large enough to matter**. +0.022
AUROC over a baseline that already reaches 0.708 is not a basis for building
anything. The signal is also horizon-limited: it vanishes at exactly the long
offsets where drafting fails most and where a routing signal would be worth the
most.

**Also does not show** anything about other models, corpora, block sizes or
adapters. Single machine, single seed per adapter, WikiText only, greedy
acceptance only.

**On the deployability asymmetry.** As recorded in the pre-registration *before*
results: the gap needs `P_full` and is not computable at decode time, whereas
denoiser entropy is free from the draft forward. That remains true and is why the
6-vs-5 log-loss gain is worth noting. It does **not** rescue the verdict, and it
is not being used to.

**RPRM interpretation, kept separate from the result.** The measured quantity is
an *RPRM-motivated ambiguity/pressure proxy* — never "RPRM pressure". The
correspondence under test was `H(X₀|X_t)` high while `H(Q|X_t)` already low. What
this experiment can say is narrow: the denoiser's entropy is a *weakly* informative
proxy for the receiver's decision, and markedly better-behaved than the teacher's
entropy — but the two entropies decouple from acceptance quickly as the horizon
grows. Nothing here establishes the RPRM correspondence, and nothing here refutes
it; it constrains one candidate proxy.

---

## 7. Stage 2 — NOT RUN

Stage 1 did not pass, so Stage 2 is not run. This was the pre-registered rule and
it is applied unchanged.

Independently, the Stage 0 audit established (§8a) that Stage 2 as specified is
**not runnable in this architecture at all**: Uno drafting is single-shot — one
adapter forward produces all K−1 proposals — so there are no per-position
denoising updates to avoid, and "freeze position *i* when `H_i < τ`" saves no
computation. This was written down *before* Stage 1 results existed, so that Stage 2
could not later be redefined to fit them.

---

## 8. Summary

```
Hypothesis                              Result
-------------------------------------------------------
Entropy predicts acceptance globally    PASS
Signal survives timestep control        PASS
Beats shuffled control                  PASS
Adds beyond draftability predictor      PASS (log loss); FAIL (AUROC bar)
Supports useful offline early exit      NOT RUN
-------------------------------------------------------
Pre-registered effect-size bar          FAIL  (+0.022 AUROC vs >=0.05 required)
```

# Final verdict: STOP

The hypothesis is *directionally* correct and the mechanism is more plausible than
it was before — denoiser entropy is better-behaved than the teacher entropy this
project already rejected, and it is nearly orthogonal to the draftability gap. But
the effect is less than half the pre-registered size, it is concentrated at the
short horizons that need help least, and the architecture it would serve has no
early-exit lever to pull.

STOP is the honest call. The pre-registered bar was set before the numbers existed
and it was not met.

If this line is ever resumed, the audit trail points at two specific things rather
than a general "try harder": (a) top-1 probability and the top-1/top-2 margin
behaved slightly better than entropy and were not the pre-registered measure, and
(b) entropy and the gap are complementary *by horizon*, which a single pooled model
cannot exploit. Both would need their own pre-registration, and neither is
justified by this result on its own.

---

## Reproducing

```bash
export HF_HOME=/Volumes/SHUTTLE
.venv-unsloth/bin/python scripts/rprm_stage1_extract.py \
    --adapter runs/u4b2-k8/adapter.safetensors --block-size 8 --batches 320 --tag primary
.venv-unsloth/bin/python scripts/rprm_stage1_analyze.py --tag primary
```

~35 min for extraction on an M-series Mac; analysis is seconds and touches no model.
