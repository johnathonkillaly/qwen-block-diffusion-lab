# RPRM diffusion — Stage 1 pre-registration

**Date frozen:** 2026-09-08
**Repo / commit:** `diffusion_project` @ `13e9d05` (Act IV-U4)
**Status at freeze:** no correlation, AUROC, or bucket table between denoiser
uncertainty and acceptance has been computed. `RPRM_DIFFUSION_STAGE0_AUDIT.md` was
written first and contains no such number.

**This document is frozen once results exist.** Amendments are appended with a
date and a reason, never edited in place — the house rule from `CLAUDE.md`.

---

## 0. Deviations from the brief, declared before results

Three deviations. All follow from architectural facts established in
§8 of the Stage 0 audit, and all are declared here, before any result is seen.

### 0a. "Timestep" is replaced by a two-part progress variable

Uno drafting is **single-shot**: one adapter forward proposes all `K−1`
speculative tokens, and every evaluation path in the existing harness hardcodes
`corruption="full"`. There is no denoising schedule at inference and therefore no
timestep to condition on there.

Two legitimate progress variables exist instead, and the confound control uses
**both**:

* **`offset` (`j`)** — the future offset of the position within the draft block,
  `+2 … +K`. This is the deployment-relevant progress axis, it is what the
  existing `_draftability_by_offset` code already conditions on, and it is the
  dominant driver of acceptance (0.44 at +2 → 0.04 at +8 in prior results).
* **`t` (`corruption_rate`)** — the Uno/SDAR per-sequence noise rate `t ~ U(0,1)`
  from `build_uno_batch(corruption="uniform")`. Every U3/U4 adapter was **trained**
  on this whole schedule, so `t` is in-distribution for the model. It is
  nonetheless **off-manifold for deployment**, which is why it is measured in a
  separate stratum (§2) rather than pooled.

Wherever the brief says "within-timestep", this pre-registration reads **"within
`offset` (FULL stratum) and within `offset × t`-bucket (SCHEDULE stratum)"**.

### 0b. Stage 2 as literally specified is not runnable and is not pre-registered here

The brief's Stage 2 metric — "percentage of token-position denoising updates
avoided" — has no referent in a single-shot architecture. Freezing a position
saves nothing: the same single forward produces all `K−1` proposals whether or not
any position is notionally frozen. Stage 2 is therefore **not** pre-registered in
this document. If Stage 1 passes, a Stage 2 pre-registration will be written
separately, for the nearest real lever (**adaptive block width**), with its own
thresholds fixed before evaluation. Declaring this now prevents Stage 2 from being
retro-fitted to whatever Stage 1 produces.

### 0c. `corruption level` is recorded as three fields, not one

`corrupted` (per-position flag), `n_corrupted_predecessors` (how many of the draft
rows *before* this position hold noise — the quantity the draftability hypothesis
says should matter), and `corruption_rate` (`t`, per sequence).

Nothing else in the brief is changed. Verdict thresholds are used **exactly** as
given.

---

## 1. Hypotheses

### H1 (primary)

At fixed diffusion progress, **lower denoiser uncertainty predicts higher
probability that the target model accepts the proposed token.**

The denoiser distribution at position *i* is

```
p_θ(x_{0,i} | x_t, t)  ==  softmax( model.draft_logits(student_ids, lora_mask)[i] )
```

with the **trained** adapter loaded — i.e. the adapter's own output distribution on
the noised block. Raw and normalised entropy:

```
H_i      = −Σ_v p_v log p_v
H_i^norm = H_i / log|V|
```

### H1-null (what would falsify it)

Denoiser uncertainty carries no information about acceptance beyond diffusion
progress (`offset`, `t`) — i.e. it does not beat the within-cell shuffled control,
or its incremental held-out AUROC over the progress-only baseline is negligible.

### H2 (incremental value — the result that would actually matter)

Denoiser uncertainty adds **held-out predictive information beyond the existing
draftability gap**, which is the signal this project already possesses.

---

## 2. Design

### Model under test

| | |
|---|---|
| Backbone | `unsloth/Qwen3.5-4B-Base`, frozen |
| **Primary adapter** | `runs/u4b2-k8/adapter.safetensors` — K=8, 16,000 steps, the most-trained K=8 adapter |
| **Replication adapter** | `runs/uno-k4-true/adapter.safetensors` — the adapter used for the published 2,688-position draftability result, so the new numbers are directly comparable to it |
| Block size | **K = 8** → 7 speculative slots, offsets **+2 … +8** |
| Window width | 128 |
| Corpus | WikiText-103-raw-v1 **validation split** (never trained on) |

The primary adapter is the *strongest* one, deliberately: denoiser entropy is a
property of the trained adapter, and a weakly-trained adapter would be an unfair
test of whether a well-trained denoiser's confidence is informative.

### Two strata

| Stratum | Corruption | Progress variables | Why |
|---|---|---|---|
| **FULL** | `corruption="full"` (`t = 1`) | `offset` only | The **only** condition that occurs at inference. Directly comparable to the existing draftability result. **This stratum carries the verdict.** |
| **SCHEDULE** | `corruption="uniform"`, `t ~ U(0,1)` | `offset`, `t` | Supplies the timestep axis the brief's confound controls require. In-distribution for training, off-manifold for deployment. **Secondary.** |

Target size: **1,280 windows per stratum** (320 keyed batches × 4), giving
**8,960 positions per stratum**. If runtime forces a reduction, the reduction is
declared in the results document and applied to *both* strata equally; the
floor below which the run is abandoned rather than reported is **512 windows
(3,584 positions) per stratum**.

### Splits — window level, never token level

All 7 positions in a window share a context, so token-level splitting would leak.
Windows are assigned by a seeded permutation (`split_seed = 20260908`):

| Partition | Share | Use |
|---|---|---|
| **FIT** | 50% | fitting logistic regressions |
| **VAL** | 20% | any threshold or sweep selection (Stage 2 only; unused in Stage 1) |
| **TEST** | 30% | **every reported held-out number** |

Nothing is fitted or selected on TEST. There is no hyperparameter tuning: all
models are plain unregularised logistic regression on standardised features, so no
tuning partition is needed in Stage 1.

### Reproducibility

Every batch draws an explicit `mx.random` key derived from
`analysis_seed = 20260908` (per pre-registered gate U3-0b — evaluation must consume
no global RNG state). `example_id` encodes `(stratum, batch_index, row_index)` and
is stable across reruns. The emitted dataset is the analysis input; all statistics
are recomputed from it.

---

## 3. Variables recorded

Per position (one row per speculative slot per window):

| Field | Definition |
|---|---|
| `example_id` | stable window identifier — **the unit of splitting and bootstrapping** |
| `stratum` | `FULL` \| `SCHEDULE` |
| `position` | absolute index in the window |
| `offset` | future offset `+2 … +8` (progress variable) |
| `timestep` | `corruption_rate` `t` (`1.0` in FULL) |
| `corruption_level` | per-position `corrupted` flag ∈ {0,1} |
| `n_corrupted_predecessors` | corrupted draft rows strictly before this position |
| `clean_token` | true next token from the corpus |
| `proposed_token` | `argmax` of the denoiser distribution |
| `verifier_token` | `argmax` of the frozen target distribution given true predecessors |
| **`accepted`** | **`proposed_token == verifier_token`** ∈ {0,1} — the primary outcome |
| `accepted_prefix` | secondary: 1 iff accepted *and* every smaller offset in this window was accepted (the decode-time prefix rule) |
| **`entropy`** | `H(p_θ)` — denoiser, **trained adapter**. Primary uncertainty measure |
| **`normalized_entropy`** | `entropy / log|V|` |
| `top1_probability` | denoiser top-1 probability |
| `top1_top2_margin` | denoiser `p₁ − p₂` |
| `effective_support` | `exp(entropy)` |
| `existing_draftability_score` | `TV(P_full, P_corrupt)` — the incumbent, **unchanged** from `cmd_draftability` |
| `teacher_entropy` | `H(P_full)` — the **already-tested** uncertainty (audit §5), kept separate |
| `corrupt_entropy` | `H(P_corrupt)` — frozen model, noised context, adapter **off**. The adapter-free ablation of denoiser entropy |

Slot 0 (the seed row, adapter off, trivially a hit) is **excluded**, matching the
existing analysis exactly.

Emitted as `results/rprm_stage1/positions.csv` (+ `.json` metadata with git SHA,
seeds, and exact sample counts).

---

## 4. Primary uncertainty measure — fixed now

**The verdict is keyed to `normalized_entropy` of the denoiser and to nothing
else.**

`top1_probability`, `top1_top2_margin` and `effective_support` are reported as
diagnostics — to show whether the finding is robust to how uncertainty is
operationalised, or an artefact of one choice. **They are not alternative routes
to a PASS.** If entropy fails and a diagnostic succeeds, the verdict is the
entropy verdict, and the diagnostic is reported as a hypothesis for future work.

---

## 5. Analyses

### A. Within-progress analysis

For each cell with adequate sample size, bucket positions by
`normalized_entropy` and report acceptance with 95% CIs and counts:

* **FULL**: cells = `offset` (+2 … +8).
* **SCHEDULE**: cells = `offset × t`-quintile.

Ask: `P(A=1 | H low, cell) > P(A=1 | H high, cell)`?
Minimum cell size to report a trend: **n ≥ 200** with ≥ 40 per bucket.
Smaller cells are shown with counts but excluded from the verdict.

### B. Baseline vs expanded predictors — the mandatory seven

All logistic regression, features standardised on FIT, all numbers on TEST.
`offset` enters as one-hot (its effect is strongly non-linear).

| # | Model | Features |
|---|---|---|
| 1 | progress only | `offset` (+ `t`, `corruption_level` in SCHEDULE) |
| 2 | denoiser uncertainty only | `normalized_entropy` |
| 3 | existing predictor only | `existing_draftability_score` |
| 4 | progress + uncertainty | 1 + 2 |
| 5 | progress + existing | 1 + 3 |
| 6 | **progress + existing + uncertainty** | 1 + 2 + 3 |
| 7 | **shuffled control** | 1 + shuffled entropy (see C) |
| 8 | *(extra)* progress + teacher entropy | 1 + `teacher_entropy` — the already-tested variable, for context |

Key comparisons: **4 vs 1** (does uncertainty beat progress?) and **6 vs 5** (does
uncertainty add to what we already have?).

### C. Shuffled control

`normalized_entropy` is permuted **within cell** — within `offset` in FULL, within
`offset × t`-quintile in SCHEDULE — preserving the easy progress relationship while
destroying token-specific information. Permutation uses `shuffle_seed = 20260908`
and is repeated **20 times**; the control's reported AUROC is the mean, with the
2.5–97.5 percentile range.

Real entropy must beat this control clearly (§7).

### D. Metrics

AUROC; AUPRC; log loss; acceptance by uncertainty quintile (global and
within-cell); Spearman ρ (descriptive only); calibration curves.

**Bootstrap is clustered at the window level** — resample `example_id`s, not
positions. Positions within a window are correlated, so a token-level bootstrap
would understate the intervals. 2,000 resamples, 95% percentile intervals.

Every table reports `n`.

### E. Plots

1. acceptance vs entropy quintile, globally
2. acceptance vs entropy within offset (and within `t` for SCHEDULE)
3. real entropy vs within-cell shuffled control
4. ROC + PR for models 1, 4, 5, 6, 7
5. calibration curves for models 1, 5, 6

Non-monotone and negative results are plotted as they fall.

---

## 6. Verdict criteria — frozen

Thresholds are the brief's, unchanged. All evaluated on **TEST**, FULL stratum
(the deployment condition). SCHEDULE is reported alongside and must not contradict.

### STRONG PASS — all of:

1. Lower uncertainty predicts higher acceptance **monotonically or near-monotonically**
   across held-out uncertainty quintiles. *Operationalised:* at most one inversion
   between adjacent quintiles, and that inversion ≤ 2 percentage points.
2. The relationship remains visible **within multiple individual cells** —
   *operationalised:* same-signed effect in **≥ 4 of the 7 offsets** with n ≥ 200.
3. Adding uncertainty improves held-out AUROC by **≥ 0.05 absolute** over the
   progress-only baseline (model 4 vs model 1).
4. Uncertainty beats the within-cell shuffled control — *operationalised:* model 4's
   AUROC exceeds the **97.5th percentile** of model 7's 20-permutation distribution.
5. Adding uncertainty to the existing draftability predictor improves AUROC by
   **≥ 0.02 absolute** (model 6 vs model 5), **OR** produces a clearly meaningful
   improvement in held-out log loss — *operationalised:* log-loss reduction ≥ 0.01
   nats with a bootstrap 95% CI excluding zero.
6. Bootstrap CIs support a nontrivial effect — *operationalised:* the clustered
   95% CI on the model-4-minus-model-1 AUROC delta excludes 0.

### WEAK PASS

Criteria 3 **and** 4 hold (uncertainty clearly adds information beyond progress
and beats the shuffled control), but criterion 5 fails.

*Interpretation:* the RPRM-motivated variable is real but largely redundant with
machinery already possessed.

*Recorded now, before results, so it cannot be used as a rescue:* the existing
draftability gap requires `P_full` and is **not computable at decode time**
(audit §4); denoiser entropy is a free by-product of the draft forward. A WEAK
PASS would therefore still be operationally non-trivial. **This does not upgrade a
WEAK PASS to a STRONG PASS, and it does not rescue a FAIL.**

### FAIL — any of:

* the apparent effect disappears after conditioning on `offset` / `t`;
* shuffled entropy performs similarly (model 4 within the shuffled distribution);
* the effect does not generalise to held-out windows;
* uncertainty adds negligible information to the simple baseline (AUROC delta
  < 0.05 for model 4 vs 1);
* the relationship is unstable or non-monotone enough to be operationally useless
  (criterion 1 badly violated — e.g. the low-entropy inversion already documented
  for *teacher* entropy in audit §5 reappearing for *denoiser* entropy).

**A FAIL is a successful experiment.** No post-hoc threshold changes, no switching
the primary uncertainty measure, no redefining acceptance.

---

## 7. Things fixed now that could otherwise be fudged later

* Acceptance rule: per-position teacher-forced `argmax(student) == argmax(teacher)`,
  slot 0 excluded. **Unchanged from `cmd_draftability`.** Not redefined after results.
* Primary uncertainty: `normalized_entropy` of the trained denoiser. Not swapped.
* Primary stratum: FULL. Not swapped to SCHEDULE if FULL disappoints.
* Primary adapter: `runs/u4b2-k8`. The `uno-k4-true` replication is reported
  whatever it shows.
* Splitting: window level, `split_seed = 20260908`. Not re-drawn.
* Bootstrap: clustered on `example_id`. Not switched to token level to tighten CIs.
* Terminology: entropy is an **RPRM-motivated ambiguity/pressure proxy**. It is
  never called "RPRM pressure" as though established.
* Prior checkpoints, `runs/`, `results/act4u*`, and all `docs/act4u*_preregistered_criteria.md`
  are read-only. New outputs go to `results/rprm_stage1/` only.
