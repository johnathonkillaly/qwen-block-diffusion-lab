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

*(none)*
