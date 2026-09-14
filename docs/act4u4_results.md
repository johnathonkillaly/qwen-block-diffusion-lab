# Act IV-U4 — Horizon Scaling

Pre-registered criteria: [`act4u4_preregistered_criteria.md`](act4u4_preregistered_criteria.md),
frozen before any U4 training step.
Machine-readable results: `results/act4u4/` (7 CSVs, `u4_summary.json`, 13 plots).
Raw evaluations: `runs/u4a/eval.json`, `runs/u4b/eval.json`,
`runs/u4-calib/rep{1,2,3}.json`, `runs/u4b/losslessness.json`.

---

## Question

Act IV-U3 showed that acceptance scales with training and converts into wall-clock
throughput. U4 asks the next question: **how far can the useful parallel horizon be
pushed before the extra block size stops paying for itself?**

Two strictly separated stages. `U4-A` trains K=4 to saturation, changing nothing but
duration. `U4-B` tests K=8. No dynamic-K routing.

---

## U3 baseline

U3's best K=4 checkpoint was step 3200: acceptance 0.3415, TPF 1.4314, 56.45 tok/s,
1.168× an in-session AR baseline of 48.35 tok/s. U4 re-measures that same adapter
under a tightened harness and continues from it.

**Two harness changes make U4's numbers not directly comparable to U3's**, and both
are improvements, not adjustments:

1. **Every stochastic input is keyed** (`src/qdif/uno/rng.py`). Held-out corruption,
   the draftability probe and the decode-time draft noise all come from named streams
   indexed by batch or cycle. U3 drew from the global MLX RNG, so each checkpoint met
   *different* noise and part of every checkpoint-to-checkpoint difference was the
   draw. Under keying, the same adapter gives bit-identical teacher-forced numbers in
   two different sessions a day apart — `step-12800` scored val TV 0.9674, agreement
   0.3125, TPF 1.4035 in both the U4-A and U4-B sessions.
2. **The cost model gained two measured terms** (§ Decoder cost model v2).

Re-measured under the new harness, U3's step-3200 adapter scores acceptance 0.3122
(not 0.3415) and TPF 1.3873 (not 1.4314) at K=4. The difference is the noise draw, not
a correction to U3: U3's numbers were one sample from a distribution whose sd we can
now quote (§ Measurement).

---

## U4-A: K=4 continuation

Resumed from `runs/u3a/step-3200` and trained to 12,800 — **three times U3's entire
training budget** — with nothing else changed. 9600 steps, 13,437 s, backbone
unchanged.

All seven checkpoints evaluated in one session against an in-session AR baseline of
**48.93 tok/s** (historical U3 baseline 48.35, +1.2%).

| step | val TV | tf agree | +2 | +3 | +4 | accept | mean prefix | TPF | fwd/tok | tok/s (med) | tok/s (mean) | ×AR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3200 | 0.9745 | 0.3047 | 0.4766 | 0.2656 | 0.1719 | 0.3122 | 0.937 | 1.3873 | 0.7208 | 54.84 | 55.96 | 1.121 |
| 4800 | 0.9775 | 0.2995 | 0.4688 | 0.2422 | 0.1875 | 0.3266 | 0.980 | 1.4090 | 0.7097 | 55.37 | 56.94 | 1.131 |
| 6400 | 0.9734 | 0.3008 | 0.4766 | 0.2656 | 0.1602 | 0.3083 | 0.925 | 1.3820 | 0.7236 | 55.52 | 55.97 | 1.135 |
| 8000 | 0.9597 | 0.3190 | 0.5000 | 0.2852 | 0.1719 | 0.3083 | 0.925 | 1.3820 | 0.7236 | 55.56 | 55.92 | 1.135 |
| 9600 | 0.9652 | 0.3112 | 0.4922 | 0.2461 | 0.1953 | 0.3293 | 0.988 | 1.4090 | 0.7097 | 56.83 | 57.01 | 1.161 |
| 11200 | 0.9650 | 0.3060 | 0.4805 | 0.2617 | 0.1758 | 0.3360 | 1.008 | 1.4035 | 0.7125 | 56.80 | 56.62 | 1.161 |
| **12800** | 0.9674 | 0.3125 | 0.4805 | 0.2695 | 0.1875 | 0.3226 | 0.968 | 1.4035 | 0.7125 | 56.95 | 56.77 | **1.164** |

Offset `+1` (the adapter-off seed row) was exactly 1.000 at every checkpoint and every
block size, so the token gate never leaked. Peak memory 11.6 GB throughout.

Net change over 9600 steps, against the 2 sd measurement noise from § Measurement:

| quantity | 3200 → 12800 | 2 sd | verdict |
|---|---|---|---|
| validation TV | −0.0071 | 0.0057 | at the noise floor |
| teacher-forced agreement | +0.0078 | 0.0113 | **within noise** |
| free-running acceptance | +0.0104 | 0.0084 | marginal |
| mean accepted prefix | +0.0314 | 0.0254 | marginal |
| TPF | +0.0162 (+1.2%) | 0.0062 | real, tiny |
| tok/s (mean) | +0.81 (+1.4%) | — | consistent with TPF |
| tok/s (median) | +2.11 (+3.9%) | — | **estimator artifact, see below** |

For scale: U3's 400 → 3200 gave **+12%** TPF. U4-A's 3200 → 12800, on 3× the
training, gave **+1.2%**.

### A wall-clock estimator that flatters the result

The median tok/s over the 15-prompt suite rose 3.9% while TPF rose 1.2% and per-cycle
cost was flat to within 0.6% (49.31 → 49.18 ms). Those cannot all be true. The **mean**
over prompts rose 1.4%, and the cost model — which knows only acceptance and cycle
cost — predicts +1.4%. The median moved further because with 15 prompts a single
prompt crossing the middle shifts it.

**The mean is the defensible estimator here and it says ~+1.4%.** Both are in the
tables; the median is not used for any claim.

---

## K=4 saturation analysis

The pre-registered rule (criteria §5): an interval is flat if Δacceptance < 0.020,
ΔTPF < 0.030 and ΔTV > −0.015; saturation requires two consecutive flat intervals.
`results/act4u4/u4_saturation.csv`:

| interval | Δ mean prefix | Δ TPF | Δ val TV | flat | saturated |
|---|---|---|---|---|---|
| 3200→4800 | +0.0433 | +0.0217 | +0.0030 | yes | — |
| 4800→6400 | −0.0549 | −0.0270 | −0.0041 | yes | **yes** |
| 6400→8000 | +0.0000 | +0.0000 | −0.0138 | yes | **yes** |
| 8000→9600 | +0.0630 | +0.0270 | +0.0055 | no | — |
| 9600→11200 | +0.0201 | −0.0055 | −0.0001 | yes | — |
| 11200→12800 | −0.0401 | +0.0000 | +0.0024 | yes | **yes** |

**K=4 met the pre-registered saturation condition at step 6400** and again at 12800.
Two of the six intervals are *negative* on both acceptance and TPF; the series is
fluctuation around a plateau, not a curve still climbing.

The rule was not built to fire easily: applied to U3's final 800-step interval scaled
to 1600 steps it gives ΔTPF ≈ +0.045, which is not flat.

**Gate U4-A2: saturation MET.**

---

## Future-offset learning

Offset convention: `+j` is supervised slot `j−1`, so `+1` is the adapter-off seed row.
**Act IV-U3 labelled these by speculative slot, so U3's "+1" is this document's "+2".**
U3's finding (later horizons improved more) is unaffected; only the labels shift.

K=4, 3200 → 12800, against the pre-registered improvement threshold of 0.022
(3× the pooled per-offset replicate sd):

| offset | 3200 | 12800 | Δ | clear learning? |
|---|---|---|---|---|
| +2 | 0.4766 | 0.4805 | +0.0039 | no |
| +3 | 0.2656 | 0.2695 | +0.0039 | no |
| +4 | 0.1719 | 0.1875 | +0.0156 | no |

**Gate U4-A5: Case C — all positions plateau.** Not Case A (a sharper shallow
predictor) and not Case B (an expanding horizon). K=4 training had nothing left to
give at any offset.

**Gate U4-A6: entry to K=8 via Gate A (saturation).** Gate B did not fire — it needed
+0.022 at both `+3` and `+4`, and got +0.0039 and +0.0156.

---

## Acceptance distribution

Free-running accepted-prefix distribution at step 12800 (`u4_acceptance_histogram.csv`):

| accepted specs | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| K=4 | 0.373 | 0.357 | 0.197 | 0.072 |

Committed tokens per cycle is `accepted + 2` exactly, for every K — including the
full-block case, where `K+1 = (K−1)+2`. So `E[committed] = E[accepted] + 2`, and the
horizon survival curve's area **is** the expected accepted prefix rather than merely
relating to it.

Over the whole of U4-A the mass moved from `accept 0` toward `accept 1`, not from
`accept 1` toward `accept 2/3`. That is the shape that predicts K=8 will not pay:
lengthening the block only helps if probability mass is reaching the *far* slots.

---

## Decoder cost model v2

U3's model over-predicted throughput at every checkpoint, +3.9% to +10.2%, mean +7.7%
— a residual that never changes sign, which is a missing additive term, not noise. v2
adds two terms, **both measured, neither fitted**:

* **`overhead_ms`** — the part of a cycle that is neither a forward nor a commit:
  cache snapshot/restore, draft-block construction, the proposal round-trip to the
  host, the greedy accept test, the entropy probe. `decode.py` times those regions
  directly. Measured at 0.78–0.92 ms/cycle.
* **prefill** — v1 predicted a steady-state rate and was compared against
  `tokens / wall_seconds`, which contains the prompt prefill (35–41 ms of a ~0.85 s
  generation). v2 predicts end-to-end and reports steady-state separately.

Over all **48** (checkpoint, K) points in both sessions:

| model | mean \|error\| | mean error | negative residuals |
|---|---|---|---|
| v1 (U3) | 8.82% | +8.82% | **0 / 48** |
| **v2** | **2.36%** | +2.34% | 2 / 48 |

v1 reproduces U3's bias exactly — 48 points, every one positive. v2 cuts the error
3.7× and breaks the sign pattern.

**Gate U4-A4: PASS** (mean |error| 2.36% < 4%, signs not all identical).

Residual honesty: v2 still leans positive (+2.3%). It is not unbiased, and it remains
an upper bound on absolute tok/s. It is reliable for ranking block sizes and for
inverting acceptance requirements, which is what it is used for.

Independently timed, `proposal + verify + commit + overhead` accounts for the cycle
loop to within 0.006 ms/cycle unattributed (asserted in `tests/test_uno_decode.py`).
The marginal price of one more draft position, regressed across K = 2, 4, 8:
**1.28–1.52 ms**, R² ≥ 0.993.

---

## Decision to enter K=8

Gate A fired: K=4 reached pre-registered saturation. Gate B did not. Recorded before
any K=8 training was launched.

---

## Official Uno curriculum comparison

Re-confirmed from the released source before designing arm B2:

| item | confirmed value | source |
|---|---|---|
| block curriculum | `2, 4, 6, 8, 12, 16` | `training/configs/uno_3epoch_curriculum.yaml` |
| tokens per stage | ~2.4576 B, six equal stages, 14.7456 B total | same |
| stage transition | changes **only** the block size | `training/trainer.py`, `CurriculumCheckpointCallback.on_step_begin` → `set_model_block_size` |
| optimizer at a transition | **not** reset | same — no reinitialisation in the callback |
| adapter at a transition | **not** reinitialised | same |
| LR schedule | `warmup_stable_decay` with `num_decay_steps: 0` → warmup then constant | `training/train.py` |

So the official path to K=8 is `4 → 6 → 8` with optimizer, adapter and learning rate
all carried across, which is exactly arm B2. Our `block_curriculum` machinery already
had these semantics; the confirmation is that they match IFM's rather than being our
own convention. What we do **not** reproduce is scale — IFM spend ~2.46 B tokens per
stage; we spend 1600 steps × 4 × 128 ≈ 0.8 M tokens, four orders of magnitude fewer.

---

## U4-B: K=8

Two arms, both continuing from `runs/u4a/step-12800` with 3200 additional steps and
everything else held constant, evaluated together with an untrained control in one
session against an in-session AR baseline of **49.45 tok/s**:

* **B1** — direct: block size 8 for 3200 steps.
* **B2** — official curriculum: block size 6 for 1600 steps, then 8 for 1600.

### Control

The untrained (zero-init) adapter at K=8 gives TPF 1.0542 and **36.85 tok/s = 0.745×
AR — a 25% slowdown**. Every K=8 number below is the adapter's doing, not the block
structure's.

### K=8 results

| checkpoint | val TV | tf agree | +2 | +3 | +4 | +5 | +6 | +7 | +8 | accept | mean prefix | TPF | tok/s | ×AR | ×best K=4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| untrained | 1.6583 | 0.0340 | 0.086 | 0.051 | 0.020 | 0.023 | 0.023 | 0.016 | 0.020 | 0.026 | 0.180 | 1.0542 | 36.85 | 0.745 | 0.624 |
| step-12800 (ref) | 1.3340 | 0.1819 | 0.469 | 0.273 | 0.168 | 0.129 | 0.094 | 0.094 | 0.047 | 0.1553 | 1.087 | 1.4487 | 52.71 | 1.066 | 0.892 |
| B1 +400 | 1.3197 | 0.1903 | 0.477 | 0.262 | 0.172 | 0.145 | 0.109 | 0.117 | 0.051 | 0.1565 | 1.096 | 1.4546 | 52.70 | 1.066 | 0.892 |
| B1 +800 | 1.3238 | 0.1864 | 0.453 | 0.277 | 0.180 | 0.117 | 0.121 | 0.102 | 0.055 | 0.1552 | 1.086 | 1.4371 | 52.59 | 1.063 | 0.890 |
| B1 +1600 | 1.3192 | 0.1897 | 0.457 | 0.285 | 0.172 | 0.117 | 0.117 | 0.109 | 0.070 | 0.1499 | 1.049 | 1.4371 | 50.79 | 1.027 | 0.860 |
| B1 +3200 | 1.3191 | 0.1869 | 0.484 | 0.285 | 0.176 | 0.113 | 0.109 | 0.078 | 0.063 | 0.1570 | 1.099 | 1.4429 | 53.78 | 1.087 | 0.910 |
| B2 K=6 +1600 | 1.3286 | 0.1881 | 0.465 | 0.266 | 0.203 | 0.109 | 0.105 | 0.102 | 0.066 | 0.1565 | 1.096 | 1.4546 | 53.03 | 1.072 | 0.898 |
| B2 K=8 +400 | 1.3207 | 0.1903 | 0.441 | 0.289 | 0.176 | 0.133 | 0.121 | 0.105 | 0.066 | 0.1481 | 1.037 | 1.4201 | 50.63 | 1.024 | 0.857 |
| **B2 K=8 +1600** | **1.3046** | **0.1992** | 0.477 | 0.258 | 0.215 | 0.145 | 0.137 | 0.098 | 0.066 | **0.1633** | **1.143** | **1.4664** | 53.09 | 1.074 | 0.899 |

---

## Long-horizon learning

Against the reference (`step-12800` at K=8) and the pre-registered threshold of 0.022:

| arm | Δ+5 | Δ+6 | Δ+7 | Δ+8 | Gate U4-B1 |
|---|---|---|---|---|---|
| B1 direct 4→8 | **−0.0156** | +0.0156 | −0.0156 | +0.0156 | **FAIL** |
| B2 curriculum 4→6→8 | +0.0156 | **+0.0430** | +0.0039 | +0.0195 | **FAIL** |

**Gate U4-B1 fails for both arms.** B1 shows no long-horizon movement at all — its
offsets end where they started. B2 misses on `+5` alone: +0.0156 against a threshold
of 0.022, and clears it comfortably at `+6`.

The curriculum arm is nonetheless clearly better than the direct arm on everything
that aggregates: val TV 1.3046 (the lowest of any checkpoint at K=8, and 6σ below the
reference's 1.3340), teacher-forced agreement 0.1992 vs B1's 0.1869, mean accepted
prefix 1.143 vs 1.099. **The intermediate K=6 stage helped**, exactly as the official
curriculum's shape would suggest.

But "better" here is a few hundredths on offsets that start near 0.1. There is signal
past +4; it is small, and 3200 steps at ~0.8 M tokens was not enough to make it large.

---

## Horizon survival

`P(accepted ≥ j)` at the best checkpoint of each arm (`u4_survival.csv`, plot 9):

| j | 1 | 2 | 3 | 4 | 5 | 6 | 7 | E[accepted] |
|---|---|---|---|---|---|---|---|---|
| K=4 (B1 +3200) | 0.639 | 0.303 | 0.115 | — | — | — | — | 1.057 |
| K=8 (ref, K=4-trained) | 0.643 | 0.295 | 0.091 | 0.046 | 0.012 | 0.000 | 0.000 | 1.087 |
| K=8 (B1 +3200) | 0.599 | 0.318 | 0.112 | 0.062 | 0.008 | 0.000 | 0.000 | 1.099 |
| **K=8 (B2 +1600)** | 0.630 | 0.328 | 0.109 | 0.059 | 0.013 | 0.004 | 0.000 | **1.143** |

The curve is nearly dead by j=5 and exactly dead by j=7. K=8's advantage over K=4 is
almost entirely at **j=2 and j=4**, not in the tail. Slots 6 and 7 accept in 0.4% and
0.0% of cycles respectively after K=8 training.

---

## Throughput economics

The marginal contribution of speculative slot `j` to committed tokens per cycle is
exactly `P(accepted ≥ j)`. It pays for itself iff

```
P(accepted >= j)  >  E[committed at block j+1] x (ms_per_slot / cycle_ms)
```

At `ms_per_slot = 1.50` measured (`u4_marginal_slot.csv`):

| checkpoint | K | slots that pay | first slot that does not |
|---|---|---|---|
| step-12800 | 4 | 1, 2 | 3 |
| step-12800 | 8 | 1, 2 | 3 |
| B1 +3200 | 4 | 1, 2, 3 | — (all pay) |
| B1 +3200 | 8 | 1, 2, 3 | 4 |
| B2 +1600 | 4 | 1, 2, 3 | — (all pay) |
| **B2 +1600** | **8** | **1, 2, 3** | **4** |

**Even after K=8 training, only three speculative slots earn their cost.** A block of
four (one seed + three speculative) is exactly the point where the last slot still
pays. Slots 4–7 cost 1.5 ms each and return 0.059, 0.013, 0.004 and 0.000 tokens.

Structural ceilings and what each block size would need (`u4_cost_model.csv`, plot 12):

| | K=2 | K=4 | K=8 |
|---|---|---|---|
| cycle cost | 45.8 ms | 49.2 ms | 55.0 ms |
| **ceiling** (every proposal accepted) | 1.339× | 2.048× | **3.310×** |
| best measured | 1.099× | **1.195×** | 1.074× |
| per-slot acceptance for 1.30× | 0.913 | 0.585 | 0.613 |
| per-slot acceptance for 1.50× | **unreachable** | 0.720 | 0.692 |
| per-slot acceptance for 2.00× | unreachable | 0.968 | 0.821 |
| **actual per-slot acceptance** | 0.624 | 0.348 | 0.163 |

K=8 has by far the highest ceiling and is by far the furthest from it. It needs
per-slot acceptance 0.50 just to match AR by 1.10×; it has 0.163.

### Gate U4-B2 and U4-B3

* **U4-B2 (committed tokens): PASS.** Best K=8 TPF 1.4664 > best K=4 TPF 1.4314. K=8
  genuinely commits more tokens per forward.
* **U4-B3, K=8 vs AR: PASS.** 53.09 tok/s vs 49.45 = **1.074×**.
* **U4-B3, K=8 vs best K=4: FAIL.** 53.78 (best K=8) vs 59.08 (best K=4) = **0.910×**.
  In-session repeat noise is 0.05 tok/s, so this is not a tie.

K=8 wins the algorithmic metric and loses the one that matters. It buys **+2.4% TPF**
for **+11.9% cycle cost**.

---

## The result nobody pre-registered

The K=8 and K=6 training made the adapter better at **K=4** than 9600 further steps of
K=4 training did. Paired, same session, same noise stream:

| adapter | K=4 accept | K=4 mean prefix | K=4 TPF | K=4 tok/s | ×AR |
|---|---|---|---|---|---|
| step-12800 (after 9600 K=4 steps, saturated) | 0.3226 | 0.968 | 1.4035 | 56.17 | 1.136 |
| **B1 +3200 K=8 steps** | **0.3525** | **1.057** | **1.4314** | **59.08** | **1.195** |
| B2 +1600 K=6 then +1600 K=8 | 0.3484 | 1.045 | 1.4314 | 58.60 | 1.185 |

Acceptance +0.0299 against a 2 sd of 0.0084 (≈7σ); TPF +0.0279 against 2 sd 0.0062
(≈9σ). By the mean estimator, tok/s rose 56.04 → 57.61, **+2.8%**, consistent with
+2.0% TPF and −0.9% cycle cost. The median says +5.2%; as in U4-A, the mean is the
number to quote.

3200 steps at a longer block did more for K=4 than 9600 steps at K=4. At K=4 the
adapter's own metrics were flat across all six U4-A intervals; a longer training block
moved them immediately. The gain sits at offset `+2` (0.4805 → 0.5039, +0.0234, just
over the 0.022 threshold) — training on a harder, longer-range objective sharpened the
*nearest* prediction.

This is a single comparison at one scale and it was not pre-registered, so it is a
hypothesis for U5, not a result U4 establishes. It is also the most actionable thing
in this document.

---

## Draftability vs future offset

Pre-registered hypothesis (criteria §8): *the draftability gap becomes more predictive
of acceptance as the future offset grows.*

At B2's final K=8 checkpoint, `r(gap, agreement)` by offset:

| offset | +2 | +3 | +4 | +5 | +6 | +7 | +8 |
|---|---|---|---|---|---|---|---|
| r(gap) | −0.103 | −0.154 | −0.011 | −0.047 | −0.134 | −0.018 | −0.051 |
| r(entropy) | −0.086 | −0.132 | −0.123 | −0.120 | −0.046 | −0.028 | −0.031 |

Mean |r(gap)| at +6…+8 is 0.068; at +2…+3 it is 0.128. **The hypothesis is refuted —
the gap is *less* predictive further out, not more.** And the calibration shows these
per-offset correlations moving by ±0.05 across three runs of an *identical* adapter,
so most of the structure above is noise.

Pooled, the gap remains the better predictor (−0.167 vs −0.071 for entropy) and the
quintile gradient is clean and monotone: 0.288, 0.221, 0.187, 0.168, 0.133 from most
to least draftable. Draftability predicts acceptance; it does not predict it
*increasingly* with horizon.

U3's finding that the least-draftable quintile improved most is not contradicted —
that was about who *gains from training*, this is about who *is accepted*.

---

## K=2 vs K=4 vs K=8

| | K=2 | K=4 | K=8 |
|---|---|---|---|
| best measured | 1.099× AR | **1.195× AR** | 1.074× AR |
| best TPF | 1.2565 | 1.4314 | **1.4664** |
| cycle cost | 45.8 ms | 49.2 ms | 55.0 ms |
| ceiling | 1.339× | 2.048× | 3.310× |
| slots that pay for themselves | 1 of 1 | **3 of 3** | 3 of 7 |
| 1.50× AR | unreachable | reachable | reachable |

**K=4 is the only block size where every draft slot pays for itself.** K=2 is
structurally exhausted; K=8 carries four slots that cost 1.5 ms each and return
almost nothing.

---

## Measurement

Three evaluations of one frozen adapter differing only in the two noise seeds
(`runs/u4-calib/rep{1,2,3}.json`), run before the criteria were frozen:

| quantity (K=4) | mean | sd | 2 sd |
|---|---|---|---|
| validation TV | 0.9751 | 0.0029 | 0.0057 |
| teacher-forced agreement | 0.3086 | 0.0057 | 0.0113 |
| free-running acceptance | 0.3170 | 0.0042 | 0.0084 |
| TPF | 1.3909 | 0.0031 | 0.0062 |
| median tok/s | 55.83 | 2.356 | 4.71 |

Two consequences worth stating plainly:

**Cross-session wall-clock is worth ±4%, and the cause is not thermal.** Between-session
sd of K=4 tok/s is 2.36 while *within*-session repeat noise is 0.15. The difference is
the draft-noise realisation: different noise gives genuinely different acceptance,
hence different speed. Every comparison in this document is inside one session with
the noise stream held fixed across checkpoints.

**U3's reported "repeat noise" of 1.44–1.99 tok/s was ~12× too large, in the
conservative direction.** U3 drew fresh global-RNG draft noise on every repeat, so its
"repeats" of a prompt decoded different token sequences; that figure was noise-draw
variance, not measurement jitter. Keyed, repeats are bit-identical decodes and the
residual jitter is 0.05–0.22 tok/s. Gate U3-4 required beating a baseline by more than
the inflated figure, so it was *harder* than intended. **U3's conclusion stands and
was understated.**

---

## Failure modes

**Gate U4-B1 failed for both K=8 arms.** Offsets +5 and +6 did not both clear +0.022
over the reference. The honest reading is not "K=8 cannot learn" — B2 moved +6 by
+0.043 and lowered K=8 validation TV by 6σ — but "3200 steps at ~0.8 M tokens is not
enough to build a horizon past four tokens." IFM spend ~2.46 B tokens per curriculum
stage; we spend four orders of magnitude less.

**B1 (direct 4→8) learned essentially nothing at long offsets.** Its +5 and +7 went
*down*. Jumping straight from a K=4-saturated adapter to K=8 wastes the budget; the
K=6 intermediate stage is doing real work.

**The wall-clock median is a fragile estimator** at 15 prompts. It inflated both the
U4-A trend (3.9% vs the mean's 1.4%) and the K=8→K=4 transfer effect (5.2% vs 2.8%).
Neither claim rests on it.

**The pre-registered draftability hypothesis was wrong**, and the measurement is
mostly noise anyway at this sample size.

**Cost model v2 still leans +2.3% positive.** Better than v1's +8.8% and no longer
single-signed, but not unbiased.

---

## Interpretation

The horizon does not scale, and the reason is economic rather than statistical.

There *is* learnable signal past four tokens: K=8 agreement at +5 and +6 is 0.14 and
0.14, several times the untrained control's 0.023, and the curriculum arm moved +6 by
+0.043 in 1600 steps. The problem is that a draft slot has to earn 1.5 ms, and by
slot 4 the survival probability is 0.059 — worth about 0.06 of a token against a
break-even of 0.088. Slot 6 accepts 0.4% of the time. Four of K=8's seven speculative
slots are pure cost.

So K=8 does exactly what the theory says: it commits more tokens per forward (TPF
1.4664 vs 1.4314, gate U4-B2 passes) and it is faster than autoregressive decoding
(1.074×). It is still 10% slower than K=4, because it pays 11.9% more per cycle for
2.4% more tokens. That gap would close if acceptance at slots 4–7 rose substantially,
and the cost model says exactly how much: per-slot 0.50 for K=8 to reach 1.10×,
against the 0.163 it has.

The frontier for this architecture and this training budget is **K=4** — the largest
block whose every slot pays for itself.

The unplanned finding points the other way, though, and is the reason not to close the
book. Training at K=8 and K=6 improved K=4 by more than 9600 further K=4 steps did.
A longer block is a better *training* objective than it is an *inference* configuration.
That decouples two things U4 assumed were the same question, and it is what U5 should
test.

---

## Verdict

```
K4 IS THE PRACTICAL FRONTIER
```

K=4 saturated (gate U4-A2, at step 6400). K=8 learns, commits more tokens per forward
(U4-B2 pass) and beats AR (U4-B3a pass, 1.074×), but is 10% slower than K=4 (U4-B3b
fail, 0.910×) and failed the pre-registered long-horizon gate (U4-B1). Only three
speculative slots pay for themselves at any block size tested.

Best configuration measured in Act IV-U4: **K=4, 59.08 tok/s, 1.195× AR**, on a frozen
backbone with lossless greedy decoding — reached by training at K=8.

| gate | result |
|---|---|
| U4-A0 integrity | **PASS** — backbone digest unchanged, seed row 1.000 everywhere |
| U4-A1 resume is a continuation | **PASS** — bitwise-identical split-run test, all 7 checkpoint fields |
| U4-A2 K=4 saturation | **MET** at step 6400 |
| U4-A3 causal chain | reported; flat at every link after step 6400 |
| U4-A4 cost model v2 | **PASS** — 2.36% mean \|error\| over 48 points, mixed signs |
| U4-A5 horizon case | **Case C** — all offsets plateau |
| U4-A6 entry to K=8 | **Gate A** (saturation) |
| U4-B0 integrity | **PASS** |
| U4-B1 long-horizon signal | **FAIL** — both arms; B2 missed on +5 only |
| U4-B2 committed tokens | **PASS** — K=8 TPF 1.4664 > K=4's 1.4314 |
| U4-B3a K=8 vs AR | **PASS** — 1.074× |
| U4-B3b K=8 vs best K=4 | **FAIL** — 0.910× |
| U4-B4 quality | **PASS** — bit-exact lossless at K=2/4/8, cacheless reference regime |

**11 of 13 gates pass. The two failures are the result.**

---

## Next

In priority order, and none of it started:

1. **Test the transfer directly.** Train K=8 or K=6 and *decode* at K=4. U4 found this
   by accident on one comparison; it deserves its own arm with a control that rules
   out "any 3200 extra steps would have done it" — the obvious control being 3200 more
   K=4 steps from step-12800, which U4-A's plateau predicts will do nothing.
2. **The official curriculum from the start**, `2 → 4 → 6 → 8`, rather than bolted on
   after 12800 steps of fixed K=4. B2 beat B1 on every aggregate; the curriculum may
   matter more than the endpoint.
3. **K=8 at a real token budget**, if the point is to test IFM's claim rather than the
   frontier. Our whole U4-B is 0.03% of one of their curriculum stages.
4. **Dynamic-K routing (U5)** is now properly informed: best K is 4, K=4 training
   saturates, and the per-slot economics are measured. The survival curve says a
   router would need to predict, per cycle, whether slot 3 will be accepted — that is
   where the marginal token is.

---

## Addendum — 2026-09-14: what Act IV-U5 changes here

Appended after Act IV-U5. The text above is left exactly as written.

Act IV-U5 measured something U4 could not: the spread between separate training launches
of an identical run. Launch identity is proven from artifacts
(`results/act4u5/u5_replicate_provenance.json`).

* Three launches of K=4 training from `runs/u4a/step-12800` finished at K=4 mean accepted
  prefix 1.0283, 1.0288 and 0.9759, a spread of 0.053.
* Two identical K=8 launches finished at 0.9880 and 1.0826, a spread of 0.095.

Evaluating a fixed adapter is bit-deterministic across sessions, so this is training
variance. It affects every comparison here between **separate launches**, and none of
the comparisons along **one trajectory**.

* **"The result nobody pre-registered"** (B1 +3200 K=8 steps vs step-12800: acceptance
  +0.030, prefix +0.090, "~7σ") **is not supported.** The σ was evaluation noise only.
  U5 ran the matched control: a K=4 continuation from the same start moved prefix +0.060
  by itself, no horizon arm beat it, and the verdict is `U4 OBSERVATION WAS NOISE`
  ([`act4u5_results.md`](act4u5_results.md)).
* **B2 vs B1** (curriculum vs direct, K=8 prefix 1.143 vs 1.099) were two single
  launches. Their 0.044 difference is inside the K=8 launch spread, so "the intermediate
  K=6 stage helped" is not established.
* **U4-A's saturation analysis stands.** It compares checkpoints along one continuous
  run, so launch variance does not enter.
* **K=8 vs K=4 on wall clock (0.910×) stands.** It compares decode widths on fixed adapters.
* **U4-B4, "bit-exact lossless at K=2/4/8, cacheless reference regime", passed on six
  prose and factual prompts.** Act IV-U5 repeated the check on the full 15-prompt suite
  and found two `structured` prompts diverging at bf16 ties (one exact tie, one at one
  ULP). The guarantee is exact up to bf16 ties, not unconditionally. The verdict is
  unaffected.
