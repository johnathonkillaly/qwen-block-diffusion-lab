# Act IV-U6 success criteria — pre-registered

**Written before any U6 measurement**: before the diagnostic, the cost curve, the
calibration, the pilot and the decisive run. Design: [`act4u6_design.md`](act4u6_design.md).

This document is frozen once committed. Later amendments go in §9 with a date and a reason.
§6 has one reserved slot, filled by the calibration run *before* the pilot and the
decisive run; the formula that fills it is fixed here.

---

## 0. Hypothesis and primary metric

> Decoupling draft width from verify width (`K_draft ≠ K_verify`) raises end-to-end
> committed tokens per second above the incumbent coupled K=4 decoder, on a frozen
> drafter, with greedy output unchanged.

**Primary metric:** end-to-end committed tokens per second per (prompt, repeat) unit,
paired against arm B (incumbent K=4) on the same unit in the same session.

**Secondary metrics**, reported and never decisive:
* mean accepted prefix and committed tokens per cycle;
* target (verifier) forwards and drafter forwards per committed token;
* TPF;
* cycle latency, per-stage verify latency and draft latency;
* rejected draft tokens (verified and rejected) and unused draft tokens (never verified);
* target verification tokens processed per committed token;
* per-row latency dispersion.

**TPF, acceptance or accepted prefix can never establish a win on their own.**

## 1. Arms and semantics

As defined in the design, §4:

| arm | definition |
|---|---|
| A | native AR |
| B | incumbent `decode.uno_greedy_generate`, D = V = 4 |
| B_dec | the new decoder at D = V = 4, the harness-equivalence control |
| C / D / E / F | staged `S(8,4)` / `S(8,2)` / `S(6,4)` / `S(6,3)` |
| T arms | truncate `T(8,4)`, `T(8,2)`, `T(6,4)`, `T(6,3)`, run **only if U6-1 keeps Variant 1 alive** |

Frozen drafter: `runs/u4b1/step-16000`, SHA-256
`8200c339d3d47363a3920fc4aca58f3535fc8bf75be431e35204f374700a4f42`.

Harness: the Act IV-S suite (27 prompts, 128 tokens), arms interleaved, draft-noise
stream 20260905, snapshot transactions. **No prompt, token length or noise stream is
changed after any U6 result is seen.**

## 2. Gate U6-0 — integrity (required)

* The adapter SHA-256 matches the manifest in every session.
* The backbone digest is unchanged before and after every session.
* `tests/test_uno_decoupled.py` passes, including:
  * exact reproduction of the incumbent at D = V;
  * losslessness for every arm under diffusion, oracle and adversarial drafters;
  * rejected tokens never committed;
  * stage continuation only after a fully accepted stage with a matching lookahead.
* **Losslessness modulo bf16 ties.** Every divergence from native AR, in every arm and
  every session, is audited against the target's full-context logits. Zero divergences
  may lie beyond one bf16 ULP. One unexplained divergence voids that arm.

## 3. Gate U6-1 — does a wider draft change the first four slots? (Variant 1 kill)

Diagnostic over 256 contexts (27 suite prompts plus 229 held-out WikiText-103 windows of
64 tokens). For each D ∈ {6, 8, 16}, under decoder-realistic noise, compute
`Δ_D = prefix₄(D) − prefix₄(4)`. Here `prefix₄` is the number of leading slots among
p₂…p₄ that match the target's greedy continuation.

Also compute the **re-draw control**: `Δ_redraw = prefix₄(4, independent key) − prefix₄(4)`.

**Variant 1 stays alive only if some D has a paired 95% bootstrap CI on `Δ_D` with lower
bound > 0, and a mean `Δ_D` greater than `|mean Δ_redraw|`.** Otherwise it is killed
immediately and no truncate arm is benchmarked.

Reported regardless: per-slot argmax identity with D=4 under **aligned** noise (one
width-16 draw, prefix-shared), maximum logit differences, and top-1 probability per slot.
Given causal attention, the design predicts identity up to bf16 width effects.

## 4. Gate U6-2 — the hardware cost curve (measured)

Measured on the M4 Max through the decoder's own transaction path:
* verify-forward latency at widths 2–9, 13 and 17;
* draft-forward latency at widths 2, 4, 6, 8, 12 and 16;
* commit time;
* snapshot/restore overhead.

Fitted: `ms = F + m · width`, separately for verify and draft, with R².

**Staged verification is killed before the pilot only if the mechanism cannot win even
with a perfect drafter.** That is, with every stage fully accepted, measured costs give
every staged arm fewer committed tokens per millisecond than an equally perfect
incumbent. Otherwise U6-2 is descriptive. The design (§5) records that this bound is
expected to pass and the realistic prediction to fail.

## 5. Gate U6-3 — pilot kill

The pilot runs arms A, B, B_dec, the staged arms and, if alive, the truncate arms, with 1
repeat over 27 units.

**Kill rule: if every decoupled arm (staged, plus truncate if alive) has a paired 95%
bootstrap CI on the percentage tok/s difference vs B whose upper bound is below −floor
(§6), the decisive run is not performed.** The verdict is then scored from the pilot with
the rules in §8, and labelled as a pilot verdict.

## 6. Calibration and the practical floor

**Calibration run:** arms A, B and B₂ (a second, identical incumbent), interleaved on the
full suite, 3 repeats, 81 units, run before the pilot.

For each unit `i`, `d_i = tok/s(B₂)_i − tok/s(B)_i`. With `[lo, hi]` the 95% paired
bootstrap CI of `mean(d)` (the harness bootstrap, 10,000 resamples, fixed seed):

```
floor (%) = max( 1.0 , ⌈ 100 · max(|lo|, |hi|) / mean tok/s(B) ⌉ rounded up to 0.5 )
```

That is the largest mean difference an identical-code comparison produces at 95%, and
never less than 1%.

> **Reserved slot — filled 2026-09-14 from the calibration run, before the pilot**
> (`results/act4u6/calibrate.json`, `results/act4u6/floor.json`):
>
> * 81 units; incumbent mean 58.07 tok/s;
> * mean(B₂ − B) = +0.092 tok/s, 95% CI [-0.244, +0.442];
> * raw 0.76%, so **floor = 1.0%** (the 1.0% minimum binds).
>
> Descriptive only: the two identical arms' *medians* differ by 2.3% (56.62 vs 57.95
> tok/s) while their paired mean differs by 0.16%. That is why every U6 decision uses the
> paired mean.

## 7. Gates U6-4 and U6-5 — decisive

The decisive run uses the same arms as the pilot (truncate arms only if alive), 3
repeats, 81 units.

**U6-4, decoupling wins.** Arm X passes if, against B, paired over the 81 units:
* the 95% bootstrap CI of the mean percentage tok/s difference has a lower bound > 0, **and**
* the point estimate is ≥ floor.

The loop-free subset (`ar_looped_fraction < 0.05`) is reported beside it. If its point
estimate has the opposite sign, the pass is recorded as **not robust**, and a not-robust
pass cannot produce a `… WINS` verdict.

**U6-5, no degradation (required for any win).** Zero unexplained divergences (U6-0).
B_dec, on the same units, must not be slower than B by more than floor, or the new code
path's overhead confounds every decoupled arm, and the report must say so.

Descriptive bands for a passing arm: > 0% real, ≥ 2% modest but interesting, ≥ 5%
strong enough to pursue, ≥ 10% major.

## 8. Verdict

Exactly one primary verdict, chosen after the gates are scored, by the first rule that
applies:

| # | verdict | condition |
|---|---|---|
| 1 | `STAGED VERIFICATION WINS` | a staged arm passes U6-4 (robust) and U6-5 |
| 2 | `DECOUPLED WIDTH WINS` | a truncate arm passes U6-4 (robust) and U6-5 |
| 3 | `VERIFY COST DOMINATES` | no arm wins; some staged arm commits significantly more tokens per cycle than B (paired CI lower bound > 0); and for that arm the added verify time per cycle over B exceeds the added draft time per cycle |
| 4 | `WIDER DRAFT HELPS, BUT NOT WALL CLOCK` | no arm wins; Variant 1 survived U6-1, or some decoupled arm has a significantly larger mean accepted prefix per cycle than B |
| 5 | `DRAFT WIDTH DOES NOT AFFECT USEFUL PREFIX` | no arm wins; U6-1 killed Variant 1; no staged arm commits significantly more tokens per cycle than B |
| 6 | `NO BENEFIT FROM DECOUPLING` | none of the above |

**Recorded predictions**, from design §5, so they can fail:
* U6-1 kills Variant 1;
* every staged arm loses to B, by −1.3% (S(6,4)) to −8.9% (S(8,2));
* the verdict is `VERIFY COST DOMINATES` or `DRAFT WIDTH DOES NOT AFFECT USEFUL PREFIX`.

**What does not happen, whatever the result:**
* No adapter is trained in U6-A.
* No threshold is chosen after seeing a result.
* A TPF or acceptance gain is never reported as a speedup.

## 9. Amendments

*(none)*
