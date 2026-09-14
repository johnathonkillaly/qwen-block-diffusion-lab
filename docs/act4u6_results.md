# Act IV-U6 — Decouple Draft Width from Verify Width: results

Pre-registered criteria: [`act4u6_preregistered_criteria.md`](act4u6_preregistered_criteria.md)
(frozen; the §6 floor slot was filled from the calibration run before the pilot; no
amendments). Design: [`act4u6_design.md`](act4u6_design.md).

**Ordering, provable from git:**

| commit | contents | timing |
|---|---|---|
| `1640c42` | design, criteria, decoder and tests | before any U6 measurement |
| `df71b03` | floor, U6-1 and U6-2 results, scoring script | before the pilot |
| `4567849` | pilot | before the decisive run |

Machine-readable: `results/act4u6/`: `diagnostic_wide_draft.json`, `cost_curve.json`,
`calibrate.json`, `floor.json`, `pilot.json`, `pilot_gate.json`, `decisive.json` and
`u6_scores.json`. Plots: `plots/act4u6/`.

Frozen drafter `runs/u4b1/step-16000` (SHA-256 `8200c339…`), frozen Qwen3.5-4B-Base bf16,
MLX 0.32.1, Apple M4 Max. Greedy decoding. Nothing was trained.

---

## Verdict

```
VERIFY COST DOMINATES
```

Verifying a wide draft in stages commits more tokens per cycle than the K=4 incumbent, and
is **slower in every configuration tested**, because each extra verify forward carries a
~20 ms fixed cost on this hardware. A wider draft canvas does not change the first four
draft slots at all. The coupled K=4 decoder remains the frontier.

| gate | result | evidence |
|---|---|---|
| **U6-0** integrity | **PASS** | Adapter SHA-256 matches the manifest in all five sessions. Backbone digest unchanged. 40 decoder tests pass. Every divergence from native AR is within one bf16 ULP: calibration 72, pilot 70, decisive 210 |
| **U6-1** does a wider draft change slots 1–4? | **VARIANT 1 KILLED** | Change in prefix₄ vs D=4: D=6 −0.012 [−0.063, +0.035], D=8 +0.023 [−0.027, +0.070], D=16 +0.004 [−0.039, +0.051]. Noise re-draw control: 0.000 [−0.047, +0.047] |
| **U6-2** cost curve | **staged continues** | Verify forward costs 20.1 ms + 0.80 ms/token (R² 0.976). With a perfect drafter, S(8,4) would still beat the incumbent |
| **U6-3** pilot kill | **did not fire** | S(8,4) −1.89% [−4.47, +0.56] and S(6,4) −1.92% [−4.15, +0.00] were not clearly below −1.0% |
| **U6-4** decoupling wins | **FAIL** | Every staged arm is slower than K=4, each with a 95% CI entirely below zero |
| **U6-5** no degradation | **PASS** | Zero unexplained divergences. B_dec −0.08% [−0.25, +0.07], inside the 1.0% floor |

Every preregistered prediction held:

| prediction (design §5) | measured |
|---|---|
| U6-1 kills Variant 1 | killed |
| S(6,4) −1.3% | −1.58% |
| S(8,4) −1.9% | −1.61% |
| S(6,3) −2.9% | −3.03% |
| S(8,2) −8.9% | −9.01% |
| verdict `VERIFY COST DOMINATES` or `DRAFT WIDTH DOES NOT AFFECT USEFUL PREFIX` | `VERIFY COST DOMINATES` |

---

## 1. Question

> Can the diffusion drafter propose a wider block while the target verifies only a
> narrower window per forward, so that `K_draft ≠ K_verify` raises end-to-end committed
> tokens per second above the coupled K=4 decoder?

The arms use D for draft width and V for verify width (design §4):

* **B** is the incumbent, D = V = 4.
* **B_dec** is the same configuration run through the new decoder, as a harness control.
* **S(D, V)** is staged: draft once, verify V at a time on the committed cache, and
  continue only after a fully accepted stage whose lookahead matches the draft.
* **T(D, V)** is truncate. It was run only if gate U6-1 allowed.

## 2. What ran

| session | contents | wall | native AR |
|---|---|---|---|
| U6-1 diagnostic | 256 contexts × widths 4/6/8/16 × decoder, aligned and re-draw noise | 0.51 s/context | — |
| U6-2 cost curve | 27 contexts × 5 repeats × 10 verify widths and 6 draft widths, shuffled | 79 s | — |
| calibration | A, B, B₂ × 27 prompts × 3 repeats | 605 s | 46.85 tok/s |
| pilot | A, B, B_dec, 4 staged arms × 27 prompts × 1 repeat | 455 s | 48.04 tok/s |
| decisive | A, B, B_dec, 4 staged arms × 27 prompts × 3 repeats | 1,312 s | 48.56 tok/s |

All runs used the Act IV-S suite and harness, unchanged. Arms were interleaved inside each
prompt inside each repeat, with 128 generated tokens, draft-noise stream 20260905 and
snapshot transactions. Every decision is paired over (prompt, repeat) units.

## 3. U6-1 — a wider canvas does not reach the first slots

prefix₄ is the number of leading slots among p₂…p₄ that match the target's greedy
continuation.

| condition | D=4 | D=6 | D=8 | D=16 |
|---|---|---|---|---|
| decoder noise: mean prefix₄ | 0.859 | 0.848 | 0.883 | 0.863 |
| aligned noise: mean prefix₄ | 0.879 | 0.875 | 0.875 | 0.879 |
| aligned noise: slots 1–4 identical to D=4 | — | 0.99 / 0.99 / 0.98 / 0.97 | 0.99 / 0.99 / 0.98 / 0.97 | 1.00 / 0.98 / 0.99 / 0.98 |
| aligned noise: max \|Δ logit\| vs D=4 | — | 0.14 / 0.15 / 0.16 / 0.15 | 0.14 / 0.15 / 0.16 / 0.15 | 0.13 / 0.15 / 0.16 / 0.16 |

The re-draw control (D=4 with an independent noise key) has mean prefix₄ 0.859. Its slot
identity with D=4 is 1.00 / 0.88 / 0.82 / 0.80, which is the same as D=8 under decoder
noise (0.99 / 0.88 / 0.82 / 0.74).

The draft pass is causal, and the data show it:
* **With identical noise at every width, slots 1–4 match D=4 97–99% of the time.** The
  logit differences are about 0.15, roughly one bf16 ULP at these magnitudes. Even slot 1,
  which no noise token can affect, moves by 0.13, which is a pure width effect.
* **With the decoder's own noise, a wider draft differs from D=4 exactly as much as simply
  re-drawing D=4's noise does.**

No width changes prefix₄ beyond the re-draw control, so the truncate variant has no
mechanism, and it was killed before any wall-clock work.

## 4. U6-2 — the M4 Max cost curve

Medians, measured through the decoder's own snapshot-transaction path, with each context
prefilled to its prompt plus 64 generated tokens.

**Verify forward:**

| width (tokens) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 13 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|
| ms | 21.57 | 22.53 | 23.56 | 24.55 | 24.53 | 25.66 | 26.17 | 27.69 | 33.56 | **62.29** |

**Draft forward:**

| width (tokens) | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| ms | 24.57 | 25.38 | 25.85 | 26.72 | 29.34 | **64.11** |

**Fit over verify widths 2–9: 20.1 ms fixed + 0.80 ms per token (R² 0.976).** At width 5,
the fixed part is 82% of the forward. There is also a cliff past width 13 in both passes:
width 17 costs more than twice width 13.

| comparison | ms |
|---|---|
| verify(3) twice vs verify(5) once | **45.1 vs 24.5** |
| verify(5) + verify(4), S(8,4) both stages, vs verify(9) | 48.1 vs 27.7 |
| verify(3) three times, S(8,2), vs verify(9) | 67.6 vs 27.7 |
| draft(8) − draft(4) | +1.34 |
| draft(6) − draft(4) | +0.48 |

**Perfect-drafter bound** (gate U6-2): committed tokens per ms if every stage were fully
accepted, using the measured forward, commit and overhead costs.

| arm | stage widths | tokens | ms | tokens / ms |
|---|---|---|---|---|
| incumbent (4, 4) | 5 | 5 | 51.1 | 0.0978 |
| S(8, 4) | 5, 4 | 9 | 77.2 | **0.1166** |
| S(6, 4) | 5, 2 | 7 | 74.3 | 0.0942 |
| S(6, 3) | 4, 3 | 7 | 74.3 | 0.0942 |
| S(8, 2) | 3, 3, 3 | 9 | 97.9 | 0.0920 |

Only S(8,4) could beat the incumbent, even with perfect drafting, so staged verification
was not killed. The other three configurations lose to K=4 before acceptance is even
considered.

## 5. Calibration and the floor

Two identical incumbent arms, 81 units: mean(B₂ − B) = +0.092 tok/s, 95% CI
[−0.244, +0.442], a raw 0.76% of the incumbent's mean. **Floor = 1.0%** (the minimum
binds).

The two identical arms' *medians* differ by 2.3% (56.62 vs 57.95 tok/s). A median-based
comparison would have been able to "find" a 2% effect in identical code.

## 6. Pilot (gate U6-3)

One repeat, 27 units, percent change vs B:

| arm | vs B [95% CI] | clearly below −1.0%? |
|---|---|---|
| B_dec | −0.33% [−0.69, −0.03] | (control) |
| S(8, 4) | −1.89% [−4.47, +0.56] | no |
| S(6, 4) | −1.92% [−4.15, +0.00] | no |
| S(6, 3) | −3.17% [−5.38, −1.39] | yes |
| S(8, 2) | −9.33% [−12.07, −6.75] | yes |

Two arms were not clearly below the floor, so the kill rule did not fire and the decisive
run was performed.

## 7. Decisive run (gates U6-4, U6-5)

Three repeats, 81 units. Paired mean tok/s against the incumbent (60.97 tok/s on these
units). The loop-free subset has 48 units.

| arm | vs B [95% CI] | loop-free vs B [95% CI] | median tok/s | × AR | TPF | U6-4 |
|---|---|---|---|---|---|---|
| **B, incumbent K=4** | — | — | 61.01 | **1.256** | 1.506 | — |
| B_dec (4, 4) | −0.08% [−0.25, +0.07] | −0.11% [−0.31, +0.08] | 60.71 | 1.250 | 1.506 | control |
| S(6, 4) | −1.58% [−2.89, −0.36] | −2.27% [−4.38, −0.48] | 57.75 | 1.189 | 1.454 | FAIL |
| S(8, 4) | −1.61% [−3.10, −0.17] | −3.39% [−5.31, −1.83] | 58.33 | 1.201 | 1.488 | FAIL |
| S(6, 3) | −3.03% [−4.28, −1.91] | −2.90% [−4.97, −1.19] | 56.33 | 1.160 | 1.407 | FAIL |
| S(8, 2) | −9.01% [−10.58, −7.51] | −9.91% [−12.2, −7.86] | 53.98 | 1.112 | 1.306 | FAIL |

**Every staged arm is significantly slower than K=4**, on the full suite and on loop-free
text. The incumbent reproduces Act IV-S (1.245×, 1.248×, 1.238× there; 1.256× here), and the
new decoder matches it to within 0.08%.

**Secondary metrics.** In the tables below, B's accepted-per-cycle mean is a mean of
per-unit means, so it is not directly comparable with Act IV-S's pooled 1.069.

Per cycle, with paired deltas vs B:

| arm | committed / cycle | accepted / cycle | added verify ms | added draft ms |
|---|---|---|---|---|
| B | 3.087 | 1.124 | — | — |
| S(6, 4) | 3.198 (+0.112 [+0.047, +0.180]) | 1.151 | +1.83 | +0.37 |
| S(8, 4) | 3.254 (+0.167 [+0.071, +0.282]) | 1.205 | +1.84 | +1.16 |
| S(6, 3) | 3.198 (+0.112 [+0.047, +0.180]) | 1.065 | +2.58 | +0.35 |
| S(8, 2) | 3.254 (+0.167 [+0.071, +0.282]) | 0.902 | +5.97 | +1.15 |

Forwards, stages and latency:

| arm | verify forwards / token | draft forwards / token | verify tokens / committed token | 2nd-stage rate | cycle ms (sd) |
|---|---|---|---|---|---|
| B | 0.330 | 0.330 | 1.648 | — | 49.8 (0.8) |
| S(6, 4) | 0.344 | 0.323 | 1.657 | 6.7% | 51.7 (5.5) |
| S(8, 4) | 0.338 | 0.319 | 1.672 | 6.0% | 52.4 (5.6) |
| S(6, 3) | 0.368 | 0.323 | 1.428 | 14.2% | 52.3 (7.8) |
| S(8, 2) | 0.426 | 0.319 | 1.277 | 30.6% (+2.7% third) | 56.3 (11.8) |

Mean unused draft tokens per cycle (never verified): S(8,4) 3.76, S(8,2) 5.00, S(6,4) 1.87,
S(6,3) 2.58.

Gate U6-5 holds: 210 divergences, all within one bf16 ULP. The rule-3 conditions hold for
all four staged arms: significantly more committed tokens per cycle, and more added verify
time than added draft time.

## 8. Why staging cannot win here

**At fixed D, staging never changes what is committed.** S(8,4) and S(8,2) produced
identical per-cycle commit sequences on 81 of 81 units, and so did S(6,4) and S(6,3). The
algebra agrees: a staged cycle commits the draft prefix up to the first mismatch plus the
target's own token there, capped at D + 1, and so does a coupled K=D cycle.

As a cross-session check, S(8,4)'s per-cycle commits match Act IV-S's coupled K=8 exactly on
33 of 81 units. On the rest the cycle schedule separates mid-generation (first at cycle 24
of one dialogue prompt), with no output divergence from AR in either run. That fits bf16
near-ties in a drafter reading a cache built by different forward widths. It was not
investigated further.

**So staging is only a forward schedule for committing a K=D decoder's tokens.** It trades
a narrower first verify, saving about 0.8 ms per token of width, for extra verify forwards
that cost about 20 ms each. The arithmetic follows:

* **S(8,4):** +0.167 tokens per cycle (+5.4%) for +1.16 ms of draft and +1.84 ms of verify
  (+6.0% of the cycle), giving −1.6%. A second stage happens in only 6% of cycles, because
  it needs four accepted slots and a matching fifth.
* **S(8,2):** the same +0.167 tokens for +5.97 ms of verify, because 31% of cycles pay a
  second forward. Result: −9.0%.
* **D=6 arms:** a smaller token gain (+0.112) for less draft cost, but the same shape.

**Against coupled K=8, staging recovers most of the wide verify's cost.** Act IV-S measured
K=8 at 0.906× K=4 on this suite, in a different session. S(8,4) commits K=8's tokens at
0.984× K=4. That is still below K=4, because the extra 0.17 tokens per cycle do not pay for
drafting 8 positions plus the occasional second forward.

**Latency variance gets worse.** Cycle-time sd rises from 0.8 ms (K=4) to 5.6 ms (S(8,4))
and 11.8 ms (S(8,2)), because some cycles pay one forward and others two or three.

## 9. What limited performance

1. **The fixed cost of a verify forward.** 20.1 ms of the 24.5 ms width-5 forward does not
   shrink with width. Splitting verification multiplies the fixed part.
2. **Survival.** A second stage needs a fully accepted first stage and a matching
   lookahead: 6% of cycles at V=4, 14% at V=3, 31% at V=2. Where continuation is common, V
   is small, so stage 1 commits little and the forwards pile up.
3. **Causal drafting.** A wider canvas cannot inform earlier slots, so nothing is gained
   at the positions that are actually verified.
4. **Draft width is cheap until it is not.** +0.48 ms for 6 over 4, +1.34 ms for 8 over 4,
   and a cliff past 12.

## 10. What U6 cannot conclude

* **Other hardware or runtimes.** The ~20 ms fixed forward cost is this stack's (MLX,
  Python harness, M4 Max). A runtime with a much smaller fixed cost per forward shifts every
  number in §4, and staging could behave differently there.
* **Other drafters.** A drafter with bidirectional attention inside the block could make
  the truncate variant meaningful. This architecture's draft pass is causal.
* **Configurations not tested.** Only the four preregistered configurations were run,
  greedy, one adapter, short prompts.

## 11. U6-B (training) — not proposed

The limit is per-forward cost, not prediction quality. With a perfect drafter, three of the
four staged configurations still lose to K=4 (§4). The fourth, S(8,4), needs stage-2
continuation far above the 6% measured, and even then adds cost that no training changes.
Nothing here suggests a training objective that would move the result, so U6-B is not
proposed.

## Next

**Close the decoupling line.** The one measured lever is the fixed cost of a forward: about
82% of a width-5 verify is independent of width. Before any further scheduling idea for
this decoder, profile how much of those 20 ms is Python and Metal dispatch rather than
model compute. That number bounds what any draft/verify schedule can gain on this stack.

---

## Reproduction

```bash
cd /Users/johnathonkillaly/code/diffusion-u5
export HF_HOME=/Volumes/SHUTTLE PYTHONPATH=$PWD/src
PY=/Users/johnathonkillaly/code/diffusion_project/.venv-unsloth/bin/python
$PY scripts/u6_wide_draft_diagnostic.py      # U6-1 (~2 min)
$PY scripts/u6_cost_curve.py                 # U6-2 (~2 min)
$PY scripts/u6_bench.py calibrate            # floor (~10 min)
$PY scripts/u6_bench.py pilot                # U6-3 (~8 min)
$PY scripts/u6_bench.py decisive             # U6-4/U6-5 (~22 min)
$PY scripts/u6_report.py final --floor 1.0   # scoring, verdict, plots (no model, seconds)
```

Tests: `tests/test_uno_decoupled.py` (40) and `tests/test_uno_u6_report.py` (16). The
fast suite runs 492 tests, all passing.

---

## Addendum — 2026-09-14: project closed

The research project closed after this result. The fixed-forward-cost remark in **Next** is
recorded as an observation only; no follow-up experiment is planned. See
[`PROJECT_SUMMARY_THROUGH_U6.md`](PROJECT_SUMMARY_THROUGH_U6.md).
