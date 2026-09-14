# Act IV-U5 — Train Long, Decode Short: results

Pre-registered criteria: [`act4u5_preregistered_criteria.md`](act4u5_preregistered_criteria.md)
(frozen; dated amendments A1–A4 in §11, all written before any U5 endpoint was
evaluated). Design: [`act4u5_design.md`](act4u5_design.md).

Runs completed 2026-09-13; scored and written 2026-09-14.

Machine-readable results are in `results/act4u5/`:

* `u5_summary.json`, `u5_arms.csv`, `u5_transfer.csv`, and nine plots from the frozen
  report;
* `u5_matched_step_pairing.json`;
* `u5_losslessness.json`;
* `u5_replicate_check.json` and `u5_replicate_provenance.json`;
* `u5_supplementary_robustness.{json,md}`.

Raw evaluations: `runs/u5/eval_curve.json`, `eval_endpoints.json` and
`eval_replicates.json`. Training outputs: `runs/u5/*/result.json`. `runs/` is not
committed; the committed copies are:

* `results/act4u5/raw/`: the three evaluations;
* `results/act4u5/u5_training_summary.json`: the per-arm training record;
* `results/act4u5/provenance/`: launch provenance, plus the A1 log excerpt.

---

## Verdict

```
U4 OBSERVATION WAS NOISE
```

`NO CROSS-HORIZON TRANSFER` also holds. §9 gives no rule for choosing between the two;
this one is chosen because the evidence goes further than "no transfer". It accounts for
the U4 observation itself (§5 below): a matched K=4 control gained two thirds of U4's
effect on its own, and identical training launches span U4's effect size.

| gate | result | evidence |
|---|---|---|
| **U5-0** integrity | **PASS** | Backbone digest unchanged through all 7 trainings and all 34 evaluated checkpoints; seed-row agreement 1.000 everywhere; identical start; 4 distinct final adapters; 41 required tests pass; resource guard recorded as passed in every artifact |
| **U5-1** transfer exists | **FAIL** | No arm beats matched `A_k4` at any of 5 checkpoints (15 comparisons). Endpoint prefix change: B_k6 +0.009 [−0.108, +0.101], C_k8 −0.040 [−0.155, +0.071], D_curr +0.000 [−0.145, +0.150] |
| **U5-2** horizon mechanism | not applicable | No arm passed U5-1 |
| **U5-3** TPF | **FAIL** | No TPF gain with an interval above zero at any matched checkpoint |
| **U5-4** tok/s | **FAIL** | No tok/s gain with an interval above zero. The only significant difference anywhere is D_curr@13200 being **slower** (−1.26 [−2.76, −0.12]) |
| **U5-5** no degradation | **FAIL by the letter** | Clause 1: 13/15 prompts bit-identical per arm, both divergences within one bf16 ULP. Clause 2: pass. Clause 3: C_k8 falls 0.040 below the control at the endpoint, beyond the 0.0254 floor, and D_curr falls 0.044 below at step 14400 |

The criteria predicted U5-1 would pass and U5-2 would be the interesting gate. U5-1 failed.

**Power.** With 15 prompts, paired intervals on accepted prefix are about ±0.10 wide. U5
cannot exclude a true horizon effect smaller than that. U4's effect (+0.09) sits inside
both that width and the measured launch-to-launch spread.

---

## 1. Question

> Under an equal training budget, does training the adapter against longer future
> blocks produce a better **K=4** decoder than training directly at K=4?

The experiment exists because of one unmatched, unregistered comparison in Act IV-U4:
3200 steps at K=8 after `runs/u4a/step-12800` raised K=4 accepted prefix by +0.090. Every
arm decodes at `K_decode = 4`. K=6/K=8 decoding is diagnostic only.

## 2. What ran

| arm | training horizon | wall | s/step |
|---|---|---|---|
| `A_k4` | K=4 (control) | 1.264 h | 1.422 |
| `B_k6` | K=6 | 1.267 h | 1.426 |
| `C_k8` | K=8 | 1.264 h | 1.423 |
| `D_curr` | K=4 → 6 → 8, ~1067 steps each | 1.258 h | 1.415 |

All arms resume from `runs/u4a/step-12800`, with identical adapter and AdamW state, and
train 3200 matched steps with right-aligned corruption (`noise_align_width = 8`).
Compute is matched to 0.7%. Checkpoints were saved at 13200, 13600, 14400, 15200 and
16000.

The first launch was killed at step 13090 of `A_k4` by an unrelated restart of the
controlling session. It wrote no checkpoint and contributes no data; its log is kept at
`runs/u5-launch.attempt1-killed-at-13090.log`. Comparing that log with the second launch
is how amendment A1 was discovered.

Evaluation sessions, all with 15 prompts, 48 tokens, 256 held-out rows and draft-noise
stream 20260905:

| session | contents | control in its transfer table | AR baseline |
|---|---|---|---|
| curve (`eval_curve.json`) | all 20 checkpoints plus the start, K=4 | `A_k4@12800` (the start) | 49.87 tok/s |
| endpoints (`eval_endpoints.json`) | step-16000 arms, start, untrained; K=4 plus K=6/8 diagnostics | `A_k4@16000` | 49.52 tok/s |
| replicates (`eval_replicates.json`, A2/A4) | 3 × A_k4, B_k6, C_k8, C_k8_r2, D_curr at step 16000; K=4 | `A_k4@16000` | 49.46 tok/s |

**Evaluation is deterministic across sessions.** The same checkpoint gives bit-identical
accepted prefix, TPF and teacher-forced agreement in the endpoint and replicate
sessions; only tok/s moves (≤0.5). So every spread reported below between separately
trained adapters is training variance, not evaluation noise.

## 3. U5-0 — integrity: PASS

* **Backbone:** digest `7cd0f5a9…` before and after every training run, and in every
  evaluation session, with `backbone_unchanged` true for all 34 evaluated checkpoints.
* **Seed row:** agreement exactly 1.000 at every arm and every `K_decode`.
* **Shared start:** identical for all arms (`u5_start_check.at_primary_launch.json`,
  written at launch).
* **Divergence:** the four final adapters have four distinct tensor digests.
* **Tests:** `tests/test_uno_rng.py`, `test_uno_resume.py` and
  `test_uno_horizon_pairing.py` pass (41).
* **Resource guard:** recorded `heavy_execution_allowed: true` with no override in every
  training and evaluation artifact.

## 4. U5-1, U5-3, U5-4 — FAIL at every checkpoint

### Endpoint session (`A_k4@16000` control, paired over 15 prompts)

| arm | Δ accepted prefix [95% CI] | Δ TPF [95% CI] | Δ tok/s [95% CI] |
|---|---|---|---|
| untrained | −0.891 [−1.043, −0.790] | −0.382 [−0.458, −0.332] | −15.92 [−18.36, −13.39] |
| start@12800 | −0.061 [−0.147, +0.014] | −0.011 [−0.053, +0.022] | −0.46 [−2.09, +0.95] |
| B_k6 | +0.009 [−0.108, +0.101] | +0.017 [−0.037, +0.057] | +0.49 [−1.47, +2.36] |
| C_k8 | −0.040 [−0.155, +0.071] | −0.017 [−0.074, +0.035] | −0.63 [−2.81, +1.56] |
| D_curr | +0.000 [−0.145, +0.150] | +0.000 [−0.075, +0.070] | +0.02 [−2.91, +2.82] |

The untrained adapter is detected easily, so the harness can see a real effect. B_k6's
TPF point estimate clears the 0.0062 floor, but its interval includes zero.

Absolute values at K=4, endpoint session:

| arm | accepted prefix | TPF | tok/s (median) | × AR |
|---|---|---|---|---|
| untrained | 0.138 | 1.033 | 41.56 | 0.839 |
| start@12800 | 0.968 | 1.404 | 57.36 | 1.158 |
| A_k4 | 1.028 | 1.415 | 59.10 | 1.193 |
| B_k6 | 1.037 | 1.431 | 59.43 | 1.200 |
| C_k8 | 0.988 | 1.398 | 56.44 | 1.140 |
| D_curr | 1.028 | 1.415 | 56.20 | 1.135 |

A_k4 and D_curr share an identical prefix and TPF because both happened to accept 254
tokens over 247 cycles. They are different models: their acceptance histograms differ,
14 of 15 prompts and 52 of 256 held-out rows differ, and their adapter digests differ.

### Every checkpoint against the matched control

The frozen curve session paired every checkpoint against the shared start, not against
`A_k4` at the same step. So `scripts/u5_matched_pairing.py` re-pairs its recorded
per-prompt data, `arm@step` against `A_k4@step`, with the harness's own bootstrap and the
§3 floors (`u5_matched_step_pairing.json`):

| arm | 13200 | 13600 | 14400 | 15200 | 16000 |
|---|---|---|---|---|---|
| B_k6 Δ prefix | −0.012 | +0.012 | −0.000 | +0.019 | +0.009 |
| C_k8 Δ prefix | +0.064 | +0.015 | +0.033 | +0.039 | −0.040 |
| D_curr Δ prefix | −0.008 | +0.008 | −0.044 | +0.047 | +0.000 |

**No arm passes U5-1, U5-3 or U5-4 at any checkpoint.** C_k8 leads the control at four of
five checkpoints and trails at the endpoint. Its closest approach is step 15200: prefix
+0.039 [−0.014, +0.116], TPF +0.022 [−0.003, +0.055]. Across 45 tests (15 comparisons ×
3 metrics), exactly one is significant: D_curr@13200 is slower than the control. At 95%,
about two false positives would be expected.

**A single run moves as much as the arms differ.** The control's own K=4 accepted prefix
along its trajectory is 0.968 → 0.948 → 0.953 → 0.992 → 0.929 → 1.028, a range of 0.099
within one launch.

## 5. The U4 observation, accounted for

Evaluation is deterministic, so prefix values from different sessions are directly
comparable. The start scores 0.968 here, exactly as in U4-A.

| adapter | K=4 accepted prefix | Δ vs start |
|---|---|---|
| `runs/u4a/step-12800` (start) | 0.968 | — |
| U4 arm B1, +3200 K=8 steps (U4, one launch) | 1.057 | **+0.090** |
| A_k4, K=4, launch 1 | 1.028 | +0.060 |
| A_k4, K=4, launch 2 | 1.029 | +0.061 |
| A_k4, K=4, launch 3 | 0.976 | +0.008 |
| B_k6 | 1.037 | +0.069 |
| C_k8, K=8, launch 1 (scored) | 0.988 | +0.020 |
| C_k8, K=8, launch 2 (supplementary) | 1.083 | +0.115 |
| D_curr | 1.028 | +0.060 |

3200 more steps of *any* horizon, the control included, move prefix up from the start.
Identical launches of the same run land anywhere from +0.008 to +0.061 (K=4) or +0.020 to
+0.115 (K=8). U4 compared one K=8 launch against the start, with no matched control and
an evaluation-noise σ, and saw +0.090. That number is inside both ranges.

## 6. U5-5 — no degradation: FAIL by the letter

1. **Cacheless bit-exact losslessness at K=4, every arm: FAIL by the letter, zero
   unexplained.** The frozen launcher never runs this measurement.
   `scripts/u5_losslessness.py` measured it afterwards, with the `stage0` / U4-B4 method
   (`u5_losslessness.json`): 13 of 15 prompts bit-identical for every arm. The two
   divergences are identical in all four arms:
   * `{"name": "Ada Lovelace", …` at token 42: an exact bf16 tie, gap 0.0;
   * `| Country | Capital | …` at token 42: gap 0.125, exactly one ULP.

   The AR reference is identical across arms, so the adapter never reaches the AR path.
   These are ties in the backbone's bf16 arithmetic that a width-1 and a width-(K+1)
   forward break differently, the limit Act IV-S documented. U4-B4 passed because it
   checked 6 prose and factual prompts and no `structured` ones.
2. **Cached-regime AR agreement at least the chunking floor: PASS.** Every arm scores
   0.8528, exactly what the untrained adapter scores.
3. **No arm's prefix below the control by more than 0.0254: FAIL.** C_k8 is −0.040 at the
   endpoint (paired interval [−0.155, +0.071]), and D_curr is −0.044 at step 14400. The
   clause has no interval condition, so it fails as written. Both shortfalls are inside
   the measured K=4 launch spread (§7).

U5-5 fails without clause 1. Clause 1 was measured so the gate is not scored with a
missing input.

## 7. Supplementary training-launch robustness

*Not part of the preregistered scorecard.* Amendments A1–A4 (§11), fixed before any
endpoint was evaluated. `C_k8_r2` is not substituted for `C_k8`, the C launches are not
averaged, and no frozen gate is rescored.

### Launch identity

Proven by `results/act4u5/u5_replicate_provenance.json`. For each of `A_k4_r2`, `A_k4_r3`
and `C_k8_r2`, all of the following match the primary arm:

* the start checkpoint digests at replicate launch against the record made at the
  primary launch;
* the command line, token for token, excluding `--out` (argument hash `37218bf4…` for
  K=4, `75c926a2…` for K=8);
* `HF_HOME` and `PYTHONPATH`;
* training code and package digests across 5 snapshots;
* the end state: config hash, seed, block size, step, data position and RNG state.

Every final adapter diverged:

| launch | adapter digest |
|---|---|
| A_k4 | `14202b49…` |
| A_k4_r2 | `98ca42c2…` |
| A_k4_r3 | `d2f00398…` |
| C_k8 | `7c1673d6…` |
| C_k8_r2 | `3b38eae4…` |

### A_k4 launch spread and the A3 comparison

Replicate session, K=4:

| metric | A_k4 launches | A mean | A band | C1 effect (original) | C2 effect (replicate) | \|C1 − C2\| | A4 case |
|---|---|---|---|---|---|---|---|
| mean accepted prefix | 1.0283, 1.0288, 0.9759 | 1.0110 | 0.0529 | −0.0230 | +0.0716 | 0.0946 | 5 |
| TPF | 1.4145, 1.4371, 1.4035 | 1.4184 | 0.0336 | −0.0203 | +0.0245 | 0.0448 | 5 |
| tok/s | 57.68, 57.50, 56.34 | 57.17 | 1.35 | −0.47 | +1.24 | 1.71 | 5 |

The prefix band is **2.1×** the frozen U5-1 floor and the TPF band **5.4×** its floor.
Paired against the original control, one identical launch (`A_k4_r3`) scored −0.052
prefix [−0.147, +0.021] and −1.35 tok/s [−3.05, +0.13]. Launch noise alone produced a
bigger gap than the floor.

**A3 has nothing to qualify:** no arm passed a frozen gate. Descriptively, measured
against the A mean and band:

| arm | Δ prefix | > band | Δ TPF | > band | Δ tok/s | > band |
|---|---|---|---|---|---|---|
| B_k6 | +0.0259 | no | +0.0130 | no | +0.55 | no |
| C_k8 | −0.0230 | no | −0.0203 | no | −0.47 | no |
| C_k8_r2 | +0.0716 | yes | +0.0245 | no | +1.24 | no |
| D_curr | +0.0173 | no | −0.0039 | no | +0.32 | no |

### C_k8 replicate: A4 case 5 on every metric

The two C launches have opposite-signed effects on all three metrics:

> **C_k8 is highly launch-sensitive under this training setup; no stable positive
> horizon-training effect is established.**

Two other A4 conditions also hold, and are reported as A4 requires:

* *Case 2, on prefix:* only the replicate clears the A band. The C_k8 effect is
  launch-sensitive; the original frozen U5 result may be real but is not robust to an
  independent C training launch.
* *Case 3, on all three metrics:* the C launches differ by more than the A band. The
  assumption that A_k4 launch spread is representative of C_k8 training variability is
  not supported by this check.

**Mechanism profile.** Teacher-forced agreement relative to the A mean:

| launch | Δ +2 | Δ +3 | Δ +4 | profile |
|---|---|---|---|---|
| C1 (original) | +0.0260 | −0.0026 | +0.0104 | first slot only |
| C2 (replicate) | +0.0104 | −0.0026 | +0.0104 | none |

**Replication outcome: neither performance nor the mechanism profile replicates.**

The scored `C_k8` happened to be the weaker of the two launches. That does not license
using the other one. The finding is that the sign of "the C effect" depends on which
launch you happen to run.

**Limits.** Two C_k8 launches do not establish C's standard deviation, a confidence
interval across training launches, heteroskedasticity, or a distribution of K=8
outcomes. This check answers only whether the C effect survives one independent rerun,
and whether the difference between the launches looks roughly compatible with the A_k4
spread. It does not.

## 8. Mechanism analysis (§6 of the criteria; reported, not gated)

Single launches, so descriptive only.

**Offset profile**, endpoint session, teacher-forced agreement relative to `A_k4@16000`:

| arm | +2 (first speculative) | +3 | +4 |
|---|---|---|---|
| B_k6 | +0.008 | +0.000 | +0.016 |
| C_k8 | +0.027 | −0.012 | +0.016 |
| D_curr | −0.004 | +0.004 | +0.031 |

C_k8's gain sits at +2 with +3 negative. That is the shape criteria §1 predicted from U4
(Mechanism B, generic sharpening), but on an arm with no aggregate gain, and it did not
replicate: C_k8_r2 moved +2 by only +0.010.

**Draftability stratification.** K=4 acceptance by frozen-model draftability quintile,
q1 = most draftable:

| arm | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|
| start | 0.379 | 0.301 | 0.320 | 0.327 | 0.237 |
| A_k4 | 0.366 | 0.301 | 0.268 | 0.288 | 0.308 |
| B_k6 | 0.418 | 0.359 | 0.281 | 0.288 | 0.288 |
| C_k8 | 0.425 | 0.346 | 0.268 | 0.301 | 0.263 |
| D_curr | 0.425 | 0.346 | 0.261 | 0.307 | 0.269 |

The missing-predecessor-bridging hypothesis predicted that longer-horizon training
improves the *least* draftable quintiles. The longer-horizon arms are higher in q1 and
lower in q5 than the control. **Non-support**, from single launches whose quintile values
U4 already measured moving ±0.05 on an identical adapter.

**Transfer efficiency** (Δ endpoint prefix per training hour): B_k6 +0.007, C_k8 −0.032,
D_curr 0.000. Compute is matched, so this ranks exactly as the prefix deltas do.

## 9. Diagnostics: `K_decode = 6` and `8` (endpoint session, never decisive)

| arm | K=6 prefix | K=6 tok/s (× AR) | K=8 prefix | K=8 tok/s (× AR) |
|---|---|---|---|---|
| untrained | 0.166 | 40.94 (0.827) | 0.180 | 38.41 (0.776) |
| start@12800 | 1.058 | 56.80 (1.147) | 1.087 | 54.22 (1.095) |
| A_k4 | 1.008 | 57.03 (1.152) | 1.100 | 54.54 (1.101) |
| B_k6 | 1.024 | 56.22 (1.135) | 1.129 | 54.20 (1.095) |
| C_k8 | 1.061 | 57.64 (1.164) | 1.116 | 56.22 (1.135) |
| D_curr | 1.041 | 57.27 (1.157) | 1.053 | 51.16 (1.033) |

K=4 remains the fastest decode width for A and B (59.1 and 59.4 tok/s). C_k8 is the
fastest arm at K=6 and K=8, the widths it was trained for, which is unsurprising and
irrelevant to the U5 question.

## 10. Findings about the rig and the method

1. **Training launches are not reproducible on MLX/Metal, and the spread exceeds the
   frozen floors** (A1, §7). Launch-to-launch divergence starts at the second optimizer
   step. Any two separately trained adapters are one draw each.
2. **The frozen curve session paired against the shared start, not the matched
   control.** Re-pairing its recorded data changes nothing: no matched comparison
   passes.
3. **The frozen launcher omitted U5-5 clause 1.** It was measured afterwards and fails by
   the letter on bf16 ties only.
4. **Evaluation is bit-deterministic across sessions** for prefix, TPF and teacher-forced
   agreement; tok/s varies by at most 0.5.
5. **`u5-eval` records each prompt's tok/s as the best of three keyed repeats.** The
   repeats decode identical tokens, so this selects on timing jitter only and applies to
   every arm equally. Noted, not changed.
6. **An agent-owned process group dies with the agent.** The primary run's first launch
   was killed by a session restart. Long runs are now launched in their own session
   (`AGENTS.md`).

## 11. What this changes elsewhere

* [`act4u4_results.md`](act4u4_results.md) has a dated addendum. U4's "result nobody
  pre-registered" and its B2-vs-B1 curriculum comparison were single launches inside the
  launch spread, and are not supported. U4-A's along-trajectory saturation analysis and
  the K=8-vs-K=4 decode comparison stand.
* [`RESULTS_SPECULATIVE.md`](RESULTS_SPECULATIVE.md) has a dated addendum. Its §12 G was
  run and found nothing. Its decode measurements of the fixed adapter
  `runs/u4b1/step-16000` stand.
* `FINDINGS.md` carries the qualifications and two new entries; `AGENTS.md` carries the
  verdict and the rule "one training launch is one draw".

## 12. What U5 cannot conclude

* **Small effects.** Paired intervals are about ±0.10 prefix at 15 prompts. A true
  horizon effect below that is not excluded.
* **Horizon vs density** (U5-B, criteria §7). There is no effect to decompose, so U5-B
  is not warranted.
* **IFM's scale.** Each arm trained on about 0.8 M tokens; IFM's curriculum spends about
  2.46 B tokens per stage. U5 says nothing about cross-horizon transfer at that scale.
* **Training variance in general.** The spread comes from 3 K=4 launches and 2 K=8
  launches, all from one start. It shows the variance is large; it is not an estimate of
  its size.

## Next

**Close the training-horizon line at this budget.** Nothing here justifies U5-B or
longer runs of the same design. If the question is reopened, the design needs at least
three launches per arm and a much larger evaluation set, because both the launch spread
and the 15-prompt intervals are larger than every training effect Act IV-U has chased.

For the decoder itself, Act IV-S's runner-up remains the cheapest untested idea:
decouple draft width from verify width.

---

## Reproduction

```bash
cd /Users/johnathonkillaly/code/diffusion-u5
export HF_HOME=/Volumes/SHUTTLE PYTHONPATH=$PWD/src
PY=/Users/johnathonkillaly/code/diffusion_project/.venv-unsloth/bin/python
bash scripts/u5_launch.sh           # primary: 4 arms, curve + endpoint evaluation, frozen report (~8 h)
bash scripts/u5_replicates.sh       # A2/A4: 3 replicate launches, provenance, supplementary session (~4.5 h)
$PY scripts/u5_losslessness.py      # U5-5 clause 1, measured after the fact (~6 min)
$PY scripts/u5_matched_pairing.py   # every checkpoint vs matched A_k4, from recorded data (seconds)
$PY scripts/u5_replicate_report.py  # supplementary robustness section (seconds)
```

Launch long runs detached, in their own session (see `AGENTS.md`).

The two analysis scripts also run on the committed copies:
`--eval results/act4u5/raw/eval_curve.json` for `u5_matched_pairing.py`, and
`--eval results/act4u5/raw/eval_replicates.json` for `u5_replicate_report.py`.
