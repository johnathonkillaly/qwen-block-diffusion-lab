# Act IV-U4 success criteria — pre-registered

**Written before any U4 training step and before any U4 checkpoint is evaluated.**
The only thing that exists at the time of writing is the calibration in §2, which
measures the harness against a *frozen* Act IV-U3 checkpoint (step 3200) and contains
no U4 result. Frozen from here; amendments go in §12 with a date and a reason.

The decoder, the transactional-state mechanism and the adapter architecture are
**frozen** at the Act IV-U2/U3 implementation. Stage A's only variable is training
duration. Stage B's only variable is block size.

---

## 0. The question, and the two stages

Act IV-U3 established the causal chain: training raises acceptance, acceptance raises
TPF, TPF raises wall-clock throughput. U4 asks how far the *horizon* can be pushed
before the extra block size stops paying for itself.

```
U4-A   saturate K=4          does K=4 keep improving, and where does it stop?
U4-B   test K=8              is there learnable signal past +4, and does it pay?
```

Dynamic-K routing is explicitly out of scope. U4 measures the static frontier.

---

## 1. Notation: two conventions that must not be mixed

### 1.1 Future offsets

A block of size `K` has `K` supervised positions. Position `block_start + j` predicts
the token `j+1` steps after the last committed token, so:

> **future offset `+j` == supervised slot `j-1`, for `j = 1 … K`.**

Offset `+1` is the **seed row**. It runs with the adapter off, so its agreement with
the frozen model is `1.0` by construction. It is reported as an integrity check — a
value below 1.0 means the token gate is leaking onto clean rows and every acceptance
number downstream is void — and never as evidence of learning. Offsets `+2 … +K` are
the speculative ones.

**Act IV-U3's tables labelled these by speculative slot index**, so U3's `acc_plus_1`
is this document's `+2`. U3's finding is unchanged (later horizons improved more); the
labels shift by one. Every U4 table uses the convention above.

### 1.2 Acceptance

Three different numbers, never quoted as each other:

| name | what it is |
|---|---|
| **teacher-forced agreement** | on held-out corpus text, does the student's argmax match the frozen model's at offset `+j`? Conditions on *corpus* predecessors. |
| **free-running acceptance rate** | during generation, the fraction of *offered* speculative slots accepted. In `[0, 1]`. |
| **mean accepted prefix** | during generation, `E[a]` where `a` is the number of speculative tokens accepted before the first rejection. In `[0, K-1]`. |

At block size `K` the last two are related by `mean accepted prefix = (K-1) x rate`
only because the rate is *defined* as the mean over offered slots; they are not
interchangeable across different `K`. Per §19 of the brief, **mean accepted prefix is
the primary quantity** — a higher acceptance *rate* at a larger `K` can still commit
fewer tokens per cycle.

Committed tokens per cycle is `mean accepted prefix + 2` exactly, for every `K`.

---

## 2. Harness calibration — measured before freezing thresholds

Three full evaluations of the **same frozen adapter** (`runs/u3a/step-3200`), differing
only in the two noise seeds: the held-out corruption seed and the decode draft-noise
seed. Windows, prompts, adapter and machine are identical. Everything below is
therefore pure measurement variance, with zero contribution from training.

`runs/u4-calib/rep{1,2,3}.json`, 256 held-out rows, 15 prompts, 48 tokens, 3 repeats.

| quantity | mean | sd (n=3) | 2 sd |
|---|---|---|---|
| AR baseline tok/s | 49.04 | 0.114 | 0.23 |
| **K=4** validation TV | 0.9751 | 0.0029 | 0.0057 |
| **K=4** teacher-forced agreement (specs) | 0.3086 | 0.0057 | 0.0113 |
| **K=4** free-running acceptance rate | 0.3170 | 0.0042 | 0.0084 |
| **K=4** mean accepted prefix | 0.9510 | 0.0127 | 0.0254 |
| **K=4** full-block rate | 0.0849 | 0.0084 | 0.0169 |
| **K=4** TPF | 1.3909 | 0.0031 | 0.0062 |
| **K=4** median tok/s | 55.83 | 2.356 | 4.71 |
| **K=4** speedup vs AR | 1.1385 | 0.0465 | 0.0930 |
| **K=8** mean accepted prefix | 0.9803 | 0.0377 | 0.0754 |
| **K=8** TPF | 1.4055 | 0.0206 | 0.0412 |
| ms per extra draft position | 1.278 | 0.029 | 0.059 |

Per-offset teacher-forced agreement, same three replicates:

| offset | K=4 sd | K=8 sd |
|---|---|---|
| +2 | 0.0137 | 0.0081 |
| +3 | 0.0039 | 0.0039 |
| +4 | 0.0045 | 0.0090 |
| +5 | — | 0.0090 |
| +6 | — | 0.0039 |
| +7 | — | 0.0023 |
| +8 | — | 0.0060 |

**Pooled per-offset sd = 0.0072** (RMS over all ten entries). Individual per-offset
sds from `n = 3` are themselves uncertain by roughly 40%, so thresholds below use the
pooled value rather than the per-offset one.

### 2.1 Two things the calibration changes about how U3 must be read

**(a) Wall-clock tok/s is not comparable across sessions, and the reason is not
thermal.** The between-session sd of K=4 median tok/s is 2.36 (4.2% of the mean),
while the *within*-session repeat noise is 0.15. The difference is not jitter: it is
the draft-noise draw. Different noise realisations produce genuinely different
acceptance, hence different speeds. Every U4 comparison is therefore made inside one
session against an in-session AR baseline, with the decode noise stream **held fixed
across checkpoints** so the comparison is paired.

**(b) U3's reported "repeat noise" was ~12x too large, in the conservative
direction.** U3 drew fresh global-RNG draft noise on every repeat, so its three
"repeats" of a prompt decoded *different* token sequences; the 1.44–1.99 tok/s it
called repeat noise was mostly noise-draw variance, not measurement jitter. With the
stream keyed, repeats of a prompt are bit-identical decodes and the residual jitter is
0.10–0.18 tok/s. Gate U3-4 required beating a baseline by more than that inflated
figure, so it was *harder* than intended. **U3's conclusion stands and is understated,
not overstated.** This is recorded here because it is the kind of correction that
would otherwise look, later, like a result being re-described after the fact.

---

## 3. U4-A: what is held constant

Everything except the number of steps. Backbone, adapter architecture (rank 16, alpha
256, `q,k,v,o,gate,up,down` on the 8 full-attention layers plus all MLPs), TV-only
objective, uniform corruption schedule, `random_uniform` noise, WikiText-103 windows
of width 128, AdamW at 1e-5 with no decay, batch size 4, snapshot transaction mode,
the 15-prompt held-out decode suite, and the evaluation harness.

Verified mechanically: `config_hash` is a SHA-256 over the training config with
`steps`, `checkpoint_steps` and the evaluation schedule excluded, and a mismatch
against the resumed checkpoint's hash is reported at startup.

### 3.1 Checkpoints

Resume from `runs/u3a/step-3200`, checkpoint at **4800, 6400, 8000, 9600, 11200,
12800**. The brief lists 4800/6400/8000/9600/12800; 11200 is added so every interval
is exactly 1600 steps, which the saturation rule in §5 needs in order to compare
consecutive intervals without rescaling.

Training runs to 12800 in one job. The saturation rule's purpose is to make the "has
it saturated?" call **pre-registered rather than post-hoc**, not to schedule compute:
all checkpoints are evaluated regardless, and the rule is applied afterwards to
identify where saturation was first met. The run is aborted early only on the
regression condition in §9.

---

## 4. U4-A gates

### U4-A0 — integrity (required)

* the frozen backbone's digest is unchanged before and after every training run and
  every evaluation;
* seed-row agreement (offset `+1`) is exactly 1.0 at every checkpoint and every `K`;
* only `lora_a` / `lora_b` are trainable.

A failure here voids the stage. Not negotiable.

### U4-A1 — the resume is a continuation, not a restart (required)

Judged on the **mechanism**, not on a downstream metric, because U3-0b showed a
headline metric can match to four decimals while the adapters differ everywhere:

* `tests/test_uno_resume.py` shows that training `N` steps straight through and
  training `N/2` + checkpoint + reload + `N/2` produce **bitwise-identical** adapter
  tensors, with a companion test proving the check is not vacuous;
* the checkpoint carries all seven fields §2 of the brief asks for (adapter,
  optimizer, scheduler, step, PRNG streams, data position, config hash);
* the resumed run's config hash equals the checkpoint's.

**Disclosed limitation, stated before the run.** `runs/u3a/step-3200` was written
before `data_position` existed and before the training noise was keyed. Two
consequences, neither hidden:

1. The data position *is* recovered exactly, by replaying the draw counts a
   `WindowSource` is a pure function of (3200 training draws, 32 evals x 8 val draws).
   This is reconstruction, not approximation, and it is asserted.
2. The training **noise stream cannot be recovered**, because U3 drew from the global
   MLX RNG and saved no state. From step 3200 the noise is keyed on
   `(seed, TRAIN_NOISE, step)`. The distribution, the schedule and the seed are
   unchanged, and the optimizer and data are exact — so this is a continuation. It is
   **not** bit-identical to what an uninterrupted global-RNG run would have drawn, and
   no claim in U4 will say that it is.

### U4-A2 — K=4 saturation, decided by the rule in §5

Reported as met / not met at each checkpoint. Not pass/fail: both answers are results.

### U4-A3 — the causal chain either continues or breaks at a named link

Across 3200 → 12800 at K=4, report each arrow separately:

```
val TV  ->  teacher-forced agreement  ->  mean accepted prefix  ->  TPF
        ->  forwards/token  ->  tok/s
```

If TV keeps falling while acceptance is flat, **that mismatch is the finding** and is
reported as such rather than as a training failure.

### U4-A4 — decoder cost model v2

The v2 model adds two terms to v1, both **measured, never fitted**:

* `overhead_ms` — cache snapshot/restore, draft-block construction, the proposal
  round-trip to the host, the greedy accept test and the entropy probe, timed
  directly in `decode.py`;
* prefill — v1 predicted a steady-state rate and was compared against
  `tokens / wall_seconds`, which includes the prompt prefill.

Passes if, over all evaluated `(checkpoint, K)` pairs:

* mean `|error|` < 4%, **and**
* the errors do not all share one sign (v1's signature was +3.9% to +10.2%, all
  positive).

The v1 prediction is computed alongside v2 at every point and reported, so the
improvement is visible rather than asserted. Calibration §2 gives v2 a mean error of
+1.6% with mixed signs against v1's +7.9% all-positive, but that is one checkpoint;
the gate is over the whole U4 set.

### U4-A5 — horizon case, per §8 of the brief

Classified from the K=4 offset curves between step 3200 and the final checkpoint,
using the §6 improvement threshold:

* **Case A** — `+2` improves, `+3`/`+4` do not. Stronger shallow predictor, saturated
  parallel horizon.
* **Case B** — `+3` and/or `+4` keep improving. The horizon is still expanding.
* **Case C** — no offset improves. K=4 training has saturated.

### U4-A6 — the decision to enter K=8

Recorded with which gate fired:

* **Gate A** — §5 saturation met at some checkpoint; or
* **Gate B** — offsets `+3` **and** `+4` each improve by at least the §6 threshold
  between step 3200 and the final K=4 checkpoint, with the gain not concentrated in a
  single interval (at least two of the five intervals contribute a positive change at
  each of the two offsets).

If neither fires, K=8 is not run and that is the U4 result.

---

## 5. The K=4 saturation rule — frozen

Per 1600-step interval, computed on the **free-running** metrics and held-out TV:

```
interval is FLAT  iff
      Δ(free-running acceptance rate)   <  0.020
  and Δ(TPF)                            <  0.030
  and Δ(validation TV)                  > -0.015     (TV not falling meaningfully)

K=4 is NEAR SATURATION at checkpoint s  iff
      the interval ending at s and the interval before it are both FLAT
```

Calibration basis (§2), all at K=4:

| threshold | value | measured 2 sd | ratio |
|---|---|---|---|
| acceptance rate | 0.020 | 0.0084 | 2.4x |
| TPF | 0.030 | 0.0062 | 4.8x |
| validation TV | 0.015 | 0.0057 | 2.6x |

The acceptance threshold of 0.020 on the rate is equivalent to **0.060 on the mean
accepted prefix** at K=4 (2.4x its 2 sd of 0.0254); both are reported.

**Wall-clock tok/s is deliberately excluded from this rule.** Its 2 sd is 4.71 tok/s,
about 8% of the value, which is larger than any plausible per-interval change. It is
reported at every checkpoint but cannot adjudicate saturation. This follows §5 of the
brief.

For reference, applying this rule to U3's last 800-step interval (scaled to 1600
steps) gives Δaccept ≈ +0.019 and ΔTPF ≈ +0.045: flat on acceptance, not flat on TPF,
so **U3's final interval would not have been declared saturated**. The rule is not
constructed to fire immediately.

---

## 6. The offset-improvement threshold — frozen

An offset counts as showing **clear held-out learning** if its teacher-forced
agreement improves by

```
>= 0.022 absolute
```

between the reference checkpoint and the checkpoint under test. This is 3x the pooled
per-offset replicate sd of 0.0072 (§2). One number is used for every offset and every
`K`, because the per-offset sds come from `n = 3` and do not deserve to be
distinguished.

---

## 7. U4-B: K=8 gates

Pre-registered now, before any K=8 training exists.

The **reference point** for "improvement" is the K=8 behaviour of the best K=4
checkpoint — i.e. the same adapter evaluated at block size 8, which is measured in
Stage A and is the honest starting line for arm B1. Its calibrated values at step 3200
are already in §2 (`+5` 0.112, `+6` 0.117, `+7` 0.100, `+8` 0.044).

### U4-B0 — integrity (required)

As U4-A0.

### U4-B1 — long-horizon signal

Offsets `+5` **and** `+6` must each improve by at least 0.022 (§6) over the reference,
on held-out text. Below that, K=8 is not scientifically interesting and §24's stop
condition applies.

### U4-B2 — committed tokens, not acceptance rate

K=8 must exceed the **best K=4 TPF** measured in the same session. A higher acceptance
*rate* at K=8 that does not raise TPF fails this gate, which is the point of §19.

### U4-B3 — wall clock, two separate outcomes

Reported independently, never merged:

* `K=8 tok/s > AR tok/s` (same session);
* `K=8 tok/s > best K=4 tok/s` (same session) — **this is the U4 result**.

A difference smaller than the in-session repeat noise (0.10–0.18 tok/s, §2) is
reported as a tie.

### U4-B4 — quality (required)

Verified decoding remains lossless: identical tokens to the AR reference in the
cacheless regime at every `K`, and at least the cached-AR chunking floor in the cached
regime. Unchanged from Gate 3 of Act IV-U.

### U4-B5 — arms

At most two, per §14:

* **B1** — continue from the best K=4 checkpoint with block size 8.
* **B2** — the official IFM curriculum path. Confirmed from
  `ifm-ai/uno@training/configs/uno_3epoch_curriculum.yaml`: six stages
  `2, 4, 6, 8, 12, 16` of ~2.4576B tokens each, and confirmed from
  `training/trainer.py` that a stage transition changes **only** the block size —
  optimizer state, adapter and learning-rate schedule all continue uninterrupted.
  B2 is therefore `K=4 -> K=6 -> K=8` with everything else carried over.

Training K=8 from random initialisation is run only if needed as a control.

---

## 8. Draftability hypothesis (§20)

Stated before measurement: **the draftability gap becomes more predictive of
acceptance as the future offset grows.** Tested by the per-offset Pearson correlation
between the frozen model's gap `TV(p(y | true predecessors), p(y | noised
predecessors))` and whether the trained adapter's argmax matched the frozen model's.

Confirmed if `|r(gap)|` at offsets `+6…+8` exceeds `|r(gap)|` at `+2…+3` at the final
K=8 checkpoint. This is an analysis, not a gate, and **no router is built in U4**.

Note the calibration already shows `r(gap)` at K=4 to be small and unstable across
replicates (`+2` ranged −0.117 to −0.066, `+4` −0.148 to −0.043 across three
identical-adapter runs), so any per-offset claim must clear that spread.

---

## 9. Stop conditions

Training is aborted, and the abort reported, if:

* the backbone fingerprint changes (voids the stage);
* the saturation guard fires (vanishing gradient with saturated logits);
* the loss is non-finite;
* held-out TV in the in-training evaluation rises for 5 consecutive evaluations
  (500 steps) — a regression, not a plateau.

K=8 training additionally stops, per §24, if offsets `+5…+8` show no held-out learning
by the §6 threshold, or if mean accepted prefix plateaus below what K=4 economics
require, or if TV falls while TPF does not.

**Falling training loss is not a reason to continue.**

---

## 10. What would falsify the U4 hypothesis

The exciting outcome is that K=8 inherits K=4's short-horizon behaviour, learns
`+5…+8`, and overtakes K=4 on wall clock. Each of these is a separate way to fail, and
each is a publishable answer:

| observation | verdict |
|---|---|
| K=4 keeps improving through 12800 and K=8 overtakes it | `HORIZON SCALES` |
| K=4 flattens, K=8 overtakes it | `K4 SATURATES, K8 WINS` |
| K=8 learns `+5…+8` but stays slower | `K8 LEARNS BUT K4 REMAINS FASTER` |
| `+5…+8` learn, no throughput gain anywhere | `LONG-HORIZON SIGNAL, NO SPEED GAIN` |
| K=8 neither learns past +4 nor pays for itself | `K4 IS THE PRACTICAL FRONTIER` |
| nothing beyond U3 replicates | `NO ADDITIONAL HORIZON SCALING` |

---

## 11. Out of scope

No dynamic-K routing of any kind: no entropy router, no draftability router, no
acceptance-history router, no learned router. U4 establishes the static frontier.
Routing is U5's problem and is gated on knowing the best `K`, the training saturation
point and the per-position economics — which is exactly what this stage measures.

---

## 12. Amendments

*(none)*
