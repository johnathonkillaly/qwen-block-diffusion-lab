# Act IV-U success criteria — pre-registered

**Written after the AR baseline and the untrained-adapter control were measured, and
before any training run.** Frozen from here. Amendments, if any, go in §8 with a date
and a reason; the original text is never edited.

Thresholds are anchored to *measured* quantities rather than chosen for
attractiveness. Where a criterion could be met by a boring mechanism, the boring
mechanism gets its own control.

---

## 0. Measurements this document is anchored to

All from `runs/uno-baseline/baseline.json` and
`runs/uno-bench/control1_untrained.json`, Apple M4 Max / 128 GB, MLX 0.32.1,
Qwen3.5-4B-Base BF16, greedy, 48 tokens, 15-prompt held-out suite.

**Frozen AR baseline**

| | |
|---|---|
| held-out NLL / perplexity (WikiText-103 val, 64×256 tok) | 2.2635 / **9.617** |
| next-token accuracy | **52.38%** |
| median decode speed | **48.03 tok/s** (49.08 in the bench harness) |
| tokens per forward, by definition | **1.000** |
| peak unified memory | 12.34 GB |
| backbone digest | recorded in both artifacts |

**Control 1 — untrained adapter** (zero-init `lora_b`, so a bit-exact no-op; the
proposals are the frozen model reading uniform noise)

| B | spec acceptance | TPF | tok/s | vs AR | max possible TPF | **TPF needed to break even on wall clock** |
|---|---|---|---|---|---|---|
| 1 | — | 0.980 | 39.91 | 0.81× | 1.00 | 1.205 |
| 2 | **0.182** | 0.754 | 34.02 | 0.69× | 1.50 | **1.088** |
| 4 | **0.071** | 0.723 | 30.12 | 0.61× | 2.50 | **1.178** |
| 8 | **0.023** | 0.710 | 25.37 | 0.52× | 4.50 | **1.373** |

The break-even column is measured, not assumed: it is the ratio of a Uno forward's
cost to an AR forward's cost at that block size, so it already includes the wider
draft/verify forwards, the replay tax, and Python overhead.

**Two structural facts that constrain what can be claimed**

- `max TPF = (L+1)/2`. At `K=1` that is exactly 1.0, so **K=1 can never be a speedup**
  and serves only as a control.
- **`K=1` trains nothing.** With `L=1` there are no draft rows and the LoRA mask is
  all-zero. The brief's "Gate 1 — K=1 sanity" is vacuous under Uno's actual design, so
  the learning-sanity gate is defined at **K=2** instead. Recorded as a deviation.

---

## 1. Gate 0 — integrity. **Required. Failure voids everything.**

1. The backbone byte-digest is identical before and after training.
2. `trainable_parameters()` contains only `lora_a`/`lora_b`, asserted at setup.
3. Trainable parameters are **< 1%** of the backbone.
4. With the adapter disabled, the model reproduces the recorded AR baseline
   generations token-for-token.
5. Student and teacher logits are **bit-identical at the seed position** (the gate is
   not leaking onto clean rows).

Status at pre-registration: all five hold — stage 0 is **26/26**.

## 2. Gate 1 — does the adapter learn anything? *(sanity, at K=2)*

Teacher-forced agreement at future slot **+1**, held-out, `corruption="full"`:

| outcome | condition |
|---|---|
| **PASS** | trained agreement ≥ **untrained + 10 pp**, and the shuffled control does **not** improve by more than 3 pp |
| **FAIL** | otherwise |

If Gate 1 fails there is no reason to read anything below it.

## 3. Gate 2 — future prediction at K=4

Held-out, teacher-forced, `corruption="full"`. Untrained per-slot agreement is the
comparator, and the shuffled control must stay flat.

| outcome | condition |
|---|---|
| **STRONG** | mean spec agreement ≥ 0.50, and slot +1 ≥ 0.65 |
| **MODERATE** | mean spec agreement ≥ 0.25 |
| **WEAK** | mean spec agreement ≥ untrained + 10 pp |
| **NULL** | below untrained + 10 pp |

Reported alongside the free-running acceptance rate, which is a *different* number
(teacher-forced conditions on corpus text; decoding conditions on the model's own
output). Neither may be quoted as the other.

## 4. Gate 3 — verifier correctness. **Required.**

Adapter + verifier must reproduce frozen-AR greedy output **exactly**, in the
cacheless reference regime, on every prompt in the held-out suite, at K = 1, 2, 4, 8.
Target: **100% token equality**.

This is a property of the *algorithm*, not of training, so it must hold for an
untrained adapter too. Status at pre-registration: holds at K=1,2,4 in stage 0.

**The cached regime is reported separately and is not held to exact equality**,
because the backbone is not chunking-invariant: AR-with-cache and AR-without-cache
already disagree with each other on this model, with no adapter involved. The
pre-registered criterion for the cached path is therefore:

> cached-Uno vs cached-AR agreement ≥ cached-AR vs uncached-AR agreement
> (i.e. Uno tracks AR at least as well as AR tracks itself).

## 5. Gate 4 — sequential-step reduction

Tokens per forward, free-running, on the held-out suite:

| outcome | condition |
|---|---|
| **bare minimum** | TPF > **1.000** at any K > 1 |
| **interesting** | TPF ≥ 1.5 |
| **strong** | TPF ≥ 2.0 |
| **very strong** | TPF ≥ 3.0 |

TPF ≤ 1.0 means the method uses more sequential model calls than plain AR and there is
no algorithmic win to convert.

## 6. Gate 5 — wall clock

Median tok/s on the held-out suite must exceed the AR baseline **measured in the same
harness on the same machine in the same session**. Equivalently, TPF must clear the
measured break-even in §0: **1.088 at K=2, 1.178 at K=4, 1.373 at K=8**.

If Gate 4 passes and Gate 5 fails, the verdict is
**ALGORITHMIC SIGNAL, NO SPEEDUP** — recorded as such, not as a win.

No number here is ever compared to IFM's ~3× or 2.71 TPF. Different hardware,
different model, different runtime, unpublished protocol.

## 7. Kill criteria — stop and diagnose rather than train longer

1. The backbone digest changes. **Void.**
2. Any trainable tensor is not `lora_a`/`lora_b`. **Void.**
3. The seed-position TV is non-zero (the gate leaks). **Void.**
4. The shuffled control matches the true-target arm within 3 pp → the adapter is not
   using the context and the result is uninterpretable.
5. `K>1` gives no acceptance advantage over the untrained control at any K.
6. Loss non-finite, or TV fails to fall over the first 25% of a run.
7. Trainable share rises above 1% of the backbone.

## 8. Interpretation rules, fixed in advance

- **A falling TV is not success.** Act I's lesson. Only held-out agreement and
  free-running acceptance count.
- **Teacher-forced agreement is not acceptance.** They are separate numbers; both get
  reported.
- **Losslessness is not speed.** Gate 3 is free — the verifier guarantees it for a
  random adapter. Passing Gate 3 alone is worth nothing on its own and must never be
  presented as the headline.
- **Single seed, one machine.** Differences smaller than the thresholds above are not
  effects.
- **Scale honesty.** IFM train 14.75B tokens over 28,125 steps on 16 GPUs. Anything we
  run here is ~4 orders of magnitude smaller. A null result at this scale is evidence
  about *this scale*, not about the method.
- If the measured outcome is "the algorithm is exactly lossless, the adapter learns
  something, and it is still slower than AR on this runtime", that is the honest
  finding and it gets written down as the finding.

## 9. Verdict vocabulary

Exactly one of:

```
PROCEED
PROMISING BUT ENGINEERING-LIMITED
ALGORITHMIC SIGNAL, NO SPEEDUP
NO MEANINGFUL SIGNAL
REJECT
```

## 10. Amendments

*(none)*
