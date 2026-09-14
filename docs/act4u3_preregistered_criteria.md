# Act IV-U3 success criteria — pre-registered

**Written before resuming training and before evaluating any checkpoint beyond step
400.** Frozen from here; amendments go in §10 with a date and a reason.

The decoder and the transactional-state mechanism are **frozen** at the Act IV-U2
implementation and are not to be modified unless a correctness bug is found. The only
variable in Stage A is **training duration**.

---

## 0. The causal chain under test

```
more useful adapter training
      -> higher accepted-prefix length
         -> higher TPF
            -> fewer forwards per generated token
               -> higher wall-clock tok/s
```

Each arrow is measured separately. **A falling TV loss is not a result.** If loss keeps
falling while acceptance plateaus, that objective/metric mismatch *is* the finding and
is more informative than training longer.

## 1. The U2 baseline this is scored against

Recorded mechanically from `runs/uno-bench/u2_transactional.json`, single session,
AR and Uno back to back, snapshot transaction mode, greedy, 48 tokens, 15-prompt suite.

| | K=2 | K=4 |
|---|---|---|
| free-running acceptance rate | 0.5413 | 0.2376 |
| mean committed / cycle | 2.4978 | 2.6552 |
| full-block rate | 0.5276 | 0.0255 |
| **TPF** | **1.2101** | **1.2789** |
| forwards / token | 0.8264 | 0.7819 |
| tok/s | 54.051 | 51.023 |
| **× AR** | **1.0879** | **1.0270** |
| AR baseline (same session) | 49.683 tok/s | 49.683 tok/s |

Held-out teacher-forced agreement at step 400 (`runs/uno-compare/compare_final.json`,
128 rows): spec-mean **0.3906** (K=2), **0.2031** (K=4); per-slot at K=4
`[1.000, 0.4141, 0.1094, 0.0859]`.

**Because absolute tok/s drifts between sessions with machine load, every U3
comparison must re-measure AR and the step-400 checkpoint in the same session as the
checkpoint under test.** Ratios are the quantity; absolute tok/s is context.

## 2. Two metric names that must never be mixed

- **verifier acceptance (teacher-forced)** — on held-out corpus text, does the
  student's argmax match the frozen model's argmax at future slot *j*? Computed by
  `compare`. Conditions on *corpus* predecessors.
- **free-running acceptance (decode)** — during actual generation, does the verifier
  accept the proposal? Computed by `bench`. Conditions on the *model's own* output.

They are different numbers (0.3906 vs 0.5413 at K=2 in U2). Every table states which.

## 3. Training continuation — and an honest disclosure

**Exact optimizer resume from the Act IV-U checkpoint is impossible.** That run saved
only `adapter.safetensors` (the `lora_a`/`lora_b` tensors). AdamW's first and second
moments and the data-iterator position were never written. Warm-starting with a fresh
optimizer would inject a discontinuity at exactly the step of interest and would be a
restart wearing a continuation's clothes.

**Nearest controlled comparison, chosen and recorded before running:** a *single
continuous run from scratch* with the identical config and seed (`20260903`),
check-pointing at the target steps. Both `mx.random.seed` and the `WindowSource`
`np.random.default_rng(seed)` are deterministic, so steps 0–400 reproduce the original
trajectory, and the curve past 400 has one unbroken optimizer trajectory.

**Validation of that claim is itself a gate** — see U3-0b. The trainer now saves
optimizer state so this cannot recur.

Checkpoints: **400, 800, 1200, 1600, 2400, 3200**.

## 4. Gates

### U3-0 — integrity. **Required.**
Backbone byte-digest unchanged across all training and evaluation. Only
`lora_a`/`lora_b` trainable.

### U3-0b — the re-run reproduces the original. **Required.**
The re-run's step-400 checkpoint must match the archived Act IV-U adapter's held-out
metrics to within measurement noise: teacher-forced spec agreement within **±0.03**
at both K=2 and K=4. If it does not, the "continuation" framing is void and the
original 400-step point must be reported as a separate run rather than as step 400 of
this curve.

### U3-1 — semantic preservation. **Required.**
Transactional verified decoding continues to satisfy the U/U2 correctness criteria:
exact greedy losslessness in the cacheless reference regime, and cached-mode
AR-agreement no worse than the backbone's own chunking floor. Any violation fails U3
outright regardless of speed.

### U3-2 — acceptance scaling
At least one of K=2 / K=4 shows a **sustained** improvement in held-out accepted-prefix
statistics over the step-400 baseline. "Sustained" means either:
- improvement at **two or more successive** checkpoints, or
- a best-validation checkpoint clearly above baseline by **≥ 0.03 absolute** spec
  agreement *and* not contradicted by its neighbours.

A single non-monotone spike is not sustained.

### U3-3 — TPF scaling
Improved acceptance produces improved TPF at the same K versus the U2 baseline, in a
direction consistent with the empirical decoder cost model (§5). A checkpoint whose
acceptance rose but whose TPF did not is reported as a **cost-model failure**, not
quietly dropped.

### U3-4 — realized throughput
At least one later checkpoint beats the same-K U2 baseline in wall-clock tok/s **by
more than benchmark noise**, where noise is the measured standard deviation across
repetitions in the same session. Descriptive milestones, not targets to steer toward:

```
>= 1.15x AR   meaningful
>= 1.25x AR   strong
>= 1.50x AR   very strong
```

### U3-5 — generalization
Acceptance improvements must appear on the **held-out** validation split and the
held-out prompt suite, not only on training-distribution metrics. A training-loss
improvement with flat held-out acceptance fails this gate.

### U3-6 — no quality trade. **Required.**
Throughput must not come from weakened verification. The verifier is unchanged and
output equivalence is re-checked at every checkpoint. Any drift in the losslessness
guarantee fails U3.

## 5. Decoder cost model

Built from measured per-stage timings (`proposal_ms`, `verify_ms`, `commit_ms`) rather
than assumed. It maps an accepted-prefix distribution to expected forwards/token and
expected tok/s, and is inverted to answer: **what acceptance is required for
1.10× / 1.20× / 1.30× / 1.50× / 2.00× AR?**

TPF is **not** assumed to map linearly to throughput — the model is fitted and then
checked against measurement (required plot 7). If predicted and measured tok/s diverge
materially, we do not understand the decoder economics and say so.

## 6. Draftability-gap stratification

Computed from the **frozen base model** and held fixed as the adapter trains — the
metric definition must not move. At every checkpoint, bucket held-out positions into
frozen-model draftability quintiles and measure acceptance per quintile. Pre-registered
readings:

- **Outcome A** — low-gap buckets improve, high-gap stays hard → the adapter is getting
  better at exploiting inherently draftable regions.
- **Outcome B** — mid/high-gap buckets improve → training is genuinely learning to
  bridge missing-predecessor dependence.
- **Outcome C** — all buckets saturate → architecture, not training, is the limit.

## 7. Stop conditions

Stop Stage A early if, across **multiple** checkpoints:
validation loss worsens persistently; acceptance regresses persistently; acceptance is
flat; correctness or backbone integrity fails; training becomes numerically unstable.

**Do not stop on one noisy throughput measurement.** Use validation loss and acceptance
trends together.

## 8. Scope fences

- **Stage A changes training duration and nothing else.** No rank, objective, data,
  curriculum, or decoder changes.
- **K=8 is not evaluated** unless K=4 acceptance improves substantially. Proving a
  poorly-trained K=8 is worse is not worth the compute.
- **No dynamic-K routing in U3.** That is U4, and only if this chain holds.
- Stage B (official Uno curriculum) runs only if inspecting the current IFM source
  reveals a real discrepancy with our schedule, and is labelled separately.

## 9. Verdict vocabulary

Exactly one of:

```
ACCEPTANCE AND SPEED SCALE
ACCEPTANCE SCALES, SPEED SATURATES
LOSS SCALES, ACCEPTANCE SATURATES
OFFICIAL CURRICULUM REQUIRED
NO FURTHER SCALING
REGRESSION
```

## 10. Amendments

*(none)*
