# Act IV-U3 — Acceptance Scaling

Scored against [`act4u3_preregistered_criteria.md`](act4u3_preregistered_criteria.md),
frozen before the run continued and before any checkpoint past step 400 was evaluated.

**Verdict: `ACCEPTANCE AND SPEED SCALE`.**

---

## Question

> Does additional Uno diffusion-distillation training increase accepted-prefix length
> enough to produce a predictable increase in TPF and real wall-clock throughput under
> the already-proven transactional decoder?

Answer: **yes, and every link in the chain moved together.** Training 8× longer, with
the decoder, backbone, adapter architecture, objective, data and harness all held
constant, took K=4 from 1.056× to **1.168× AR** — and K=4 overtook K=2, reversing the
ordering U2 measured.

## U2 baseline

The U2 headline (`runs/uno-bench/u2_transactional.json`) was measured in a different
session, so it is **not** the comparator. Every U3 number below comes from one
evaluation session with its own in-session AR baseline of **48.35 tok/s**.

## Training continuation — and why this is not a continuation

**Exact optimizer resume from the Act IV-U checkpoint was impossible.** That run wrote
only `adapter.safetensors`; AdamW's moments and the data-iterator position were never
saved. The pre-registered fallback was a single continuous run from scratch with the
identical seed, checkpointing along the way, on the assumption that determinism would
reproduce steps 0–400.

**Gate U3-0b tested that assumption and it failed.**

| | orig-400 (archived) | re-run step-400 | Δ | within ±0.03? |
|---|---|---|---|---|
| K=2 teacher-forced agreement | 0.4062 | 0.3359 | −0.0703 | **no** |
| K=4 teacher-forced agreement | 0.2240 | 0.2240 | +0.0000 | yes |

The K=4 tie looked like a reproduction. It is not: comparing the adapters directly,
**0 of 256 tensors are bit-identical**, mean |Δ| ≈ 0.003 against a `lora_a` init scale
of 0.01. The K=4 match was a coincidence at reporting granularity (86/384 positions
either way).

**Cause, found and fixed.** `evaluate()` builds batches with `build_uno_batch`, which
drew corruption rates and noise tokens from the **global** MLX RNG — the same stream
training draws from. Changing only the evaluation *schedule* (`eval_every` 50→100,
`eval_batches` 4→8) therefore shifted every subsequent training noise draw, and the two
runs diverged despite identical seeds and identical training configuration. Batch
construction now accepts an explicit PRNG key and evaluation passes one, so evaluation
consumes no global RNG. Two tests lock this in, one of which asserts the un-keyed path
*does* still advance the global stream so the check cannot go vacuous.

**Consequence for interpretation, stated plainly:** the U3-A curve is a **single
self-consistent run** — one continuous optimizer trajectory with checkpoints along it —
and every within-curve comparison is valid. `orig-400` is a *separate run*, reported as
a reference point, not as step 400 of this curve. That the two step-400 points land at
nearly the same performance (K=4: 1.069× vs 1.056× AR, acceptance 0.2271 vs 0.2275)
suggests the divergence is seed-level noise rather than anything systematic, but that
is an observation, not a reproduction.

Training: 3200 steps, 4574 s (76 min), backbone byte-identical throughout.

## Acceptance scaling

Held-out. 128 validation rows for teacher-forced metrics; 15-prompt held-out suite,
48 tokens, greedy, snapshot transaction mode, 3 repeats for free-running metrics.

### K=4 — every link moves

| step | val TV | teacher-forced agree | free acceptance | TPF | fwd/token | tok/s | ×AR |
|---|---|---|---|---|---|---|---|
| *orig-400* | *1.1011* | *0.2240* | *0.2271* | *1.2834* | *0.7792* | *51.69* | *1.069* |
| 400 | 1.1041 | 0.2240 | 0.2275 | 1.2789 | 0.7819 | 51.05 | 1.056 |
| 800 | 1.0768 | 0.2370 | 0.2710 | 1.3358 | 0.7486 | 52.93 | 1.095 |
| 1200 | 1.0438 | 0.2604 | 0.2891 | 1.3662 | 0.7319 | 52.48 | 1.085 |
| 1600 | 1.0311 | 0.2708 | 0.2782 | 1.3458 | 0.7431 | 54.64 | 1.130 |
| 2400 | 1.0258 | 0.2630 | 0.3320 | 1.4090 | 0.7097 | 54.94 | 1.136 |
| **3200** | **0.9862** | **0.2917** | **0.3415** | **1.4314** | **0.6986** | **56.45** | **1.168** |

Validation TV falls monotonically. Free-running acceptance rises **+50%** (0.2275 →
0.3415). TPF rises **+12%**. Wall clock rises **+10.6%** (51.05 → 56.45 tok/s).

### K=2 — improves, but noisier and now behind

| step | val TV | teacher-forced agree | free acceptance | TPF | tok/s | ×AR |
|---|---|---|---|---|---|---|
| 400 | 0.6121 | 0.3359 | 0.5205 | 1.2020 | 50.46 | 1.044 |
| 800 | 0.5502 | 0.3984 | 0.5669 | 1.2350 | 50.73 | 1.049 |
| 1200 | 0.5496 | 0.4375 | 0.6000 | 1.2522 | 52.70 | 1.090 |
| 1600 | 0.5608 | 0.3828 | 0.5470 | 1.2224 | 50.34 | 1.041 |
| 2400 | 0.5449 | 0.4297 | 0.6282 | 1.2654 | 52.06 | 1.077 |
| **3200** | **0.5368** | **0.4453** | **0.6143** | **1.2522** | **52.52** | **1.086** |

K=2 acceptance is much higher in absolute terms (0.61 vs 0.34) but its TPF ceiling is
1.5 against K=4's 2.5, so the extra acceptance buys less. **K=4 overtook K=2 between
steps 800 and 1600** and finishes 7.6% ahead — the reverse of U2, where K=2 led.

## Future-horizon scaling

Teacher-forced agreement by future offset, K=4 (offset +4 does not exist: a block of 4
offers 3 speculative slots):

| step | +1 | +2 | +3 |
|---|---|---|---|
| 400 | 0.406 | 0.141 | 0.125 |
| 800 | 0.438 | 0.188 | 0.086 |
| 1200 | 0.469 | 0.164 | 0.148 |
| 1600 | 0.477 | 0.195 | 0.141 |
| 2400 | 0.477 | 0.156 | 0.156 |
| 3200 | **0.492** | **0.242** | **0.141** |

Relative improvement: **+1 → +21%, +2 → +72%, +3 → +13%.** The middle horizon improved
most, so the gain is not merely a better next-token predictor — the *horizon is
extending*. Per the brief's §16, that is the condition under which K=4 should overtake
K=2, and it did.

## Decoder cost model

Fitted from measured per-stage timings (`cost_model.py`), inverted to ask what
acceptance each speedup target requires. At step 3200:

| | K=2 | K=4 |
|---|---|---|
| ceiling if everything accepted | 1.343× AR (TPF 1.5) | **2.107× AR (TPF 2.5)** |
| actual | 0.6143 → 1.086× | 0.3415 → 1.168× |
| 1.10× needs per-slot accept | 0.4570 | 0.3940 |
| 1.20× | 0.6803 | 0.4899 |
| 1.30× | 0.9037 | 0.5715 |
| 1.50× | **unreachable** | 0.7068 |
| 2.00× | **unreachable** | 0.9564 |

This is the most decision-relevant table in U3. **K=2 is nearly exhausted** — it is at
0.614 acceptance against 0.680 needed for 1.20×, and 1.50× is structurally impossible
no matter how good the adapter gets. **K=4 has real headroom**: 1.30× needs 0.572 and
it is at 0.342.

### Does the model actually predict measurement?

Predicted tok/s exceeds measured by **+3.9% to +10.2%** at every checkpoint and both
block sizes — a *systematic* optimistic bias, not scatter (plot 7). The model accounts
for the three timed stages only; the missing ~8% is per-cycle Python and dispatch
overhead outside them. So the model is reliable for **ranking and for relative
acceptance requirements**, and should be read as an upper bound on absolute throughput.
Reported rather than tuned away.

## Draftability-gap stratification — Outcome B

Acceptance by frozen-model draftability quintile (q1 = most draftable), K=4:

| step | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|
| 400 | 0.382 | 0.250 | 0.211 | 0.171 | 0.113 |
| 3200 | 0.408 | 0.224 | 0.289 | 0.211 | 0.237 |
| **Δ** | **+0.026** | −0.026 | +0.079 | +0.039 | **+0.125** |

**The hardest quintile improved most, by ~5× the improvement of the easiest.** This is
pre-registered **Outcome B**: training is genuinely learning to bridge
missing-predecessor dependence, not merely getting better at regions that were already
draftable. It also explains a secondary observation — the gap↔acceptance correlation
does not strengthen with training (r ≈ −0.23 at both ends), because training is
*flattening* the very relationship the metric measures.

One correction to an Act IV-U2 claim: at K=4 the draftability gap and entropy are
**comparable** predictors here (step-3200: r(gap) = −0.2423, r(entropy) = −0.2222).
U2's finding that the gap was ~2.2× the predictor was measured at **K=8 with n=2688**.
It is a K=8 result and should not be quoted as a general one.

## Gate results

| gate | result | evidence |
|---|---|---|
| **U3-0** integrity | **PASS** | backbone digest identical at every checkpoint and after every benchmark |
| **U3-0b** reproduction | **FAIL** | 0/256 adapter tensors match; cause found (eval RNG coupling) and fixed |
| **U3-1** semantic preservation | **PASS** | AR-agreement = **0.8528 at every checkpoint and both K** — identical to the U2 backbone chunking floor; suite of 376 tests green, incl. exact greedy losslessness for arbitrary adapter weights |
| **U3-2** acceptance scaling | **PASS** | K=4 free acceptance 0.2275 → 0.3415 across four successive checkpoints; teacher-forced +0.068 (> 0.03) |
| **U3-3** TPF scaling | **PASS** | K=4 TPF 1.2789 → 1.4314, direction consistent with the cost model |
| **U3-4** realized throughput | **PASS** | K=4 +5.40 tok/s (51.05 → 56.45) against repeat-noise 1.95 tok/s ≈ **2.8σ**; milestone **≥1.15× "meaningful"** reached (1.168×), below 1.25× "strong" |
| **U3-5** generalization | **PASS** | all metrics on the held-out validation split and the held-out prompt suite |
| **U3-6** no quality trade | **PASS** | verifier untouched; AR-agreement constant; backbone unchanged |

## Failure modes

1. **Evaluation perturbed training through a shared global RNG.** The defect that broke
   U3-0b. It is subtle precisely because it is invisible in the training config: two
   runs with identical seeds, identical data and identical hyperparameters diverge
   because you looked at them differently. Fixed with explicit PRNG keys and locked by
   a test that would fail if the fix were reverted.
2. **Only the adapter was checkpointed in Act IV-U**, which made exact resume
   impossible and forced this whole run to be re-done from scratch. The trainer now
   writes optimizer state and RNG metadata; a round-trip test asserts the moments are
   restored, not just the weights.
3. **A near-miss reporting error.** K=4's teacher-forced agreement matched to four
   decimals across the two runs and would have been reported as a reproduction. Only a
   direct tensor comparison showed the adapters differ everywhere.
4. **The cost model is systematically optimistic by ~8%.** Not a bug, but a real limit
   on what it can be used for.

## Interpretation

The causal chain the brief set out to test held end to end:

```
8x more training
  -> validation TV        1.1041 -> 0.9862   (monotone)
  -> teacher-forced agree 0.2240 -> 0.2917
  -> free acceptance      0.2275 -> 0.3415   (+50%)
  -> TPF                  1.2789 -> 1.4314   (+12%)
  -> forwards/token       0.7819 -> 0.6986   (-11%)
  -> wall clock            51.05 -> 56.45    (+10.6%, 2.8 sigma)
```

Same frozen backbone, same adapter architecture, same decoder, same verification
guarantee, same hardware. **Lower loss did translate into more accepted tokens and then
into real throughput** — the objective/metric mismatch that would have been the
alternative finding did not appear.

Two qualifications:

- **The relationship is sub-linear and decelerating.** Acceptance rose 50% while
  wall clock rose 10.6%. Doubling the run again should not be expected to double
  anything: from the cost model, 1.30× AR at K=4 needs acceptance 0.572, which is 68%
  above where 3200 steps landed.
- **K=2 is close to structurally exhausted** and K=4 is not. Any further scaling work
  should move to K=4, and K=8 now finally has a case (its ceiling is higher still), but
  only because K=4 acceptance actually improved — which was the pre-registered
  precondition for looking at K=8 at all.

## Verdict

```
ACCEPTANCE AND SPEED SCALE
```

Seven of eight gates pass. U3-0b fails, and its failure is a reproducibility defect in
*our* harness rather than a property of the method — found, explained, fixed, and
tested, with the interpretive consequence (this is one self-consistent run, not a
continuation of the archived one) stated rather than glossed.

## Next step

1. **Continue to ~6400 steps at K=4.** The curve had not flattened: validation TV was
   still falling at 3200 and acceptance had not plateaued. The cost model says 1.30× AR
   is reachable and quantifies exactly how much acceptance it needs.
2. **Evaluate K=8** — now justified, since K=4 acceptance improved and K=8's ceiling
   is higher again.
3. **Stage B, the official Uno block curriculum** (2→4→6→8→12→16). Our run is fixed at
   K=4; IFM's progressive schedule is the most likely source of a further gain and is
   the obvious controlled comparison.
4. **Dynamic-K routing (U4)** remains gated behind this chain holding, which it now
   does — but note the draftability-gap correlation *weakened* with training, so a
   router should be designed against a re-measured signal, not U2's numbers.

## Artifacts

`runs/u3a/result.json` (training), `runs/u3a/eval.json` (all checkpoints, one session),
`runs/u3a/step-*/` (adapter + optimizer state, resumable).
Machine-readable: `results/act4u3/u3_checkpoints.csv`,
`u3_acceptance_histogram.csv`, `u3_draftability.csv`, `u3_cost_model.csv`,
`u3_summary.json`. Plots 0–7 as PNG in the same directory.

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py train \
  --block-size 4 --steps 3200 --lr 1e-5 --seed 20260903 \
  --checkpoint-steps 400,800,1200,1600,2400,3200 --out runs/u3a
```

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py u3-eval \
  --checkpoint step-400=runs/u3a/step-400 --checkpoint step-3200=runs/u3a/step-3200 \
  --block-sizes 2,4 --out runs/u3a/eval.json
```
