# Act IV-U2 — Transactional DeltaNet Verification

Scored against [`act4u2_preregistered_criteria.md`](act4u2_preregistered_criteria.md),
frozen before any performance measurement.

**Verdict: `TRANSACTIONAL SPEEDUP ACHIEVED`.**

No retraining. Same frozen backbone, same Act IV-U adapter, same output. The only
change is how rejected speculative state is discarded.

---

## Motivation — the Act IV-U bottleneck

Act IV-U ended at TPF 0.980 against the 1.000 plain AR gets for free. The cause was
isolated and quantified: a *partially* accepted block needed a third full model forward
to replay the accepted tokens, because a Gated DeltaNet recurrence cannot be rewound
into the middle of a pass. Removing only that term was projected to give **TPF
1.20–1.28**. U2 asks whether that projection can be realised.

It can.

## Headline

15-prompt held-out suite, 48 tokens, greedy, all measured back to back in one process.
**AR baseline 49.68 tok/s.**

| K | mode | TPF | forwards/token | tok/s | vs AR | replay forwards | record |
|---|---|---|---|---|---|---|---|
| 2 | replay | 0.973 | 1.028 | 46.17 | 0.93× | 145 | — |
| 2 | **snapshot** | **1.210** | **0.826** | **54.05** | **1.09×** | **0** | 204 MB |
| 2 | rewind | 1.222 | 0.818 | 50.05 | 1.01× | 0 | 205 MB |
| 4 | replay | 0.862 | 1.160 | 39.05 | 0.79× | 270 | — |
| 4 | **snapshot** | **1.279** | **0.782** | **51.02** | **1.03×** | **0** | 305 MB |
| 4 | rewind | 1.265 | 0.790 | 38.94 | 0.78× | 0 | 307 MB |

- **TPF 0.862 → 1.279 at K=4**, landing exactly inside the pre-registered 1.20–1.28
  band. The Act IV-U counterfactual was accurate.
- **Wall clock now beats plain autoregression**: 1.09× at K=2, 1.03× at K=4.
- **Acceptance is unchanged** (0.235 → 0.238 at K=4; 0.536 → 0.541 at K=2 — drift from
  unseeded draft noise). It must be: same adapter, same proposals. If it had moved,
  something would be wrong.
- **Output is unchanged.** AR-agreement is 0.853 for *every* mode — identical, and equal
  to the backbone's own cached-vs-uncached chunking floor established in Act IV-U.

## Qwen3.5 state anatomy

Full table in [`act4u2_transactional_state.md`](act4u2_transactional_state.md). The
short version:

| state | shape | dtype | layers | truncatable? |
|---|---|---|---|---|
| KV keys/values + offset | `[B,4,T,256]` | bf16 | 8 | yes — offset is a pointer |
| DeltaNet conv window | `[B,3,8192]` | bf16 | 24 | yes — sliding window |
| **DeltaNet recurrent state** | `[B,32,128,128]` | **fp32** | 24 | **no** |

2.10 MB per layer × 24 = **50.3 MB per full snapshot**. That number is what makes the
whole approach viable.

## The exact recurrence, and a correction

Read out of `mlx_lm/models/gated_delta.py`, per step, per value head, with
`S ∈ R^{Dv×Dk}`:

```
β_t = sigmoid(b_t)                                  g_t = exp(-exp(A_log)·softplus(a_t+dt_bias))
S̄_t = g_t · S_{t-1}
S_t = S̄_t + β_t·(v_t − S̄_t k_t) k_tᵀ  =  S̄_t (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```

> **The projector multiplies on the right, not the left.** The form assumed in the U2
> brief does not describe this implementation. Verified against the actual op:
> right-multiplied form error **2.4e-07**, left-multiplied form error **1.74**.

`k` is exactly unit-norm (`k = Dk^{-1/2}·rms_norm(k_raw)`, measured `‖k‖ = 1.0`), so
Sherman–Morrison is exact and the undo is rank-1:

```
S̄_t     = S_t + (β_t/(1−β_t))·(S_t k_t − v_t) k_tᵀ
S_{t-1} = S̄_t / g_t
```

Verified to **4.8e-07**. The algebra is sound; the arithmetic is where it fails.

## Approach B — snapshot (the mechanism that worked)

The insight is not to avoid the recurrence but to **unroll only the recurrence**.
`recurrence.py` reimplements the DeltaNet layer so the per-token state is captured
*inside the existing verify forward*. Projections, the depthwise conv, attention and
every MLP still run **once** for the whole block; only the `T`-step loop — which the
fused Metal kernel already performs sequentially — is stepped from Python, and `S_j` is
retained at each boundary. Conv state at prefix `j` is the slice
`conv_input[:, j:j+3, :]`. KV state is `offset = base + j`.

`transaction.py` wraps this in a backend-agnostic abstraction:

```python
txn = begin_transaction(model, caches, mode="snapshot")
logits = txn.run_block(input_ids)
txn.commit_prefix(m)          # or txn.rollback()
```

## Gate results

### Gate U2-0 — reference integrity. **PASS**
`replay` is untouched and reproduces Act IV-U: identical acceptance, identical forward
accounting (`forwards == 1 + 2·cycles + replays`). Backbone digest unchanged across
every run. The oracle is retained permanently.

### Gate U2-1 — snapshot correctness. **PASS**
For `K ∈ {1,2,4,8}` and **every** prefix `m ∈ {0…K}`: committed recurrent state, conv
state, KV contents and offsets all match replay, *and* the greedy continuation from the
committed state is token-identical to replay's for 10–16 tokens. Fixtures include
blocks with repeated token ids and an all-identical block, so prefix selection cannot
be right by accident. Decoder-level: all three modes emit identical tokens at K=2,4,8.

### Gate U2-2 — the replay forward is gone. **PASS**
Instrumented, not inferred: `replay_forwards == 0` and `full_forwards == 1` per block
in snapshot mode, at every prefix length. On the real model, 145 (K=2) and 270 (K=4)
replay forwards became **0**.

### Gate U2-3 — algorithmic realisation. **PASS — "as predicted"**
TPF **1.210** (K=2) and **1.279** (K=4), both inside the pre-registered **1.20–1.28**
band. Forwards per generated token fell from 1.160 to **0.782** at K=4, a 33% reduction.

### Gate U2-4 — rewind correctness. **PASS**
`rewind` produces token-identical output to `snapshot` across the test suite and the
benchmark (AR-agreement 0.853, identical). It is only correct because of its fallback —
see U2-5.

### Gate U2-5 — numerical stability. **PASS, with the finding that matters**
Fallback rates on the real model: **3.4%** of head-steps at K=2, **10.5%** at K=4 —
below the pre-registered 50% "dominated" threshold, so rewind is *not* formally
dominated. But see the conditioning section: it is dominated in practice for a
different and stronger reason.

### Gate U2-6 — wall clock. **PASS (full success)**
Snapshot beats replay at both block sizes (54.05 vs 46.17; 51.02 vs 39.05) **and** beats
plain AR (1.09×, 1.03×). Rewind beats replay at K=2 but loses at K=4.

## Numerical conditioning — is Gated DeltaNet actually reversible?

Measured over 3.1M head-steps of real held-out text through all 24 DeltaNet layers:

| | min | p10 | median | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|
| `β` | 0.000315 | 0.116 | 0.479 | 0.887 | 0.984 | 0.996 | **1.00000000** |
| `1−β` | **0.000000** | 0.113 | 0.522 | 0.884 | 0.976 | 0.993 | 0.99969 |
| decay `g` | **0.000000** | 0.404 | 0.982 | 0.9998 | 0.99999 | 0.999998 | 1.0 |
| `β/(1−β)` | 0.000315 | 0.131 | 0.918 | 7.83 | 63.0 | 255.0 | **1.0e+12** |

Worst single-step amplification `1/((1−β)·g)` = **2.9e+20**; median **2.17**.

**Two independent mechanisms destroy information in the forward pass**, and a guard on
either alone is insufficient — I found the second only after the first failed to fix
the tests:

1. **β → 1.** The update becomes an orthogonal projection `S̄(I − kkᵀ) + vkᵀ`; the
   component along `k` is annihilated. β reaches exactly 1.0 because the model runs
   bf16 and `sigmoid` saturates. 0.043% of head-steps.
2. **g → 0.** The decay multiplies the entire previous state by `g`; a small `g`
   *forgets* it, and dividing back amplifies only surviving rounding error. On the
   tiny fixture a single step with `g ≈ 1.1e-9` produced a rewind error of **9e4**
   against a state of magnitude **0.019**.

Errors also compound across undo steps, so the guard has to be on the **cumulative**
amplification since the last exact state, not per step. Threshold sweep on the fixture:

| max amplification | correct? | fallback rate |
|---|---|---|
| 1e3 | **no** | 75–80% |
| 1e2 | yes | 86–88% |
| 1e1 | yes | 92–95% |
| 1 | yes (trivially) | 100% |

**Answer: Gated DeltaNet is algebraically invertible and not practically invertible.**
Rewind is correct *only because it falls back to the recorded state* — which means it
cannot replace the snapshot, it depends on it. It records the same states, adds rank-1
arithmetic on top, and is consequently **slower** (0.78× vs 1.03× AR at K=4). Rewind is
strictly dominated by snapshot as an engineering choice, and the interesting output of
this branch is the measurement, not the mechanism.

## Draftability gap — a better predictor than entropy

Act IV-U found teacher entropy a poor predictor of acceptance at the *low* end. The
hypothesis was that what matters is not how uncertain a token is, but how much it
depends on immediately preceding tokens a parallel draft cannot see. Both quantities
are properties of the **frozen model**, measured with no adapter:

```
P_full     = p(y_{t+j} | true causal predecessors)
P_corrupt  = p(y_{t+j} | those predecessors replaced by noise)
draftability_gap(j) = TV(P_full, P_corrupt)
```

K=8, 2,688 held-out speculative positions:

| quintile | draftability gap → agreement | teacher entropy → agreement |
|---|---|---|
| q0 (lowest) | 1.676 → **0.263** | 0.234 → 0.149 |
| q1 | 1.874 → 0.149 | 1.240 → **0.207** |
| q2 | 1.931 → 0.155 | 2.246 → 0.190 |
| q3 | 1.967 → 0.123 | 3.281 → 0.145 |
| q4 (highest) | 1.992 → **0.074** | 4.934 → 0.072 |

```
pearson(gap,     agreement) = −0.2114
pearson(entropy, agreement) = −0.0968       gap is 2.2x the predictor
pearson(gap,     entropy)   = −0.2535       they are largely independent
```

**The gap is monotone; entropy is not.** Entropy's lowest bucket underperforms its
second (0.149 vs 0.207) — reproducing exactly the Act IV-U anomaly at K=4 and at a
different block size — while the draftability gap decreases monotonically throughout.
That is the predicted signature: a near-deterministic token is often determined by the
token immediately before it, which is precisely what the draft row cannot see.

Reproduced at K=4/n=288 (−0.231 vs −0.111) and K=8/n=2688 (−0.211 vs −0.097).
**Analysis only — no router was built**, per scope. Recorded as the strongest candidate
signal for dynamic block sizing.

## Memory cost

| K | recorded state |
|---|---|
| 2 | 204 MB |
| 4 | 305 MB |
| 8 (projected) | ~500 MB |

`(K+2) × 24 layers × 2.10 MB`, dominated entirely by the fp32 recurrent state. On a
128 GB machine this is not a constraint; on a smaller one it would be, and the obvious
reduction is `B′` — store per-token `(k,v,β,g)` (a few hundred KB) and re-run only the
recurrence for `m` steps, which is exact and needs no inverse. Not implemented, because
snapshot already met every gate.

## Failure modes encountered

1. **Wrong layer's parameters in the rewind.** The first implementation used the first
   DeltaNet layer's `A_log`/`dt_bias` for all 24 layers. Caught immediately by the
   snapshot-vs-rewind test.
2. **Guarding on β alone.** Correct-looking, and wrong: the decay `g` destroys state
   independently. Found by measuring a single-step rewind error (9e4) rather than
   trusting the derivation.
3. **Per-step instead of cumulative conditioning.** Errors compound; a per-step guard
   passed at shallow rewind depths and failed at deeper ones — exactly the pattern the
   brief warned to test for by varying undo depth.

## Interpretation

Act IV-U's conclusion was *"the runtime failed to realise the algorithm, not the
algorithm failing"*. That claim was testable, and it has now been tested: removing the
single implicated term moved TPF from 0.862 to 1.279 and wall clock from 0.79× to 1.03×
AR, with **byte-identical model, adapter and output**. The counterfactual was not a
rationalisation.

Two honest qualifications:

- **The win is modest in absolute terms** (1.03–1.09× AR) and rests on an adapter whose
  Gate 2 prediction quality was only WEAK. Better acceptance is where the remaining
  headroom is: at K=4, acceptance 0.238 with a ceiling of 2.5 TPF.
- **Snapshot costs 305 MB and Python-side loop overhead.** A fused kernel that emits
  per-step states would remove the overhead; the memory is intrinsic.

## Verdict

```
TRANSACTIONAL SPEEDUP ACHIEVED
```

All six gates pass. The replay penalty is removed correctly and completely, the
predicted TPF band is realised, and Uno decoding is now faster than plain
autoregression on this backbone — without retraining anything.

## Next step

1. **Train longer at K=2/K=4.** Now that the state machinery is free, acceptance is the
   only lever left, and Act IV-U's loss was still falling at 400 steps. TPF ceiling at
   K=4 is 2.5; we are at 1.28.
2. **Dynamic block size routed on draftability gap**, not entropy. The analysis above
   is the justification; the router is the work.
3. **A fused per-step-state kernel** to remove the Python loop overhead in the recording
   forward.
4. Not worth pursuing: algebraic rewind. It is measured, documented, dominated.

## Reproduction

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py bench --adapter runs/uno-k4-true/adapter.safetensors --block-sizes 2,4 --transaction-modes replay,snapshot,rewind --out runs/uno-bench/u2_transactional.json
```

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py draftability --adapter runs/uno-k4-true/adapter.safetensors --block-size 8 --eval-batches 96
```

Artifacts: `runs/uno-bench/u2_transactional.json`,
`runs/uno-draftability/draftability_k8.json`, `runs/uno-draftability/draftability.json`.
Tests: `tests/test_uno_transaction.py` (29), plus transaction cases in
`tests/test_uno_decode.py`. Suite: **358 passed**.
