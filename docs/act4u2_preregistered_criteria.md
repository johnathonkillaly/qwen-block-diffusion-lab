# Act IV-U2 success criteria — pre-registered

**Written after correctness was established and before any performance measurement.**
Frozen from here; amendments go in §9 with a date and a reason.

No retraining. The Act IV-U adapter (`runs/uno-k4-true/adapter.safetensors`, r=16,
0.5023% of the backbone) is used unchanged. The only variable is **how rejected
speculative state is discarded**.

---

## 0. What Act IV-U measured, and what U2 must beat

From `docs/act4u_results.md`, same machine, same harness:

| K | acceptance | TPF (replay) | TPF without replay | wall clock vs AR |
|---|---|---|---|---|
| 2 | 0.531 | 0.980 | 1.202 | 0.83× |
| 4 | 0.230 | 0.874 | 1.265 | 0.67× |
| 8 | 0.104 | 0.879 | 1.283 | 0.63× |

The "without replay" column was an *accounting counterfactual*. U2 asks whether it can
be **realised**.

## 1. Gate U2-0 — reference integrity. **Required.**

`transaction_mode="replay"` is unchanged in behaviour and reproduces Act IV-U:
identical committed tokens, identical acceptance counts, identical forward counts.
The backbone digest is unchanged. The replay oracle is never deleted.

## 2. Gate U2-1 — snapshot correctness. **Required.**

For every block size `K ∈ {1,2,4,8}` and **every** accepted-prefix length
`m ∈ {0 … K}`, on non-degenerate fixtures including blocks with repeated token ids:

- committed cache state matches replay (recurrent state, conv state, KV contents and
  offsets), and
- the **greedy continuation** from the committed state is token-identical to replay's
  for ≥ 10 tokens.

Cache-state similarity alone does not satisfy this gate; the continuation must match.

## 3. Gate U2-2 — snapshot removes the replay forward. **Required.**

Instrumented, not inferred from timing: `TransactionStats.replay_forwards == 0` and
`full_forwards == 1` per committed block, for every prefix length. Decoder-level:
`forwards == 1 + 2·cycles` exactly.

## 4. Gate U2-3 — algorithmic realisation

Using the existing adapter, free-running on the held-out prompt suite:

| outcome | condition |
|---|---|
| **PASS** | TPF > **1.0** at some K > 1 |
| **interesting** | TPF ≥ 1.15 |
| **as predicted** | TPF in **1.20–1.28** |

The 1.20–1.28 band is Act IV-U's counterfactual and is **fixed before measurement**.
It will not be adjusted afterwards. Landing below it is a partial realisation and is
reported as such; landing above it means the counterfactual was too conservative and
that also gets said.

## 5. Gate U2-4 — rewind correctness. **Required before any rewind claim.**

`transaction_mode="rewind"` must produce token-identical greedy output to `snapshot`
across the test suite and the benchmark suite. If it does not, no rewind performance
number may be reported at all.

## 6. Gate U2-5 — numerical stability

Rewind must not fail silently. Where the per-head **cumulative** amplification
`∏ 1/((1−β)·g)` exceeds a configured threshold, the head falls back to its recorded
state, and every fallback is counted and reported.

**Pre-registered honesty condition:** if the measured fallback rate exceeds **50%** of
head-steps, rewind is recorded as **dominated by snapshot** — it cannot replace the
snapshot it depends on, and no "rewind works" claim is made regardless of its speed.

## 7. Gate U2-6 — wall clock

Report all three, measured back to back in one process:

```
replay Uno / AR        snapshot Uno / AR        rewind Uno / AR
```

| outcome | condition |
|---|---|
| **PASS** | snapshot mode is faster in wall clock than replay mode |
| **full success** | snapshot mode is also faster than plain AR |

**U2 succeeds scientifically if it removes the replay penalty correctly**, even if Uno
remains slower than AR. Those are separate claims and are reported separately.

## 8. Kill criteria

1. Snapshot output differs from replay anywhere → the transaction is wrong. Stop.
2. `replay_forwards > 0` in snapshot mode → Gate U2-2 not met; the abstraction leaks.
3. The recording forward changes the model's logits beyond fp tolerance → the
   unrolled recurrence is not faithful; every result is void.
4. Snapshot memory exceeds what the machine can hold at K=8 → report the ceiling
   rather than quietly reducing K.
5. Rewind produces a *silent* wrong answer (no fallback triggered, wrong tokens) →
   the conditioning guard is mis-specified.

## 9. Interpretation rules

- **Removing the replay is an engineering result, not a scientific one about Uno.**
  The adapter is unchanged; acceptance rates must be *identical* to Act IV-U. If
  acceptance moves at all, something is wrong.
- **Snapshot memory is a real cost** and is reported in bytes, not hidden.
- **A rewind that falls back most of the time is a snapshot.** See §6.
- Wall-clock ratios are the quantity; absolute tok/s on a shared machine is not.

## 10. Amendments

*(none)*
