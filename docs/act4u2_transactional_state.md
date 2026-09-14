# Qwen3.5-4B decode state, and whether it can be rewound

Everything here is read out of the **actual MLX implementation this repo loads**
(`mlx_lm/models/qwen3_5.py`, `mlx_lm/models/gated_delta.py`, `mlx_lm/models/cache.py`,
with `unsloth_zoo.gated_delta_vjp` patched in), and verified numerically. The Hugging
Face implementation is **not** assumed to match, and was not consulted for the
equations.

---

## 1. Architecture constants (from the checkpoint config, not from memory)

| | |
|---|---|
| layers | 32 — **8 full attention** `[3,7,11,15,19,23,27,31]`, **24 Gated DeltaNet** |
| hidden | 2560 |
| attention | 16 q heads, 4 kv heads, head_dim 256 |
| DeltaNet | `Hk=16` key heads, `Hv=32` value heads, `Dk=128`, `Dv=128` |
| | `key_dim = 2048`, `value_dim = 4096`, `conv_dim = 2·2048+4096 = 8192` |
| | conv kernel 4 → 3 rows of history |
| vocab | 248,320, tied embeddings |

`Hv/Hk = 2`, so q and k are `mx.repeat`-ed by 2 before the recurrence.

## 2. Every mutable decode state

| state | shape | dtype | layers | bytes/layer | bytes total | update rule | truncatable? | rewindable? |
|---|---|---|---|---|---|---|---|---|
| KV cache keys | `[B,4,T,256]` | bf16 | 8 attn | 2 KB/token | 16 KB/token | append at `offset` | **yes** — `offset` is a pointer | yes |
| KV cache values | `[B,4,T,256]` | bf16 | 8 attn | 2 KB/token | 16 KB/token | append at `offset` | **yes** | yes |
| `KVCache.offset` | scalar | int | 8 attn | — | — | `+= S` | yes | yes |
| DeltaNet conv state `cache[0]` | `[B,3,8192]` | bf16 | 24 | 48 KB | 1.15 MB | last 3 rows of `[state ; qkv]` | **yes** — it is a sliding window | yes, from saved `qkv` |
| DeltaNet recurrent state `cache[1]` | `[B,32,128,128]` | **fp32** | 24 | **2.10 MB** | **50.3 MB** | see §3 | **no** | **conditionally — see §5** |
| `ArraysCache.lengths` / `left_padding` | `[B]` | int | 24 | — | — | `advance(N)` | yes | yes (both `None` for us) |

Rotary position is computed from `cache.offset` at call time and holds no separate
mutable state. There is no other per-step state.

**Total speculative footprint per token:** 16 KB (KV) + negligible. **One full
recurrent snapshot: 50.3 MB.** For `K=8` that is 9 snapshots ≈ 453 MB — entirely
affordable in 128 GB, which is what makes Approach B viable at all.

## 3. The exact recurrence

From `gated_delta_update` → `_gated_delta_step_ops` / the Metal kernel, which agree.
Per step `t`, per value head, with state `S ∈ R^{Dv×Dk}` (**rows indexed by `Dv`,
columns by `Dk`**):

```
β_t = sigmoid(b_t)                                   b_t = in_proj_b(x_t)     [B,T,Hv]
g_t = exp(-exp(A_log) · softplus(a_t + dt_bias))     a_t = in_proj_a(x_t)     [B,T,Hv]

S̄_t    = g_t · S_{t-1}                       decay
kv_mem = S̄_t k_t                             [Dv]
δ_t    = (v_t − kv_mem) · β_t                [Dv]
S_t    = S̄_t + δ_t k_tᵀ                      rank-1 update
y_t    = S_t q_t                             [Dv]
```

`k` is **exactly unit-norm**: the layer computes `k = Dk^(-1/2) · rms_norm(k_raw)`,
and `‖rms_norm(x)‖₂ = √Dk`, so `‖k‖₂ = 1`. Verified numerically: `‖k‖ = 1.0`.

### Closed form, and a correction to the assumed form

Substituting `kv_mem`:

```
S_t = S̄_t (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```

> **The projector multiplies on the RIGHT, not the left.** The form assumed in the
> Act IV-U2 brief, `S_t = (I − βkkᵀ)S̄_t + βkvᵀ`, does not describe this
> implementation. `S` is `[Dv, Dk]` and `k` contracts along `Dk`, the trailing axis.
> Measured on random inputs against `_gated_delta_step_ops`:
>
> | candidate | max abs error |
> |---|---|
> | right-multiplied (above) | **2.4e-07** — correct, fp32 noise |
> | left-multiplied (brief) | 1.74 — wrong |

## 4. The inverse

`(I − βkkᵀ)` has eigenvalue `(1−β)` along `k` and `1` on its orthogonal complement, so
with `‖k‖ = 1` Sherman–Morrison is exact:

```
(I − β k kᵀ)^{-1} = I + β/(1−β) · k kᵀ
```

Substituting and simplifying (`1 + β/(1−β) = 1/(1−β)`) gives a **rank-1 undo** that
needs only `k, v, β, g`:

```
S̄_t     = S_t + (β_t/(1−β_t)) · (S_t k_t − v_t) k_tᵀ
S_{t-1} = S̄_t / g_t
```

Verified numerically: recovers `S̄` and `S_{t-1}` to **4.8e-07** (fp32 noise).

So the algebra works. The arithmetic is another matter.

## 5. Is it *numerically* reversible in Qwen3.5's operating regime? **No.**

Measured over 4 × 256 held-out WikiText tokens through all 24 DeltaNet layers
(3.1 M head-steps):

| | min | p10 | median | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|
| `β` | 0.000315 | 0.1157 | 0.4785 | 0.8867 | 0.9844 | 0.9961 | **1.00000000** |
| `1−β` | **0.000000** | 0.1133 | 0.5215 | 0.8843 | 0.9756 | 0.9931 | 0.99969 |
| decay `g` | 0.000000 | 0.4039 | 0.9821 | 0.9998 | 0.99999 | 0.999998 | 1.0000000 |
| `β/(1−β)` | 0.000315 | 0.1309 | 0.9176 | 7.83 | 63.0 | 255.0 | **1.0e+12** |

- `frac(β > 0.99)` = **0.62%**
- `frac(β > 0.999)` = **0.043%**
- `frac(1−β < 1e-4)` = **0.043%**
- worst single-step amplification `1/((1−β)g)` = **2.9e+20**
- median single-step amplification = **2.17**

**`β` reaches exactly 1.0.** The model runs bf16, `β = sigmoid(b)` is computed and
stored in bf16, and bf16 rounds `sigmoid(x)` to exactly `1.0` for `x ≳ 8`. When
`β = 1` the update is

```
S_t = S̄_t (I − k kᵀ) + v kᵀ
```

which is an **orthogonal projection**: the component of `S̄_t` along `k` is *destroyed*,
not merely scaled. No inverse exists — this is rank deficiency, not ill-conditioning,
and no amount of fp32 arithmetic in the *inverse* recovers information the *forward*
already discarded. Recomputing `β` in fp32 does not help either: it would invert an
operator different from the one actually applied.

### How often does this bite a real block?

A `K=4` verify pass touches `5 tokens × 32 heads × 24 layers = 3,840` head-steps. At
`P(1−β < 1e-4) = 4.3e-4`:

```
P(at least one near-singular head-step in a block) ≈ 1 − (1 − 4.3e-4)^3840 ≈ 81%
```

So **most blocks contain at least one head-step that cannot be inverted.** Exact
rewind of a whole block is not achievable; a per-head fallback is mandatory, which is
what pre-registered Gate U2-5 anticipates.

**Consequence for the plan:** Approach C is demoted from "the research solution" to "a
per-head opportunistic optimisation with a mandatory snapshot fallback", and the
headline mechanism must be Approach B. This was decided from measurement *before*
implementing either, and is recorded here rather than discovered later.

## 6. What this implies for each approach

| | mechanism | extra full forwards | exactness | verdict from §5 |
|---|---|---|---|---|
| **A replay** | re-run accepted prefix through the whole model | **1** | exact | the oracle; keep forever |
| **B snapshot** | unroll the recurrence inside the *existing* verify forward, keep `S` at each token boundary | **0** | exact | **primary** |
| **B′ factor-replay** | keep `(k,v,β,g)` per token, re-run only the recurrence for `m` steps | **0** | exact | cheaper memory, same guarantee |
| **C rewind** | algebraic undo from `S_K` backwards | **0** | approximate, **undefined at β=1** | opportunistic only |

Note B and B′ both need the same thing: the per-token quantities from inside the
verify forward. Getting those requires unrolling the DeltaNet recurrence over `T`,
which the fused kernel does not expose. Crucially the *expensive* parts — the
projections, the depthwise conv, attention, and every MLP — still run **once** for the
whole block. Only the recurrence, which is a small fraction of layer cost, is unrolled.

## 7. Conv state needs no algebra

`cache[0]` is the last 3 rows of `[conv_state ; qkv]`. To commit a prefix of `m`, take
the last 3 rows of `[conv_state_before ; qkv[0:m]]` — a slice of tensors we already
have. Exact, O(1), no inverse required.

## 8. KV cache needs no algebra either

`KVCache.update_and_fetch` writes into `keys[..., prev:offset, :]` and returns
`keys[..., :offset, :]`. Committing a prefix of `m` is `offset = base + m`. Stale
values beyond `offset` are never read and are overwritten by the next write. Verified
by test at every rejection length rather than assumed.
