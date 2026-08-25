# The Act III objective

Independent implementation of the methodological ideas in **FLARE: Diffusion for
Hybrid Language Model** ([arXiv:2606.01774v2](https://arxiv.org/abs/2606.01774)),
written from the paper's mathematical specification against this repository's own
harness. No source was taken from the FLARE reference implementation. See
[THIRD_PARTY.md](../THIRD_PARTY.md).

## Notation

| symbol | meaning | Act III value |
|---|---|---|
| `B` | batch | 2 |
| `P` | clean prefix length | 64 |
| `C` | canvas (diffusion block) length | 128 |
| `T = P + C` | packed sequence length | 192 |
| `V` | model vocabulary | 248,320 |
| `t` | requested mask fraction | `~ U(0.05, 1.0)` |
| `M` | masked position set within the canvas | `\|M\| = round(t·C)` |
| `M^c` | its complement in the canvas | `M ∪ M^c = canvas`, `M ∩ M^c = ∅` |
| `[MASK]` | absorbing token id | 248,319 |

## The loss

FLARE Eq. (4)–(5), specialised to our single-block canvas layout:

```
L_AR   = mean over l in [0, T-1)   of  -log p(x_{l+1} | x_{<=l})

L_diff = ( |M|  · mean_{l in M}   -log p(x_l | x~_M,  x_{<b})
         + |M^c|· mean_{l in M^c} -log p(x_l | x~_Mc, x_{<b}) ) / (|M| + |M^c|)

L_total = ar_weight · L_AR + lambda_diff · L_diff          (both 1.0 by default)
```

`x~_M` is the canvas with positions in `M` replaced by `[MASK]`; `x~_Mc` is the canvas
with positions in `M^c` replaced instead. Because `M` and `M^c` partition the canvas,
the two noisy views together supervise **every canvas position exactly once per
step** — FLARE's "every token contributes one AR signal and one diffusion signal at
unit weight".

### Normalisation — a deliberate deviation

FLARE writes unweighted **sums**. They pack `L` tokens and partition *every* block, so
their AR term and diffusion term cover the same `L` tokens: a 1:1 token balance falls
out naturally.

Our layout has one canvas block at the end of a prefix, so a raw sum would give `T-1 =
191` AR targets against `C = 128` diffusion targets — a 1.49:1 imbalance that grows
with the prefix. We therefore take a **per-token mean of each term**, which restores
exactly the 1:1 balance FLARE achieves structurally. `lambda_diff` is then the explicit
knob rather than an accident of geometry.

`ar_weight = 0.0` recovers the Act I/II pure-diffusion objective and is retained as an
ablation.

## Readout alignment

Unchanged from Act I and stated again because it is the single most important
convention in the repository:

```
AR term        : logits[:, l]  ->  x[l+1]      (shifted)
diffusion term : logits[:, i]  ->  x0[i]       (NOT shifted)
```

The two terms therefore ask the *same weights* for two different readouts of the same
position. That tension is the whole experiment. In Acts I–II, with no AR term, the
model resolved it by abandoning AR entirely (reference perplexity 8.06 → 115.87). The
AR term exists to stop that.

## Forward passes per step

| # | stream | input | mask | loss |
|---|---|---|---|---|
| 1 | clean | `[B, T]` all-clean ids | causal | `L_AR` |
| 2 | noisy `M` | `[B, P]` prefix + `[B, C]` canvas with `M` masked | causal prefix, bidirectional canvas | `L_diff` on `M` |
| 3 | noisy `M^c` | same prefix + canvas with `M^c` masked | same | `L_diff` on `M^c` |

Pass 3 is skipped when `complementary_views: false`, which halves diffusion cost and
supervises only `M`. Over many steps that is statistically similar but it is not
FLARE's construction, so it is off the default path.

### Tensor shapes (measured, `qdif act3-check`)

```
clean_stream_input           [2, 192]
clean_stream_logits          [2, 192, 248320]
ar_targets                   [2, 191]
noisy_stream_canvas_input    [2, 128]
noisy_stream_logits          [2, 128, 248320]
prefix                       [2, 64]
mask_set                     [2, 128]
forwards_per_step            3
```

## Corruption: absorbing state, not uniform random tokens

Acts I–II used uniform random-token replacement. Act I experiment 001 showed why that
is hard: corrupted positions are **unidentifiable**, so the model must detect *and*
repair, and echoing the input scores `1-t` for free. Act II Phase 3 then collapsed to
a canvas-ignoring regime.

Act III uses absorbing-state masking, where corrupted positions are identifiable by
construction and copying a `[MASK]` scores zero. `corruption: uniform` is kept as an
ablation and is not deleted.

**Mask token choice.** Qwen3.5's model vocabulary (248,320) is larger than its
tokenizer vocabulary (248,044). Every embedding row from ~248,070 upward shares the
norm 0.445587 — the signature of rows initialised and never trained. We use the last
row, **248,319**, so the absorbing token (a) can never be produced by the tokenizer and
so cannot collide with real text, and (b) carries no pretrained semantics to unlearn.
`resolve_mask_token_id` asserts the model vocabulary is genuinely larger and fails
loudly otherwise.

Masking uses an **exact count** `round(t·C)` rather than per-position Bernoulli(t), so
the realised mask fraction matches the requested `t` closely and the noise-level
diagnostics stay interpretable. Both are logged (`requested_mask_fraction`,
`actual_masked_fraction`, `num_masked`, `num_visible`).

## Gated DeltaNet state scheduling

FLARE Eq. (6), implemented in `deltanet.py` as `mode: flare`:

```
S~_0   = S_prefix                      (clean boundary state, from the causal history)
S~_i   = GDN(S~_{i-1}, x~_i)           forward through the noisy block only
o~_i   = S~_C^T q~_i / sqrt(d_k)       EVERY position reads the completed block-end state
```

Our forward pass runs straight through `[clean prefix | noisy canvas]`, so the state
entering the canvas *is* the clean boundary state and the state leaving it *is*
`S~_C` — FLARE's construction for a single block, with no extra bookkeeping. `mlx_lm`'s
`gated_delta_update` already returns the final state, so the readout is one extra
matmul and required **no kernel work**; FLARE's Route I/II fused kernels are an
optimisation we do not need at this scale.

No reverse recurrence is used. The Act II bidirectional implementation is retained as
a historical ablation (`mode: fusion`) and is not on the Act III path.

Verified in `qdif act3-check`: perturbing the last canvas token moves earlier canvas
positions (max Δ 1.31) while the clean prefix stays **exactly** fixed (Δ = 0.0).

## Full-attention visibility

| query | may attend to |
|---|---|
| prefix position | prefix positions ≤ itself (causal) |
| canvas position | the whole prefix, and every canvas position (bidirectional within the block) |
| prefix position | **never** a canvas position |

Verified in `act3-check` item 5 and by the Act II mask tests, which are reused
unchanged.

## Deviations from FLARE, collected

| # | FLARE | Act III here | why |
|---|---|---|---|
| 1 | unweighted sums over a token-balanced packing | per-token means of each term | restores the same 1:1 balance under our single-block layout |
| 2 | `K = L/B` blocks per sequence, `B = 4` | one block of 128 per sequence | far simpler; costs supervision density per forward |
| 3 | full-weight / large-scale conversion | BF16 base + BF16 LoRA r=32 (63.1M trainable, 1.48%) | single 128 GB Apple machine |
| 4 | their transfer data mix | WikiText-103 | reproducible and permissively licensed; FLARE reports data mix dominates, so this is a real limitation and we say so |
| 5 | "logit shift to the noisy-stream diffusion terms" | not implemented | the paper does not give the formula in the sections available to us; recorded as a known gap rather than guessed at |
| 6 | Route I/II fused GDN kernels | none needed | `gated_delta_update` returns the block-end state directly |
| 7 | AR-Trust / Diffusion-Trust serving | not implemented | out of scope |

Deviation 5 is the one most likely to matter and is flagged in
[FINDINGS.md](../FINDINGS.md) as a reason any failure here should not be attributed to
the method without qualification.
