# Act IV-U6 — Decouple Draft Width from Verify Width: design

Written before any U6 measurement. Pre-registered criteria:
[`act4u6_preregistered_criteria.md`](act4u6_preregistered_criteria.md).

---

## 1. Question

The Act IV-U/Act IV-S decoder couples two widths that have different costs:

```
draft width   K  =  verify width  K
```

> Can the diffusion drafter propose a wider block while the frozen target verifies
> only a narrower window per forward, and does `K_draft ≠ K_verify` raise end-to-end
> committed tokens per second above the incumbent K=4 decoder?

This is **inference scheduling on a frozen checkpoint**. Nothing is trained. It is a
different hypothesis from adaptive K (Act IV-S gate 5, failed): adaptive K varied the
single coupled width, while U6 asks whether the coupling itself is the mistake.

## 2. What is inherited and not re-derived

| fact | source |
|---|---|
| K=4 is the static frontier: 1.245× native AR (1.230× loop-free), identical to native greedy except at bf16 ties | [`RESULTS_SPECULATIVE.md`](RESULTS_SPECULATIVE.md) §2–3 |
| The prefix-survival curve does not depend on decode width: about 0.60, 0.31, 0.14, 0.05 … | same, §3 |
| A cycle is one draft forward plus one verify forward; the K=1 control, with no speculation, is 0.814× AR | same, §3 |
| Snapshot transactions commit a prefix with no extra forward | [`act4u2_results.md`](act4u2_results.md) |
| The drafter's own confidence predicts survival, but only after the draft forward | [`RESULTS_SPECULATIVE.md`](RESULTS_SPECULATIVE.md) §8 |
| A training launch is one draw; a fixed adapter evaluates deterministically | [`act4u5_results.md`](act4u5_results.md) |

## 3. The frozen drafter

All U6 comparisons use one set of weights, which removes training-launch variance from
the experiment entirely.

| | |
|---|---|
| drafter | `runs/u4b1/step-16000/adapter.safetensors`, the Act IV-S drafter |
| adapter SHA-256 | `8200c339d3d47363a3920fc4aca58f3535fc8bf75be431e35204f374700a4f42` (`results/checkpoint_manifest.json`) |
| adapter | r=16 token-conditional LoRA, 21,233,664 trainable parameters, trained at block size 8 |
| target | `unsloth/Qwen3.5-4B-Base`, snapshot `61541fe4aed37b6a1e615edd5cbf5a3f660312b1`, bfloat16, frozen |
| backbone digest | `7cd0f5a9bce55681729d390a530630b6fc8ae8c5e683ca4f417532895759e097` |
| software | MLX 0.32.1, mlx_lm 0.31.3, Python 3.13.12, macOS 27.0 arm64 |
| hardware | Apple M4 Max, 128 GB unified memory |
| code | branch `rprm-diffusion-stage1`, starting from `7172917` |

Every U6 script checks the adapter's SHA-256 against the manifest before loading it, and
records the backbone digest before and after each session.

## 4. Semantics, fixed before implementation

Terms:
* **`D`** is the draft width. One draft forward over `[seed, x₁ … x_{D−1}]` gives the
  proposal `p₁ … p_D`, where `p₁` comes from the adapter-off seed row and is the target's
  own next token.
* **`V`** is the verify width: the number of proposal tokens presented to the target in a
  verify forward.

The incumbent K=4 decoder is `D = V = 4`. Its verify forward is `[seed, p₁ … p₄]`, five
tokens wide.

### Variant 1: truncate, `T(D, V)`

Draft `D` positions and verify `[seed, p₁ … p_V]` exactly as the coupled decoder would.
The unverified suffix `p_{V+1} … p_D` is discarded.

This only helps if a wider canvas changes the proposals at slots 1…V. **The draft pass
is the pretrained causal forward** (`src/qdif/uno/model.py`: no custom masks, no
bidirectional recurrence). Position `i` therefore reads positions ≤ `i` only, and a wider
canvas can change slots 1…V only through:
* a different noise realisation at those slots, because MLX draws of different shapes are
  not sub-arrays of each other;
* width-dependent bf16 reduction order.

Neither is a mechanism. **Prediction: Variant 1 has no reason to exist.** The diagnostic
(§6) tests this before any wall-clock work.

### Variants 2 and 3: staged verification with bounded continuation, `S(D, V)`

Draft once, verify in stages, and never re-draft inside a cycle.

```
stage 1   verify [seed, p₁ … p_V]                          (as the coupled decoder)
          commit accepted prefix + correction/lookahead L
          continue only if every slot was accepted AND L == p_{V+1}
stage s   verify [c, p_{j+1} … p_{j+V}]  on the committed cache
          (c = the last committed token, which equals draft token p_j)
          commit accepted prefix + correction/lookahead
          continue only if every slot was accepted AND the lookahead equals the
          next draft token
stop      when a stage rejects, the lookahead disagrees, the draft is exhausted,
          the token budget is reached, or a stop token appears
```

Variant 3's rule, never reuse suffix tokens after a predecessor was rejected, is exactly
this continuation condition, so Variants 2 and 3 are one decoder.

**Correctness does not depend on the continuation rule.** Every committed token is the
target's own argmax for the forward that verified it, exactly as in the coupled decoder
and the n-gram control. The rule bounds speculative reuse and affects acceptance, not
correctness.

Each stage is its own snapshot transaction on the committed cache (`begin_transaction`
→ `run_block` → `commit_prefix`), so a rejected token can never enter target state. No
stage needs a replay forward. Stage ≥ 2 is the Act IV-S n-gram verify path with draft
tokens as the guesses; that path is already tested lossless.

With `D = V`, both variants reduce **exactly** to the incumbent decoder: the same tokens,
the same per-cycle acceptance, the same forward count. A test asserts this.

### Arms

| arm | name | D | V | variant |
|---|---|---|---|---|
| A | native AR | — | — | — |
| B | incumbent (`decode.uno_greedy_generate`, K=4) | 4 | 4 | coupled |
| B_dec | incumbent through the new decoder | 4 | 4 | harness-equivalence control |
| C | `staged_d8v4` | 8 | 4 | staged |
| D | `staged_d8v2` | 8 | 2 | staged |
| E | `staged_d6v4` | 6 | 4 | staged |
| F | `staged_d6v3` | 6 | 3 | staged |
| T* | `trunc_d8v4`, `trunc_d8v2`, `trunc_d6v4`, `trunc_d6v3` | | | truncate; **only if gate U6-1 keeps Variant 1 alive** |

B_dec exists so that a staged loss cannot be caused by the new code path being slower
than the incumbent's.

## 5. What the cost curve says in advance

From Act IV-S's measured medians on this adapter:
* draft forward: 23.8 / 24.7 / 25.9 ms at width 2 / 4 / 8;
* verify forward: 21.8 / 24.0 / 26.8 ms at width 3 / 5 / 9;
* commit plus overhead: ~1.1 ms per cycle.

**A verify forward has a large fixed cost (~20 ms) and a small marginal cost
(~0.7–1.1 ms per token).** Verifying 2 twice therefore costs almost twice verifying 4
once. A second stage has to return close to a whole cycle's worth of tokens to pay, and
it only happens when stage 1 was fully accepted.

With the K=8 survival curve (0.608, 0.313, 0.137, 0.054, 0.023, 0.014, 0.010), the
expected tokens per ms, relative to B, are:

| arm | predicted tokens / cycle | predicted ms / cycle | predicted vs B |
|---|---|---|---|
| B (4, 4) | 3.069 | 49.8 | — |
| S(6, 4) | 3.135 | 51.6 | −1.3% |
| S(8, 4) | 3.159 | 52.3 | −1.9% |
| S(6, 3) | 3.135 | 52.4 | −2.9% |
| S(8, 2) | 3.159 | 56.3 | −8.9% |
| T(6, 4) / T(8, 4) | 3.058 | 50.4 / 51.0 | −1.5% / −2.6% |
| T(6, 3) / T(8, 2) | 2.921 / 2.608 | 49.3 / 48.8 | −3.9% / −13% |

**Prediction: no decoupled arm beats B.** Staged arms commit slightly more tokens per
cycle and lose on wall clock because the extra verify forwards cost more than they
return. This is a prediction only; the gates in the criteria are decided by measurement,
and U6-2 measures the cost curve directly on the M4 Max rather than trusting these
medians.

## 6. Phases, in order

1. **Wide-draft diagnostic** (`scripts/u6_wide_draft_diagnostic.py`, gate U6-1, no wall
   clock). 27 suite prompts plus 229 held-out WikiText-103 windows. For each context:
   * draft at D = 4, 6, 8 and 16 and compare slots 1–4 with D=4, under **decoder
     noise** (as the decoder draws it) and **aligned noise** (one width-16 draw,
     prefix-shared);
   * add a **re-draw control**: D=4 with an independent noise key.

   Recorded: argmax identity per slot, logit differences, and accepted prefix within
   four slots against the target's greedy continuation.
2. **Cost curve** (`scripts/u6_cost_curve.py`, gate U6-2). Verify-forward latency at
   widths 2–9, 13 and 17, and draft-forward latency at 2, 4, 6, 8, 12 and 16, measured
   through the decoder's own transaction path. Widths are interleaved in a shuffled order
   and medians reported, and the fixed plus marginal cost is fitted.
3. **Calibration** (`scripts/u6_bench.py calibrate`). A, B and a second identical B,
   interleaved on the full suite, to measure within-session timing noise and fix the
   practical floor by the formula in the criteria.
4. **Pilot** (`u6_bench.py pilot`, gate U6-3). All arms, one repeat. Kill rule in the
   criteria.
5. **Decisive run** (`u6_bench.py decisive`, gates U6-4 and U6-5). All arms, three
   repeats.
6. **Report** (`scripts/u6_report.py`): gates, verdict and plots.

## 7. Harness

The Act IV-S speculative suite, unchanged: the 27 prompts of `spec_decode.SPEC_SUITE`, 128
generated tokens, greedy decoding and snapshot transactions. Every arm uses draft-noise
stream 20260905, keyed per cycle. Arms are interleaved inside each prompt inside each
repeat, with one discarded warm-up pass, and results are reported as medians, never
best-of-N. Every row carries its AR continuation's loop fraction, so every result is also
reported on the loop-free subset.

Every divergence from native AR is audited in bf16 ULPs, as in Act IV-S.

**Cost decomposition recorded per row and per cycle:**
* draft latency;
* per-stage verify latency and width;
* verifier forwards, drafter forwards and target tokens processed;
* commit time and overhead;
* committed tokens per stage and per cycle;
* rejected draft tokens (offered but rejected) and unused draft tokens (never verified);
* cycle latency.

## 8. Out of scope

* **Training (U6-B).** If U6-A finds a mechanism that targeted training could help, U6-B is
  proposed afterwards and not run automatically.
* **Sampling-preserving verification.** Greedy only.
* **New prompts, K > 16, or more arms before these are profiled.**
* **Reopening U5 or adaptive K.**
