# Act IV-S — speculative decoding: results

Design and pre-registered predictions: [`SPECULATIVE_DESIGN.md`](SPECULATIVE_DESIGN.md).
Recovered prior state: [`STATE_RECOVERY.md`](STATE_RECOVERY.md).
Machine-readable: `results/speculative/` (6 JSON, 26 CSV, 5 divergence audits).
Plots: `plots/speculative/` (12).

Target **Qwen3.5-4B-Base**, bf16, frozen. Drafter **`runs/u4b1/step-16000`** (r=16 LoRA,
21.2 M trainable, 0.50%). Greedy throughout. Apple M4 Max, 128 GB, MLX 0.32.1.
27 prompts × 9 categories × 3 repeats, arms interleaved within each prompt.

---

## Verdict

```
K=4 STANDS. ADAPTIVE K FAILS. TWO OF THE APPARENT WINS WERE ARTIFACTS.
```

The decoder is correct and it is faster. The new work's main contribution is **negative
and methodological**: three separate effects that look like results are not, and two of
them would have inflated the headline.

| gate | condition | result |
|---|---|---|
| **0** baseline | native AR stable | **PASS** — 48.0–48.9 tok/s, sd 0.29–0.91 across sessions |
| **1** correctness | speculative == native greedy | **PASS modulo bf16 ties** — 759/759 divergences within one ULP, 0 unexplained |
| **2** useful drafting | mean accepted prefix > 1 | **PASS** — 1.069 at K=4 |
| **3** actual speed | beats native AR end-to-end | **PASS** — 1.245× |
| **4** meaningful speed | ≥1.10 / 1.25 / 1.50 / 2.00 | **"real but modest"** — 1.245× full suite, 1.230× clean. Misses the 1.25× "useful" band |
| **5** adaptive value | adaptive beats best fixed K | **FAIL** — best adaptive 1.230× vs fixed K=4 1.238× |

---

## 1. Gate 0 — the native baseline

27 prompts × 3 repeats, interleaved with the speculative arm.

| generated tokens | AR tok/s (median) | sd | TTFT |
|---|---|---|---|
| 128 | 48.67 | 0.36 | 47.4 ms |
| 256 | 48.85 | 0.91 | 47.4 ms |
| 512 | 48.02 | 0.81 | 47.3 ms |

Reproduced at 48.46 / 47.34 / 47.53 tok/s in three later sessions hours apart. Act
IV-U3/U4 measured 48.35 / 48.93 / 49.45 on the same machine. **Stable to ~±1.5% across a
week.** Peak memory 8.59 GB (AR), 8.89 GB (K=4).

## 2. Gate 1 — correctness, and a limit worth naming

Byte-identity **fails**, and the reason is arithmetic rather than logic.

Across all five experiments, **759 positions** diverged from native AR. Every one was
re-examined against the target's own full-context logits:

| | |
|---|---|
| divergences audited | **759** |
| exact ties (gap = 0.0 in bf16) | **492** |
| within one bf16 ULP | **759 / 759** |
| **beyond one ULP (i.e. real defects)** | **0** |
| max contested gap | **1.0 ULP** |

The first case found: token 2 of `def fibonacci(n): …`, where `' -'` and `'-'` both carry
logit **24.875** — gap exactly `0.000000e+00`. A width-1 AR forward and a width-(K+1)
verify forward are different reduction orders over the same maths; at a tie they can
pick differently and **neither is wrong**.

The base rate is not negligible: **1.24% of decoded positions (19/1536)** have an exact
bf16 top-2 tie, so a 128-token generation meets ~1.6 of them.

**This refines, and does not contradict, Act IV-U4's gate U4-B4.** U4 verified
losslessness in the *cacheless reference regime*, where both paths use the same forward
width and ties break identically. That guarantee was real and narrower than it sounded.
The correct statement is:

> Speculative decoding commits exactly the tokens the verifier's own forward produces.
> It is byte-identical to native AR **except at positions where the target has no
> strict preference in its own dtype**, where the two differ by tie-breaking order.

Anyone wanting true byte-identity must make the target's argmax width-invariant (compute
logits in fp32, or break ties by token id), not fix the decoder.

## 3. Gates 2–4 — the K sweep

128 generated tokens, 27 prompts, 3 repeats, `runs/u4b1/step-16000`.

| arm | tok/s | sd | ×AR | TPF | accept | mean prefix | cycle ms |
|---|---|---|---|---|---|---|---|
| native AR | 48.46 | 0.87 | 1.000 | 1.000 | — | — | — |
| K=1 *(control)* | 39.44 | 0.69 | **0.814** | 0.992 | 0.000 | 0.000 | 50.0 |
| K=2 | 54.69 | 3.03 | 1.129 | 1.293 | 0.633 | 0.620 | 46.7 |
| **K=4** | **60.32** | 8.16 | **1.245** | 1.506 | 0.349 | 1.069 | 49.8 |
| K=8 | 54.68 | 13.62 | 1.128 | 1.471 | 0.143 | 1.157 | 53.8 |
| K=16 | 25.59 | 7.97 | 0.528 | 1.580 | 0.083 | 1.274 | 124.3 |

Clean subset (AR continuation not looping, 48 of 81 rows): AR 48.44, **K=4 = 59.56 =
1.230×**. The ranking and the magnitude both survive.

**The K=1 control is the most informative row.** With zero speculative slots the cycle
still costs two forwards, and throughput falls to **0.814× AR**. That is the price of
the draft→verify structure itself, and it is what every speculative slot has to earn
back before anything is gained.

**K=16 shows the algorithmic metric and the wall clock cleanly diverging.** It has the
highest TPF of any arm (1.580 — it commits more tokens per forward than K=4) and is the
*slowest* decoder measured (0.528×). Tokens per forward is not speed.

This reproduces Act IV-U4's `K4 IS THE PRACTICAL FRONTIER` on a different prompt suite,
a different generation length and a rebuilt harness. U4 measured 1.195× at 48 tokens;
1.245× here at 128 tokens is the same result with prefill better amortised.

### Acceptance-prefix distribution (brief §16)

Mean acceptance hides the shape, and the shape is bimodal:

| K | P(0) | P(1) | P(2) | P(3) | P(4) | P(5) | P(6) | P(7) |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.380 | 0.620 | — | — | — | — | — | — |
| **4** | **0.402** | 0.274 | 0.176 | 0.147 | — | — | — | — |
| 8 | 0.392 | 0.295 | 0.176 | 0.082 | 0.032 | 0.009 | 0.004 | 0.010 |

**40% of cycles accept nothing at all.** The decoder does not steadily accept ~1 token;
it alternates between accepting 0 and accepting 2–3. That matters for latency jitter and
it is why the tok/s sd is 8.16 at K=4 against 0.87 for AR.

### Horizon survival — `P(accepted ≥ j)`

| K | j=1 | j=2 | j=3 | j=4 | j=5 | j=6 | j=7 |
|---|---|---|---|---|---|---|---|
| 2 | 0.620 | — | — | — | — | — | — |
| 4 | 0.598 | 0.324 | 0.147 | — | — | — | — |
| 8 | 0.608 | 0.313 | 0.137 | 0.054 | 0.023 | 0.014 | 0.010 |
| 16 | 0.602 | 0.329 | 0.159 | 0.078 | 0.034 | 0.018 | 0.010 |

The curve is **independent of K** — the drafter's reach is a property of the adapter,
not of how many slots it is offered. It is ~0.60 at j=1, halves each step, and is under
5% by j=4. At a measured marginal slot price of ~1.5 ms against a ~50 ms cycle, a slot
must return ≳0.09 tokens to pay; slot 4 returns 0.054. **Three slots pay. The fourth
does not.** That is K=4, arrived at independently of U4.

## 4. Prompt-class stratification (brief §13)

Mean accepted prefix and end-to-end speedup at K=4, with the loop fraction of the AR
reference — the column that makes the table readable.

| category | K=2 | **K=4** | K=8 | ×AR at K=4 | AR loop fraction |
|---|---|---|---|---|---|
| `high_entropy` | 0.681 | **1.478** | 1.536 | **1.418** | **0.64** |
| `structured` | 0.713 | **1.252** | 1.342 | 1.344 | 0.07 |
| `math` | 0.626 | 1.187 | 1.400 | 1.316 | 0.17 |
| `reasoning` | 0.651 | 1.155 | 1.244 | 1.327 | 0.01 |
| `factual` | 0.630 | 1.146 | 1.297 | 1.235 | 0.04 |
| `code` | 0.597 | 1.023 | 1.047 | 1.203 | 0.31 |
| `prose` | 0.622 | 0.939 | 0.825 | 1.130 | 0.07 |
| `dialogue` | 0.573 | 0.917 | 1.155 | 1.310 | 0.30 |
| `technical` | 0.500 | **0.662** | 0.743 | 1.077 | 0.00 |

**The pre-registered diagnostic fired.** `SPECULATIVE_DESIGN.md` §2.1 said: if
`high_entropy` does not come out markedly *worse* than `code` or `structured`, the
measurement is broken rather than the adapter good. It came out **best of all nine** —
and its loop fraction is 0.64. See §5.

Restricted to the five genuinely clean categories (loop < 0.10), the ordering is
sensible and is the real prompt-class result:

```
structured 1.344x  >  reasoning 1.327x  >  factual 1.235x  >  prose 1.130x  >  technical 1.077x
```

Constrained continuations draft well; open-ended technical exposition drafts worst. The
spread is **1.077× to 1.344×** — large enough that a per-workload K would matter if a
policy could identify the workload, which is what Gate 5 tested.

## 5. The degeneration artifact

Greedy decoding from a **base** model loops, and a loop is trivially draftable. Measured
over the full suite (`results/speculative/main_degeneration.json`):

| generated tokens | mean loop fraction | rows with loop < 0.05 | acceptance, clean | acceptance, looped | r(loop, acceptance) |
|---|---|---|---|---|---|
| 128 | 0.179 | 16 / 27 | 0.362 | 0.394 | **+0.523** |
| 512 | **0.498** | **4 / 27** | 0.380 | **0.491** | **+0.429** |

At 512 tokens **half of all generated 8-grams are repeats** and only 4 of 27 prompts are
clean. Acceptance on clean text is essentially **flat** in generation length
(0.362 → 0.380); the pooled mean rises (0.375 → 0.474) only because more of the text is
a loop.

This invalidates a headline the baseline run appeared to support:

| generated tokens | AR | K=4 | apparent ×AR |
|---|---|---|---|
| 128 | 48.67 | 60.73 | 1.248 |
| 256 | 48.85 | 64.04 | 1.311 |
| 512 | 48.02 | 66.51 | **1.385** |

"Speedup grows with generation length" is **mostly degeneration**, not drafting. Every
headline in this document is therefore quoted at **128 tokens** and reported both pooled
and on the clean subset.

## 6. Context scaling — the brief's §19 hypothesis, refuted with a mechanism

Held-out WikiText-103 prose, 128 generated tokens. **Two throughput columns, because at
16K the prefill is 14.2 s against ~3 s of generation and is identical for every arm —
end-to-end ratios are dragged toward 1.0 by a cost no decoder can change.**

| context | AR end-to-end | AR decode-only | K=4 end-to-end | K=4 decode-only | ×AR end-to-end | **×AR decode-only** |
|---|---|---|---|---|---|---|
| 512 | 42.44 | 47.93 | 58.28 | 69.20 | 1.373 | **1.444** |
| 2 048 | 31.29 | 47.19 | 37.36 | 62.37 | 1.194 | 1.322 |
| 8 192 | 14.24 | 44.63 | 15.68 | 62.34 | 1.101 | 1.397 |
| 16 384 | 7.43 | 42.86 | 7.69 | 47.93 | 1.035 | **1.118** |

**Predicted in advance: the speedup grows with context. It does not.** It is flat at
1.32–1.44× from 512 to 8 192 and then falls to 1.118× at 16 384.

The mechanism is architectural and measurable. The brief's hypothesis assumes sequential
AR decode becomes progressively more expensive as context grows, leaving more for
speculation to amortise. **In this hybrid stack it does not:** AR decode-only throughput
falls only 47.93 → 42.86 tok/s (−11%) across a **32×** context increase, because 24 of
32 layers are Gated DeltaNet recurrences with **constant-size state** — their per-token
cost does not depend on context at all. Only the 8 full-attention layers grow.

Meanwhile the speculative cycle *does* get more expensive with context, because a
width-(K+1) verify forward costs O(width × context) in those 8 attention layers:

| context | verify ms, K=2 | K=4 | K=8 |
|---|---|---|---|
| 512 | 22.44 | 24.70 | 28.01 |
| 16 384 | 25.08 | 28.75 | **41.27** (+47%) |

So the two effects run the wrong way: nothing to amortise, and a verify pass that widens
in cost. K=8 drops **below** native AR at 16K (0.967× decode-only).

**The long-context case for speculative decoding is a property of attention-only
models.** A hybrid linear-attention backbone removes most of the motivation. That is a
real, architecture-specific negative result and it was not in the repository before.

## 7. Controls — does diffusion matter? (brief §14, §17)

K=4, 128 tokens, same verifier, same transaction, same accounting. Only the proposal
source differs.

| arm | tok/s | ×AR | TPF | accept | mean prefix | **clean ×AR** |
|---|---|---|---|---|---|---|
| native AR | 47.34 | 1.000 | 1.000 | — | — | 1.000 |
| **diffusion K=4** | **59.09** | **1.248** | 1.506 | 0.349 | 1.069 | **1.234** |
| n-gram, order 2 | 56.75 | 1.199 | 1.255 | 0.481 | 0.386 | **1.077** |
| n-gram, order 3 | 54.57 | 1.153 | 1.196 | 0.577 | 0.296 | **1.021** |
| refine ×1 | 59.06 | 1.247 | 1.506 | 0.349 | 1.069 | 1.221 |
| refine ×2 | 51.71 | 1.092 | 1.320 | 0.698 | 2.011 | 1.109 |
| refine ×4 | 36.72 | 0.776 | 0.941 | **0.938** | **2.784** | 0.773 |

### The n-gram control: the artifact that would have killed the headline

On the full suite a **parameter-free list scan reaches 1.199× against diffusion's
1.248×** — within 4%. Taken alone that says diffusion buys almost nothing.

On clean text the gap opens decisively: **1.234× vs 1.077×**. The n-gram drafter's sd on
the full suite is **21.50** — the largest of any arm — because it is enormous on looping
prompts and worthless otherwise. A loop *is* a repeated n-gram, so prompt-lookup
speculation is a near-perfect predictor of degenerate text and a poor one of real text.

**Diffusion matters, but only against non-degenerate text, and only when the comparison
is stratified.** An unstratified benchmark on greedy base-model output would have shown
the two methods as near-equivalent, and that conclusion would have been wrong.

### Refinement: works as a denoiser, loses as a decoder

Predicted in advance to lose. It does — and the mechanism is worth having.

Refinement is genuinely effective at its stated job: mean accepted prefix rises
**1.069 → 2.011 → 2.784** and per-slot acceptance **0.349 → 0.698 → 0.938**. At 4
refinement passes the drafter is proposing almost exactly what the target wants.

And it is **2.8× slower than the single-shot version and 22% slower than no speculation
at all** (0.776× AR), because every refinement pass is a whole extra forward of width K
against a ~50 ms cycle. TPF falls from 1.506 to 0.941 — below 1.0, meaning it now takes
*more* forwards per token than plain AR.

This is the cleanest dissociation in the experiment: **the diffusion process improves the
draft and destroys the economics.** "More denoising is better" is true of the draft and
false of the decoder.

`refine ×1` reproducing `uno_k4` to within 0.03 tok/s (59.06 vs 59.09) is the harness
self-check: the new decoder at one pass *is* the Act IV-U decoder.

## 8. Confidence features (brief §10)

The drafter's own top-1 probability, averaged over the block, against the accepted
prefix that followed. Deciles, 3 417 cycles:

| decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean top-1 | 0.216 | 0.366 | 0.460 | 0.545 | 0.636 | 0.723 | 0.787 | 0.855 | 0.927 | 0.987 |
| **mean accepted** | **0.273** | 0.518 | 0.602 | 0.768 | 0.933 | 1.208 | 1.179 | 1.404 | 1.746 | **2.061** |

**Monotone across nine of ten steps, with a 7.5× spread.** The free confidence signal is
strongly predictive of prefix survival. This is a positive result and it is *not* in
tension with RPRM Stage 1: RPRM scored the denoiser's *entropy* against *per-token*
accept/reject given progress, and stopped on +0.022 AUROC. This scores the drafter's
*top-1 probability* against *prefix length*. Different feature, different target.

The catch is structural and is the reason Gate 5 still fails: **this signal only exists
after the draft forward has run**, and the draft width must be chosen before it. It
cannot inform its own cycle's K. See §10.

## 9. Gate 5 — adaptive K fails

All policies, same session, interleaved, scheduler cost charged to the wall clock.

| arm | tok/s | sd | ×AR | TPF | mean prefix | clean ×AR |
|---|---|---|---|---|---|---|
| native AR | 47.53 | 0.87 | 1.000 | 1.000 | — | 1.000 |
| fixed K=2 | 53.67 | 3.11 | 1.129 | 1.293 | 0.620 | 1.133 |
| **fixed K=4 (best)** | **58.87** | 8.28 | **1.238** | 1.506 | 1.069 | **1.215** |
| fixed K=8 | 53.92 | 13.70 | 1.134 | 1.471 | 1.157 | 1.136 |
| B `adapt_entropy` | 58.45 | 11.80 | 1.230 | **1.542** | 1.197 | 1.213 |
| D `adapt_ev` | 56.20 | 13.33 | 1.182 | 1.407 | 0.990 | 1.133 |
| C `adapt_survival` | 53.61 | 3.11 | 1.128 | 1.293 | 0.620 | 1.129 |

**No adaptive policy beats the best fixed K.** `adapt_entropy` ties it (1.230 vs 1.238
pooled; 1.213 vs 1.215 clean — inside the noise either way) and the other two lose.
**Gate 5 FAIL**, as pre-registered in `SPECULATIVE_DESIGN.md` §2.5.

What the policies actually chose, over ~3 500 cycles each:

| policy | K=2 | K=4 | K=8 |
|---|---|---|---|
| `adapt_entropy` | 0.16 | 0.30 | **0.54** |
| `adapt_ev` | 0.36 | 0.32 | 0.33 |
| `adapt_survival` | **1.00** | 0.00 | 0.00 |

Three distinct failure modes, reported separately because they are not the same finding:

1. **`adapt_entropy` is genuinely adaptive and still does not win.** It varies K over a
   real distribution and achieves the **highest TPF of any arm (1.542)** and a mean
   prefix of 1.197 — better than fixed K=4 on both *algorithmic* metrics. It converts
   none of it into wall clock, because it spends 54% of cycles at K=8 whose extra slots
   cost 1.5 ms each and return almost nothing. This is the same trap U4 documented for
   static K=8, reached by a different route.
2. **`adapt_survival` collapsed to a constant.** It chose K=2 in 100% of cycles and its
   numbers are identical to fixed K=2 to three decimals. That is **a defect in the policy
   rule, not evidence about adaptive decoding**: the rule widens only while the recent
   accepted prefix supports it, and at a mean prefix of ~0.6 it can never justify K=4, so
   it can never gather the evidence that would. It is reported as a relabelled fixed-K=2
   arm and nothing is concluded from it.
3. **`adapt_ev` never committed.** Its selection is near-uniform (0.36/0.32/0.33) —
   round-robin exploration plus a noisy online acceptance estimate left it indecisive.
   With cycle costs of 46.7 / 49.8 / 53.8 ms, the entire spread a scheduler is competing
   for is **15%**, and it must find it from a one-cycle-lagged signal against
   per-cycle acceptance that is 40% zeros.

The honest summary: the margin is too small and the signal too lagged. Adaptive
scheduling is not rescued by a different metric here.

## 10. What actually limited performance

In order of how much each costs:

1. **The survival curve.** `P(accept ≥ 1) = 0.60`, halving each slot, under 5% by j=4.
   Three slots pay; a fourth never has. Everything else is second-order.
2. **The two-forward cycle floor.** The K=1 control measures it directly: **0.814× AR**
   with zero speculation. 24 of 32 layers are a recurrence that cannot be rewound
   mid-forward, so the verify pass re-sends the seed and partial rejections replay.
3. **40% of cycles accept nothing.** The decoder alternates rather than steadily
   accepting; sd is 8.16 tok/s against AR's 0.87.
4. **The confidence signal arrives one forward too late** to choose its own K (§8).
5. **At long context, there is nothing to amortise** (§6) — a property of the hybrid
   linear-attention backbone, not of speculative decoding.

## 11. Failure modes and things that would have been overclaimed

* **"Speedup grows with generation length" (1.25× → 1.39×) is degeneration.** Clean-text
  acceptance is flat in length.
* **"`high_entropy` drafts best" is degeneration** — loop fraction 0.64, the worst in
  the suite. The pre-registered diagnostic caught it.
* **"A list scan matches diffusion" is degeneration** — true on the full suite (1.199×
  vs 1.248×), false on clean text (1.077× vs 1.234×).
* **"The decoder is not lossless" is arithmetic** — 759/759 divergences within one bf16
  ULP, 0 defects.
* **"Speculative decoding wins bigger at long context" is false here**, for a reason
  specific to constant-state recurrent layers.
* **`adapt_survival` is a broken policy**, not a result about adaptivity.
* **`refine ×4` has the best acceptance in the whole experiment (0.938) and is the
  second-slowest arm.** Acceptance is not speed.

## 12. Answers to the brief's §26

**A. Exactly identical greedy output?** Yes, up to bfloat16 tie-breaking. 759 divergences
audited across 5 experiments, **all within one ULP, zero unexplained**. 1.24% of decoded
positions carry an exact bf16 top-2 tie, which no decoder can remove.

**B. Beat native AR wall-clock?** Yes. **1.245×** pooled, **1.230×** on clean text, at
128 generated tokens, against a baseline stable to ±1.5% across a week.

**C. Fastest configuration.**

| | |
|---|---|
| target model | Qwen3.5-4B-Base, bf16, frozen (digest unchanged throughout) |
| drafter checkpoint | `runs/u4b1/step-16000` (r=16 LoRA, 21.2 M params, 0.50%) |
| K | **4** (1 seed + 3 speculative slots) |
| refinement steps | **1** — single-shot; more refinement is strictly worse |
| transaction mode | `snapshot` |
| context / generation | short prompt / 128 tokens |
| native AR | **48.46 tok/s** (sd 0.87) |
| speculative | **60.32 tok/s** (sd 8.16) |
| **speedup** | **1.245×** (clean subset 1.230×) |
| mean accepted prefix | 1.069 (P(0)=0.40, P(3)=0.15) |
| verification calls | 1 per cycle; 1.506 tokens per forward overall |
| peak memory | 8.89 GB vs 8.59 GB for AR |

Best single measurement anywhere in the run: **1.444× decode-only at 512-token context**.

**D. Did adaptive K beat the best fixed K?** **No.** Best adaptive 1.230× vs fixed K=4's
1.238× pooled (1.213 vs 1.215 clean). The best adaptive policy wins the algorithmic
metrics (TPF 1.542, prefix 1.197) and converts none of it to wall clock.

**E. What limited performance?** The survival curve first (§10.1), then the two-forward
cycle floor the K=1 control prices at 0.814× AR.

**F. Did diffusion matter versus simpler controls?** **Yes, but only on clean text, and
the margin is smaller than it looks.** Against prompt-lookup: 1.248× vs 1.199× pooled —
a 4% edge that would not justify 21 M parameters. On clean text: **1.234× vs 1.077×** —
a real edge. The diffusion drafter earns its keep exactly where the cheap control fails.

**G. The single next experiment.** **Act IV-U5, already written and frozen**
(`act4u5_design.md`, `act4u5_preregistered_criteria.md`), never run, and now unblocked.
It asks whether training at K=6/K=8 yields a better *K=4* decoder than training at K=4
under a matched budget. §3 here says why it is the right next move: the survival curve is
**independent of K at decode time**, so the drafter's reach is a property of the adapter,
and the only lever left is what the adapter was trained against. U4 saw exactly that
effect once, unmatched and un-pre-registered.

The runner-up, cheap and not yet designed: **decouple draft width from verify width.**
§8 shows the drafter's post-draft confidence predicts prefix survival with a 7.5× spread,
and it arrives too late to choose K — but not too late to choose how many slots to
*verify*. Draft at K=8, verify only the confident prefix. Upside is bounded by the
verify-cost spread (22.4→28.0 ms at 512 context), so this is worth ~5 ms of a 50 ms
cycle: real, but small, and it should be costed before it is built.

---

## Reproduction

```bash
cd /Users/johnathonkillaly/code/diffusion-u5
export HF_HOME=/Volumes/SHUTTLE PYTHONPATH=$PWD/src
PY=/Users/johnathonkillaly/code/diffusion_project/.venv-unsloth/bin/python

$PY scripts/spec_decode.py --tag main --repeats 3 baseline --token-counts 128,256,512
bash scripts/spec_run_all.sh                 # ksweep, controls, confidence, context
$PY scripts/spec_decode.py --tag main --repeats 3 --tokens 128 adaptive \
    --allowed 2,4,8 --cost-ms '{"2":46.67,"4":49.81,"8":53.81}'
$PY scripts/spec_audit.py divergence --file main_ksweep.json
$PY scripts/spec_audit.py degeneration --lengths 128,512
$PY scripts/spec_report.py --tag main       # CSVs + 12 plots, no model needed
```

Tests: **405 passed** (`-m "not model"`), 22 of them new in
`tests/test_uno_speculative.py`.

---

## Addendum — 2026-09-14: §12 G was run

Act IV-U5 ran the experiment §12 G named and found no cross-horizon transfer. No arm
trained at K=6, K=8 or a 4→6→8 curriculum beat a matched K=4 control at K=4 decoding.
The verdict is `U4 OBSERVATION WAS NOISE` ([`act4u5_results.md`](act4u5_results.md)).
The lever G pointed at, what the adapter is trained against, did not move the drafter's
reach at this budget, within U5's power of about ±0.10 accepted prefix.

U5 also found that identical training launches diverge: a K=4 prefix spread of 0.053
across three launches. Nothing in this document compares separately trained adapters,
so its measurements stand. Every number here is a decode-time measurement of one fixed
adapter, `runs/u4b1/step-16000`. What does not carry over is why that adapter was chosen:
U4 picked it as the best K=4 drafter from a single-launch comparison, and U5 shows that
comparison is inside launch noise.

G's runner-up, decoupling draft width from verify width, is untouched by U5 and remains
untested.
