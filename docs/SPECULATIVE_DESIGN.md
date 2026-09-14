# Act IV-S — speculative decoding: design

The narrow question, taken from the incoming brief:

> Can a diffusion-trained adapter predict multiple future tokens in parallel cheaply
> enough that the untouched autoregressive target can verify and accept them, producing
> exactly the same output faster?

Act IV-U3/U4 already answered the core of that: **yes, at 1.195× AR with K=4**
(`docs/act4u4_results.md`). This document covers only what U4 left unmeasured, listed in
[`STATE_RECOVERY.md`](STATE_RECOVERY.md) §5. It is a *completion*, not a restart.

Written before the results existed. Where a prediction is made, it is made here so it
can fail.

---

## 1. What is inherited and not re-litigated

| inherited | from |
|---|---|
| the draft→verify cycle, transaction, cache snapshot/restore | `src/qdif/uno/decode.py`, `transaction.py`, `cache_utils.py` |
| greedy acceptance, checked against a slow sequential reference | `verifier.py` |
| keyed draft noise (one stream per cycle index) | `rng.py` |
| cost model v2, measured not fitted, 2.36% mean error | `cost_model.py` |
| `committed = accepted + 2`, exactly | `cost_model.committed_tokens` |
| K=4 is the best static block size; only 3 slots ever pay | `docs/act4u4_results.md` |

`decode.uno_greedy_generate` is **not modified**. Everything new lives in
`src/qdif/uno/speculative.py`, and `tests/test_uno_speculative.py` asserts that the new
decoder at a fixed K reproduces the old one's tokens, per-cycle acceptance, per-cycle
commits and forward count exactly. Without that test, a new number and an Act IV-U
number would not be comparable.

## 2. What is new

Five measurements, all **evaluation on frozen checkpoints**. No training. The brief's
§15 gates retraining behind evidence from existing checkpoints, and `AGENTS.md` §8
forbids launching a long training run unasked.

### 2.1 Prompt-class stratification (brief §13)

Nine categories. The first five are `data.PROMPT_SUITE` **verbatim** so the numbers sit
beside U4's without a caveat; four are new because the brief names categories the Act
IV-U suite never had: `technical`, `dialogue`, `reasoning`, `high_entropy`.

`high_entropy` is the control that makes the rest interpretable. It is deliberately
near-unpredictable text. **Prediction: acceptance there should be markedly lower than
`code` or `structured`.** If it is not, the measurement is broken rather than the
adapter being good.

### 2.2 Context scaling (brief §19)

512 / 2048 / 8192 / 16384 tokens of held-out WikiText-103 validation prose.

The hypothesis worth testing is specific: sequential AR decode gets more expensive per
token as the KV cache grows, while a speculative cycle amortises that cost over several
committed tokens. **Prediction: the speedup grows with context.**

The context is real prose, not a repeated phrase. Padding with repetition would make the
context trivially draftable and acceptance would rise with length for a reason that has
nothing to do with length.

Note the asymmetry this experiment is *not* allowed to exploit: 24 of 32 layers are a
Gated DeltaNet recurrence with **constant** state size. Only the 8 full-attention layers
have a KV cache that grows. So the expected effect is real but eighth-ish, not the
effect a pure-attention model would show. Stated in advance so a small number is not
later described as a disappointment.

### 2.3 A non-diffusion drafter control (brief §17)

Prompt-lookup / n-gram speculation: find the most recent earlier occurrence of the last
`order` tokens, propose whatever followed it. No model, no parameters, a list scan.

It runs through the **identical** verifier, transaction, commit rule and accounting, so
the only thing that differs is where the proposal comes from.

This is the control that can kill the headline. If a list scan matches a 21M-parameter
diffusion adapter, then diffusion is not what is buying the speedup. Two orders (2 and 3)
because the right order is not obvious.

### 2.4 Refinement steps (brief §14)

The Act IV-U draft pass is **single-shot**: one forward over `[seed, noise…]` yields the
whole proposal. That is a block predictor more than a denoiser. Refinement re-runs the
pass with the previous proposal in the canvas, which is what makes it a diffusion
sampler at decode time.

Indexing matters and is easy to get wrong, so it is fixed in code and in a test: the
draft block is `[t₀, x₁ … x_{L-1}]`, output position `i` predicts `t_{i+1}`, so
`proposal = [t₁ … t_L]` and the estimate of `t_i` is `proposal[i-1]`. Refinement feeds
`[seed] + proposal[:-1]`. Feeding `proposal[1:]` would shift the canvas by one and make
refinement quietly destructive.

**Prediction: refinement loses.** Each extra pass is a full forward of width L against a
measured marginal slot price of 1.28–1.52 ms, and it must raise acceptance enough to pay
for a whole extra forward per cycle. Recorded so that "more refinement is better" is
tested rather than assumed (the brief explicitly warns against assuming it).

### 2.5 Adaptive K (brief §11–12, Gate 5)

**The constraint that shapes every policy: the block width must be chosen *before* the
draft forward runs.** No feature of the current cycle's draft or verify pass is
available — those are downstream of the choice. Only history is. Any design that uses
the current cycle's drafter confidence to pick the current cycle's K is circular, and
several obvious-looking schedulers are circular in exactly that way.

Four policies, matching the brief's A/B/C/D:

| | policy | signal |
|---|---|---|
| A | best fixed K | none — the comparator |
| B | `EntropyThresholdK` | the verifier's entropy at the previous cycle's first speculative position, already recorded at zero cost |
| C | `SurvivalK` | an exponentially-weighted estimate of the recent accepted-prefix length |
| D | `ExpectedValueK` | `value(K) = (E[accepted|K] + 2) / measured_ms(K)`, with the cost curve **measured** and passed in, and per-K acceptance estimated online with round-robin exploration |

Scheduler cost is inside `overhead_seconds` and therefore inside the wall clock.

**Prediction: adaptive K does not beat fixed K=4, and the reason is structural rather
than statistical.** U4 measured the whole usable survival curve as living at j ≤ 3; the
spread between the best and worst K at 45.8–55.0 ms/cycle is ~20%, and a policy has to
find that margin using a one-cycle-lagged signal. RPRM Stage 1 separately found the
*drafter's* uncertainty worth only +0.022 AUROC. Recorded as a prediction so that a
failure is a result and not a retreat.

#### Relationship to RPRM Stage 1 (STOP)

RPRM asked whether the **denoiser's** uncertainty about **its own drafted tokens**
predicts acceptance well enough for **adaptive early exit inside a block**. Verdict FAIL
→ STOP, frozen.

This is a different predictor with a different target: choosing the *width of the next
block* from the **verifier's** entropy and from acceptance history. RPRM's thresholds are
not touched, its STOP is not reopened, and its diagnostic top-1 result is not cited as
support. If adaptive K also fails, that is a second independent negative result, not a
re-run of the first.

## 3. Losslessness, and a limit discovered while building this

Gate 1 requires the speculative output to equal the native greedy output token for
token. The first smoke run broke it: 1 row in 27, at token 2 of a `code` prompt.

It is not a logic defect. At that position the target's top two logits are
**exactly equal in bfloat16** (both 24.875, gap 0.000000e+00), for `' -'` and `'-'`. A
width-1 AR forward and a width-(L+1) verify forward are different reduction orders over
the same maths, so they can break that tie differently, and **neither is wrong** — both
are the frozen model's greedy next token.

Act IV-U4's gate U4-B4 verified losslessness in the **cacheless reference regime**, where
both paths use the same forward width and ties therefore break identically. That is why
this never appeared before. It does not invalidate U4; it means U4's guarantee was the
narrower one, and this document states the wider one.

Consequently two verdicts are reported and never conflated:

* **`lossless`** — byte-identical token IDs.
* **`lossless_modulo_ties`** — identical except at positions where the target's top-2
  logits are exactly equal in its own dtype.

Every divergence is audited automatically (`audit_divergences`): the full-context logits
are recomputed at that position and the contested gap reported. A divergence with a
non-zero gap is a **bug** and is reported as one. The base rate of such ties is measured
separately (`measure_tie_rate`) as a property of the target model in bf16, which no
decoder can drive to zero.

## 4. Measurement hygiene

The brief's §20, enforced structurally:

* **Arms interleave inside each prompt inside each repeat.** Thermal drift hits every
  arm equally. Measuring the baseline at the start of a session and the candidate an
  hour later is the one mistake this harness cannot make.
* **Median and dispersion. Never best-of-N.** Noted in `STATE_RECOVERY.md` §5:
  `scripts/uno.py bench` does report best-of-N. No published Act IV-U number came from
  it, and nothing here uses it.
* U4 measured **between-session** sd of K=4 tok/s at 2.36 against **within-session**
  repeat noise of 0.15, and the cause is the draft-noise realisation, not heat. So every
  comparison is in-session with the noise stream keyed by cycle index.
* U4 also found the **median over 15 prompts is fragile** and inflated two effects. Both
  median and mean are reported; where they disagree the mean is quoted.
* Warm-up passes are discarded so Metal compilation is outside the timed region.

## 5. Gates

From the brief's §22, fixed before the runs:

| gate | condition |
|---|---|
| 0 baseline | native AR stable and reproducible |
| 1 correctness | speculative greedy output == native, modulo audited bf16 ties |
| 2 useful drafting | mean accepted prefix > 1 on some useful workload |
| 3 actual speed | some configuration beats native AR end-to-end |
| 4 meaningful speed | ≥1.10× modest / ≥1.25× useful / ≥1.50× strong / ≥2.00× exceptional |
| 5 adaptive value | adaptive K beats the best fixed-K policy on the mixed suite |

Gate 5 is the one with a recorded prediction of failure. If it fails it is reported as a
failure, not rescued with a different metric.

## 6. Out of scope

* **Act IV-U5's training run** (~8 h). Its criteria are frozen and it deserves to run as
  its own pre-registered experiment.
* **Sampling-preserving speculative decoding.** The brief's §18 says not to mix sampling
  into the initial verdict. Greedy only.
* **A learned confidence network.** §10 says measure the free features first.
* **K=32.** §5 permits it only if earlier evidence justifies it. U4's survival curve is
  dead by j=7; it will not.
