# Act IV-U — Frozen-Backbone Diffusion Distillation (Uno)

Scored mechanically against [`act4u_preregistered_criteria.md`](act4u_preregistered_criteria.md),
which was written and committed **before** any training run and has not been edited.

**Verdict: `ALGORITHMIC SIGNAL, NO SPEEDUP`.**

---

## Question

> Can a small adapter learn enough of a frozen AR model's future-token trajectory to
> propose blocks of K tokens that the frozen model itself accepts — preserving output
> exactly, and reducing sequential decoding steps?

Answer at this scale: **it learns a real, control-separated amount of the trajectory,
the decoding is exactly lossless, and it is still slower than plain autoregression on
this backbone.** The gap between those three facts is the result.

## Relationship to IFM Uno

IFM released Uno's training code, runtime and a working adapter on 2026-09-03, so the
objective here is **transcribed, not guessed**: TV between a noisy student and a clean
teacher computed from the same frozen weights, `α=β=0, γ=1`, with a token-conditional
LoRA that is off on every non-noise row. Full provenance, tagged
confirmed / inferred / ours, is in [`act4u_uno_source_notes.md`](act4u_uno_source_notes.md).

**One substantive deviation**, forced by the architecture: IFM get teacher and student
from a single forward using a custom `block_diff_mask` over a doubled sequence. That
mask cannot exist on a Gated DeltaNet layer, which is causal by construction rather
than by masking, so we use one noisy block per window at the suffix and two plain
causal forwards. This is *exactly* the computation the draft pass performs at
inference, so it is arguably a tighter train/test match — it just costs two forwards.

**Not reproduced:** the tree sampler (needs FlashAttention-3), and IFM's scale. They
train 14.75B tokens / 28,125 steps / 16 GPUs. This is 400 steps on one Mac —
roughly four orders of magnitude smaller. **No number here is compared to IFM's
~3× or 2.71 TPF.**

## Method

| | |
|---|---|
| Backbone | Qwen3.5-4B-Base, 4,205,751,296 params, **frozen and byte-hashed** |
| | 8 full-attention layers `[3,7,11,15,19,23,27,31]`, 24 Gated DeltaNet |
| Trainable | gated LoRA r=16 α=256 on `q,k,v,o` + `gate,up,down` |
| | **21,233,664 params = 0.5023%** of the backbone |
| Objective | total variation, `α=0 β=0 γ=1` (IFM's default) |
| Corpus | WikiText-103, 4,000 train windows × 128 tokens, held-out validation split |
| Run | 400 steps, batch 4, lr 1e-5, warmup 20, ~9.5 min |
| Eval | 128 held-out rows, teacher-forced; 15-prompt suite, free-running |

## Results

### Gate 0 — integrity. **PASS**

The backbone byte-digest is identical before and after every training run, after
decoding, and after benchmarking. Only `lora_a`/`lora_b` are ever trainable (asserted
at setup, not merely intended). Trainable share 0.5023% < 1%. Student and teacher
logits are bit-identical at the seed position, so the gate provably does not leak onto
clean rows. Stage 0: **26/26**.

The digest is now normalised so that an adapted model hashes identically to the
pristine checkpoint — attaching the adapter renames `q_proj.weight` to
`q_proj.base.weight`, and folding the raw name in would have made a wrapped model look
"changed" when nothing was. Verified on the 4B: bare and adapted both hash to
`7cd0f5a9bce55681…` over 426 tensors.

*One bookkeeping caveat:* `runs/uno-k4-*/result.json` were produced **before** that
normalisation and record the older, wrapper-qualified digest (`e21c6dae…`). Their
integrity claim is unaffected — each run compared its own before and after and they
match — but those two digests are not comparable with the post-fix artifacts. Noted
rather than papered over by re-running.

### Gate 1 — does the adapter learn anything? (K=2) **PASS**

Held-out teacher-forced agreement at slot +1, `corruption="full"`, 128 rows:

| arm | agreement | vs untrained |
|---|---|---|
| untrained (Control 1) | 0.063 | — |
| **trained** | **0.391** | **+32.8 pp** |
| shuffled teacher (Control 2) | 0.039 | −2.4 pp |

Required: ≥ +10 pp over untrained, with the shuffled control not improving by more
than 3 pp. Both hold, comfortably.

### Gate 2 — future prediction at K=4. **WEAK**

Mean agreement over the speculative slots: **0.203**. Pre-registered bands: STRONG
≥ 0.50, MODERATE ≥ 0.25, WEAK ≥ untrained + 10 pp (= 0.157). So **WEAK** — it clears
the control decisively but falls short of MODERATE.

Per-slot agreement (slot 0 is the unadapted seed and is 1.0 by construction):

| | seed | +1 | +2 | +3 | +4 | +5 | +6 | +7 |
|---|---|---|---|---|---|---|---|---|
| **K=4** trained | 1.000 | 0.414 | 0.109 | 0.086 | | | | |
| K=4 untrained | 1.000 | 0.117 | 0.031 | 0.023 | | | | |
| K=4 shuffled | 1.000 | 0.102 | 0.094 | 0.039 | | | | |
| **K=8** trained | 1.000 | 0.469 | 0.148 | 0.125 | 0.102 | 0.094 | 0.070 | 0.055 |
| K=8 untrained | 1.000 | 0.141 | 0.031 | 0.016 | 0.023 | 0.039 | 0.031 | 0.008 |

**Accuracy degrades sharply with distance and then flattens.** The first future token
is ~3.3× better than the untrained control; by +3 the advantage is smaller but still
present, and it never returns to chance. The adapter learns most of what it learns
about the *immediately* next token.

### Gate 3 — verifier correctness. **PASS**

Adapter + verifier reproduces frozen-AR greedy output **exactly** — 100% token
equality — at K = 1, 2, 4 in the cacheless reference regime, with both a trained and
an untrained adapter. The single-forward verifier was also checked against a
deliberately slow implementation that performs L separate AR calls; accepted-prefix
lengths agree in every trial at K = 2, 4, 8, on a corrupted block as well as a clean one.

This is a property of the algorithm, not of training, and it is worth stating that
plainly: **passing Gate 3 is free and means nothing on its own.**

**Cached regime, reported separately as pre-registered.** The backbone is *not
chunking-invariant*: greedy decoding with a KV cache and greedy decoding with full
recomputation produce different token sequences on Qwen3.5-4B **with no adapter
involved at all** (agreement 0.875 on the stage-0 prompt). Cached Uno tracks cached AR
at exactly that same 0.875, i.e. Uno tracks AR as well as AR tracks itself. The
criterion was written this way in advance precisely because exact equality is not a
fair demand of a runtime that cannot reproduce itself.

### Gate 4 — sequential-step reduction. **FAIL**

Free-running on the 15-prompt held-out suite, 48 tokens, greedy:

| K | acceptance (untrained) | acceptance (trained) | committed/cycle | **TPF** | TPF without replay |
|---|---|---|---|---|---|
| 1 | — | — | 2.00 | 0.980 | 0.980 |
| 2 | 0.182 | **0.531** | 2.48 | **0.980** | 1.202 |
| 4 | 0.071 | **0.230** | 2.64 | 0.874 | 1.265 |
| 8 | 0.023 | **0.104** | 2.68 | 0.879 | 1.283 |

Training raises free-running acceptance by **2.9× to 4.5×** over the untrained control
at every block size. Note acceptance is *higher* free-running than teacher-forced
(0.531 vs 0.391 at K=2): the model's own continuation is more predictable than real
corpus text.

But **TPF never reaches 1.0**, so the method uses more sequential model calls than
plain AR. Gate 4 required TPF > 1.000; the best measured is 0.980 at K=2.

**Why, precisely.** A Gated DeltaNet recurrence cannot be rewound into the middle of a
forward, so a *partially* accepted cycle needs a third forward to replay the accepted
tokens from the last committed snapshot. Removing only that term — an accounting
counterfactual for a rewindable, pure-attention backbone like IFM's Qwen3-8B — gives
**TPF 1.20–1.28**, i.e. above 1.0 at every K > 1. The replay tax is the entire
difference between a modest algorithmic win and a small loss. It is an accounting
figure and no wall-clock claim is derived from it.

### Gate 5 — wall clock. **FAIL**

Measured in one harness, AR and Uno back to back in the same process:

| K | tok/s | vs AR |
|---|---|---|
| AR baseline | 10.70 | 1.00× |
| 1 | 7.69 | 0.72× |
| 2 | 8.94 | **0.83×** |
| 4 | 7.12 | 0.67× |
| 8 | 6.78 | 0.63× |

Uno is **slower than AR at every block size**. The pre-registered break-even was
TPF ≥ 1.088 at K=2; measured 0.980.

*Caveat on absolute throughput:* this session's AR baseline measured 10.70 tok/s
against 48–49 tok/s in the standalone baseline and Control-1 runs — the machine was
roughly 4.6× slower throughout this session. Because AR and Uno are measured
back-to-back in the same process, the **ratios** are the meaningful quantity, and they
reproduce: Control 1 gave 0.61× at K=4 on a fast machine, this run gives 0.67×. The
absolute tok/s figures in this table should not be quoted.

### Quality preservation

Under greedy decoding the output *is* the AR output by construction (Gate 3), so there
is no quality delta to measure — that is the point of verified decoding. The frozen AR
reference itself: held-out NLL 2.2635, **perplexity 9.617**, next-token accuracy
52.38%, peak 12.34 GB.

### Entropy analysis (the §26 hypothesis) — **partially supported**

Teacher-forced agreement bucketed by teacher entropy at the speculative slots, K=4:

| bucket | entropy range | mean H | agreement |
|---|---|---|---|
| 0 | 0.00–0.55 | 0.231 | 0.197 |
| 1 | 0.55–1.54 | 1.011 | **0.316** |
| 2 | 1.56–2.54 | 2.084 | 0.237 |
| 3 | 2.56–3.57 | 3.046 | 0.197 |
| 4 | 3.58–7.96 | 4.821 | **0.075** |

Block-level correlation between teacher entropy at the block start and accepted-prefix
length: **r = −0.15 (K=2), −0.28 (K=4), −0.25 (K=8)** — negative, as predicted, at
every block size.

But the relationship is **not monotone**, and the interesting part is the failure of
the prediction at the *bottom*. High entropy strongly predicts failure (0.075 in the
top bucket, ~4× worse than any other). Low entropy does **not** predict success: the
most-determined bucket scores 0.197, *below* the second bucket's 0.316.

The plausible reading is that near-zero teacher entropy often means the next token is
determined by the *immediately preceding* token — finishing a multi-token word or a
proper noun — which is exactly the information a draft row does not have, because it
sees noise there. "Predictable to the teacher" and "predictable without the previous
token" are different properties, and only the second one helps here. That is a real
argument against naive entropy-routed dynamic block sizing, and it would not have
shown up without bucketing.

Per-domain acceptance at K=4 is fairly flat — math 0.297, factual 0.267, structured
0.208, prose 0.200, code 0.176 — with none of the large code/structured advantage one
might have expected.

### Controls

| control | result |
|---|---|
| 1 — untrained adapter | acceptance 0.182 / 0.071 / 0.023 at K=2/4/8. The floor. |
| 2 — shuffled teacher | agreement 0.039 at K=2 vs trained 0.391. **Separates cleanly.** |
| 5 — K=1 | reproduces AR exactly; TPF 0.980 ≤ 1.0 by construction. |
| 4 — NTP LoRA | implemented (`ce_weight>0`), **not run** |
| 6 — native MTP | **not run** — `mlx_lm` drops the MTP head at load |

## Failure modes and things that went wrong

Four are worth recording because each produced a *plausible-looking wrong answer*
before being caught.

1. **A verifier off-by-one that the whole test suite missed.** The commit rule dropped
   the clean token, shifting every output by one. Every decoding test passed anyway,
   because the randomly-initialised tiny fixture emitted a *constant* token — and an
   off-by-one in a constant sequence is invisible. Found only on the real 4B backbone.
   The fixture is now untied (≈19 distinct tokens in 20) and an `assert_varied` guard
   makes a degenerate reference a hard error.
2. **A no-op control.** The first `shuffle_targets` rolled the teacher's *and* the
   student's context together, leaving them perfectly aligned. Under a TV-only
   objective, which never reads the data labels, that made the control bit-identical
   to the real arm — and it duly reported "no separation" (0.250 vs 0.229) which read
   as *the adapter is not using the context*. It was measuring nothing. Fixed to roll
   only the teacher; the control then behaves as a control should (TV rises to 1.85,
   agreement falls to 0.083).
3. **TV's gradient vanishes under saturation.** At lr 3e-4 the student's logits blow up
   (absmax 24 → 76), the softmax goes one-hot, and TV's gradient
   `p_s·(sign − Σ sign·p_s)` becomes identically zero. Training then *freezes at a
   constant loss with |g| = 0* rather than diverging loudly — i.e. it looks exactly
   like "the adapter learned nothing". IFM's 1e-5 trains cleanly. A guard now raises.
4. **A flaky integrity check.** Byte-hashing MLX arrays via `memoryview` on a lazy
   array, and via `.view(mx.uint8)` (a *narrowing* reinterpret), both produced unstable
   digests for numerically identical tensors — a false "the frozen backbone changed"
   alarm on exactly one tensor, `layers.18.linear_attn.conv1d.weight`. A flaky
   integrity check is worse than none, because it teaches you to ignore it.

Also recorded: **the one-batch overfit could not rank anything.** True and shuffled
arms both reached TV ≈ 0.25 with identical agreement, because 4 fixed windows are
equally memorizable either way. This reproduces `FINDINGS.md`'s existing conclusion
that one-batch overfit saturates at 4B and is a plumbing check, not a benchmark.

## Comparison to Acts I–III

Acts I–III all modified *how the backbone reads its input* — bidirectional DeltaNet,
canvas masks, timestep conditioners — and repeatedly hit degeneracy: copy collapse,
canvas ignoring, an AR path destroyed (perplexity 8.06 → 115.87). Act IV-U modifies
nothing: the backbone is byte-identical throughout and the AR path is untouched by
construction, so **the failure modes that dominated Acts I–III cannot occur here.**
That is a genuine structural improvement, and it is also why the negative result is
clean: there is no health gate to argue about, because there is nothing to be unhealthy.

Where Act III produced a healthy denoiser that could not generate, Act IV-U produces a
correct decoder that cannot go faster. Both are honest negatives about *throughput and
generation*, arrived at from opposite directions.

## Interpretation

Three separable claims, of which we achieved two:

1. **Prediction quality — partial.** The adapter learns a real amount of the frozen
   model's next-token behaviour from a noise-filled context (+32.8 pp over untrained at
   K=2, 2.9–4.5× acceptance), it separates decisively from a shuffled control, and it
   degrades with future distance in the way one would expect. Gate 2 lands at WEAK.
2. **Verified lossless decoding — yes, exactly.** And free: it is guaranteed by the
   verifier for any adapter, so it is not evidence for the method.
3. **Speedup — no.** TPF 0.980 at best against a required 1.0; wall clock 0.63–0.83× AR.

The reason is specific and quantified rather than mysterious: on a 24/32-recurrent
backbone a rejected block costs a third forward, because a recurrence cannot be rewound
mid-pass. Strip that single term and TPF is 1.20–1.28. **This is an
implementation/runtime limitation of the architecture, not a refutation of the
algorithm** — the distinction the brief asked to be kept, and the measurement that
keeps it.

Whether more training would close the gap is untested and should not be assumed. To
break even at K=2 the acceptance rate must reach ~0.50 *and* overcome the replay tax;
measured acceptance is already 0.531 at K=2 and TPF is still 0.980.

## Verdict

```
ALGORITHMIC SIGNAL, NO SPEEDUP
```

Gate 4 was missed by 0.02 TPF and Gate 5 by a clear margin, while Gates 0, 1 and 3
passed and Gate 2 landed at WEAK. There is unambiguous, control-separated learning and
a provably lossless decoder; there is no speedup on this hardware and this backbone.

## Next experiment

In priority order, cheapest and most decisive first:

1. **Eliminate the replay forward.** The single highest-value change, worth more than
   any amount of extra training. Either (a) capture per-step DeltaNet states during the
   verify pass so the cache can be rewound to an accepted prefix, or (b) restrict the
   adapter and the block path to the 8 full-attention layers and check whether the
   recurrent layers can be advanced separately. Success turns a measured 0.98 into a
   measured ~1.25.
2. **Train longer and at K=2 specifically.** K=2 has the best economics here
   (break-even TPF 1.088, the lowest of any block size) and the highest acceptance.
   The current run is 400 steps; the loss was still falling.
3. **Control 4 — ordinary next-token LoRA** at matched parameter count, to establish
   how much of the gain is block-specific rather than generic task adaptation. Cheap,
   already implemented, and currently the largest hole in the control set.
4. **`corruption="full"` vs `"uniform"`.** Evaluation always runs at full corruption
   because that is the only state inference produces; training at IFM's `uniform`
   schedule may simply be mismatched at this budget.
5. Only then: rank sweep, K-curriculum, and the MTP baseline.

## Reproduction

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py stage0
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py baseline --decode-tokens 48
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py train --block-size 4 --steps 400 --lr 1e-5 --out runs/uno-k4-true
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py train --block-size 4 --steps 400 --lr 1e-5 --shuffle-targets --out runs/uno-k4-shuffled2
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py compare --arm true=runs/uno-k4-true/adapter.safetensors --arm shuffled=runs/uno-k4-shuffled2/adapter.safetensors
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py bench --adapter runs/uno-k4-true/adapter.safetensors --block-sizes 1,2,4,8
```

Artifacts, each carrying git SHA, versions, seed and full config:
`runs/uno-stage0/stage0.json`, `runs/uno-baseline/baseline.json`,
`runs/uno-bench/control1_untrained.json`, `runs/uno-k4-true/result.json`,
`runs/uno-k4-shuffled2/result.json`, `runs/uno-compare/compare_final.json`,
`runs/uno-bench/trained_k4.json`.
