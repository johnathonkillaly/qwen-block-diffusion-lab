# Findings

**Research closed 2026-09-14.** Nothing below is a plan.

Short enough to read without the journal. Everything here is backed by a run in
`runs/` with `run_metadata.json` recording the commit, package versions, dataset
revision and seeds. Longer narrative: [docs/ACT3_JOURNAL.md](docs/ACT3_JOURNAL.md)
and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

Model throughout: **Qwen3.5** (0.8B in Act I, 4B-Base from Act II on), BF16, LoRA,
Apple Silicon. From Act IV-S on, raw results are committed under `results/`. The map of
every stage, report, dataset and checkpoint is [docs/RESULTS_INDEX.md](docs/RESULTS_INDEX.md);
the short story is [docs/PROJECT_SUMMARY_THROUGH_U6.md](docs/PROJECT_SUMMARY_THROUGH_U6.md).

---

## Demonstrated

**Qwen3.5's Gated DeltaNet is causal by construction, not by masking.** A causal
depthwise `conv1d` feeds a sequential recurrence with `tril`/`triu` baked in. The
`attention_mask` those layers receive is a padding mask. In Qwen3.5-4B that is 24 of
32 layers. No mask makes them bidirectional — this is an architectural fact, verified
in code and empirically (`qdif probe-bidir`: 0/15 earlier positions move under the
native recurrence, 15/15 with our reverse pass).

**A pretrained causal recurrence can be run in both directions on shared weights.**
Implemented, verified by object identity (4B stays 4B — parameters grew only by LoRA +
fusion), with the prefix-leakage boundary holding exactly (0 prefix positions move).
It works. It just does not help — see Negative results.

**FLARE's block-end state readout needs no kernel work in MLX.** `mlx_lm`'s
`gated_delta_update` already returns the final recurrent state, so "every noisy
position reads the completed block-end state" is one extra matmul.

**A falling loss proves nothing.** Act I experiment 001 had loss fall 5× and identity
accuracy reach 96.9% while the model had learned only to *echo its input* — identity
accuracy tracked the copy baseline to the decimal. `lift_over_copy` and
`corrupted_accuracy` exist because of that.

**Copy collapse is a transient, not a terminal state.** Act I experiment 004 corrected
experiment 001: the same configuration run 150 steps instead of 60 escapes it and
reaches 95% corrupted-position accuracy. A run stopped inside that region looks like
success by loss and failure by mechanism.

**Pure-diffusion adapters destroy the AR path.** With Act I/II adapters active,
reference perplexity degraded 8.06 → 115.87 (14.4×) and greedy decoding repeated a
single token — the exact consequence of teaching an unshifted readout with no AR term.
Toggling adapters off restores the base model bit-for-bit.

**Adding an AR term fixes that.** In Act III, with `L_total = L_AR + L_diff`,
held-out AR perplexity went 8.91 → ~5.2 *while* the diffusion path was learning. The
model got better at its original job, not worse.

**Absorbing-state masking is what made denoising work here.** Act II Phase 3 under
uniform random-token corruption plateaued at ~6–7% masked accuracy, flat in `t`. Act
III under `[MASK]` corruption reached **58.2% / 41.8% / 13.0%** at t = 0.10 / 0.50 / 0.90
on held-out text — strongly noise-dependent, which is the signature of a model actually
reading its canvas. Held-out diffusion cross entropy fell 8.64 → 2.04 at t = 0.10.

**The model learned conditional infilling, not unconditional generation.** Given real
partial context it reconstructs unseen WikiText well. Started from a *fully masked*
canvas it collapses to high-frequency repetition ("the the the sun was the the"), which
the health table predicts: masked accuracy at t = 1.00 is only 5.37%.

**On Apple Silicon, Unsloth is an MLX stack.** `DEVICE_TYPE == "mlx"`; `unsloth_zoo`'s
darwin-arm64 dependencies are mlx/mlx-lm/mlx-vlm and exclude torch/peft/trl. The one
component that matters here is its Qwen3.5-specific Gated DeltaNet custom VJP, which
is verified active in every run header and is genuinely optional.

**Unsloth Studio cannot own a custom diffusion training loop.** `/api/train/start` has
a closed `training_type` enum and no custom-script, hook or callback field among its 72
parameters. (Its `/api/train/diffusion/*` endpoints are *image* diffusion LoRA.)

---

**A frozen backbone plus a token-conditional LoRA gives exactly-lossless block
decoding, for free.** Act IV-U's adapter+verifier reproduces frozen-AR greedy output
with 100% token equality at K=1,2,4 — with a *trained or an untrained* adapter, because
the verifier guarantees it. Worth stating precisely because it is the claim most easily
mistaken for a result: it is a property of the algorithm, costs nothing, and is not
evidence that the method works.
*Qualified 2026-09-14:* "exactly lossless" held in the cacheless reference regime on the
prompts Act IV-U checked. Measured more widely it is exact **up to bf16 ties**. Act IV-S
audited 759 cached-regime divergences from native greedy output, all within one bf16 ULP
of a tie, and Act IV-U5 found 2 of 15 prompts diverging at ties in the cacheless regime too.
The verifier commits exactly the target's own argmax for the forward width it runs (see
*speculative greedy output equals native greedy output except at bf16 ties*, below).

**The adapter learns a real, control-separated part of the AR trajectory.** Held-out
agreement at future slot +1 rose from 6.3% (untrained) to 39.1% at K=2, while a
shuffled-teacher control stayed at 3.9%. Free-running acceptance improved 2.9–4.5×
over untrained at every block size. See [docs/act4u_results.md](docs/act4u_results.md).

**The replay forward was the whole bottleneck, and it can be removed.** Act IV-U2
unrolls the Gated DeltaNet recurrence *inside the existing verify forward* so per-token
state is captured, making a prefix commit a state selection rather than a model pass.
With **no retraining, the same adapter and output token-identical to the replay decoder's**,
tokens-per-forward
went 0.862 -> **1.279** at K=4 and wall clock 0.79x -> **1.03x** AR (1.09x at K=2).
Act IV-U's counterfactual prediction of 1.20-1.28 was accurate.
See [docs/act4u2_results.md](docs/act4u2_results.md).

**Gated DeltaNet is algebraically invertible but not numerically invertible.** The
one-token inverse is exact rank-1 algebra (verified to 4.8e-07), yet two independent
mechanisms destroy information in the forward pass: `beta` saturates to exactly 1.0 in
bf16 (0.043% of head-steps), making the update an orthogonal projection that annihilates
the `k` direction; and decay `g` reaches 0, forgetting the state outright. Worst
single-step amplification `1/((1-beta)g)` is 2.9e+20. Algebraic rewind therefore needs a
snapshot fallback, cannot replace the snapshot it depends on, and is slower than it.

**"Draftability gap" predicts block acceptance about twice as well as entropy.**
`TV(p(y | true predecessors), p(y | noised predecessors))` -- a property of the frozen
model alone -- correlates -0.211 with acceptance against entropy's -0.097 (n=2688), and
its quintiles are monotone where entropy's are not. This explains the Act IV-U anomaly
that the *lowest*-entropy bucket drafted badly: a near-deterministic token is often
fixed by the token immediately before it, which is exactly what a parallel draft cannot
see.

**Acceptance, TPF and wall-clock throughput all scale with adapter training.** Act
IV-U3 trained 8x longer (400 -> 3200 steps) with the decoder, backbone, adapter
architecture, objective, data and harness all held constant. At K=4 every link of the
chain moved together: validation TV 1.1041 -> 0.9862, held-out teacher-forced agreement
0.2240 -> 0.2917, free-running acceptance 0.2275 -> 0.3415 (+50%), TPF 1.2789 -> 1.4314,
forwards/token 0.7819 -> 0.6986, and wall clock 51.05 -> 56.45 tok/s (1.056x -> 1.168x
AR, a 2.8-sigma move against measured repeat noise). Lower loss really did become more
accepted tokens and then real throughput. The relationship is sub-linear: +50%
acceptance bought +10.6% wall clock. See [docs/act4u3_results.md](docs/act4u3_results.md).

**K=4 overtook K=2 once the horizon extended, and K=2 is nearly exhausted.** In U2, K=2
was the faster decoder (1.09x vs 1.03x AR). After scaling, K=4 leads 1.168x vs 1.086x.
The reason is visible in the future-offset curve: relative improvement was +21% at
offset +1 but **+72% at +2**, so the parallel horizon genuinely lengthened rather than
the next-token predictor merely sharpening. The fitted decoder cost model puts K=2's
structural ceiling at 1.343x AR (1.50x is unreachable at *any* acceptance) against
K=4's 2.107x.

**Training helps hardest where drafting is hardest.** Bucketing held-out positions by
the frozen model's draftability gap, the *least* draftable quintile improved most
(+0.125 acceptance) and the most draftable improved least (+0.026) between steps 400 and
3200. The adapter is learning to bridge missing-predecessor dependence, not just
exploiting regions that were already easy — and this flattens the gap-acceptance
correlation, which is why that correlation does not strengthen with training.

**The useful parallel horizon is four tokens, and the limit is economic.** Act IV-U4
scaled K=4 to 12,800 steps (3x U3's whole budget) and then trained K=8 two ways. K=4
hit its pre-registered saturation condition at step 6400: the remaining 6400 steps
moved held-out agreement +0.008 against a measured 2 sd of 0.011. K=8 does learn --
its agreement at offsets +5/+6 is 0.14, several times the untrained control's 0.023 --
and it commits **more** tokens per forward than K=4 (TPF 1.4664 vs 1.4314), yet it is
**0.910x K=4** on wall clock, because it pays +11.9% per cycle for +2.4% TPF. The
marginal analysis says why: a draft slot must earn 1.5 ms, and `P(accepted >= 4)` is
0.059 against a break-even of 0.088. Only **three speculative slots pay for themselves
at any block size tested**, which is exactly a block of four.
See [docs/act4u4_results.md](docs/act4u4_results.md).

**~~A longer training block improved *short*-block decoding more than more short-block
training did.~~ Not supported: see the Act IV-U5 qualification below.** *(The original
Act IV-U4 text follows, kept as the record.)* 3200 steps at K=8 (or K=6 then K=8) raised K=4 free-running acceptance
0.3226 -> 0.3525 (~7 sd) and K=4 TPF 1.4035 -> 1.4314, after 9600 steps of K=4
training had left both flat. Paired, same session, same noise stream. Not
pre-registered and measured once, so it is a hypothesis rather than a result -- but it
separates "best block size to train at" from "best block size to decode at", which
Act IV-U4 had assumed were one question.
*Qualified 2026-09-14:* Act IV-U5 tested this with a matched control and replicate
launches and did not support it. The ~7 sd compared two separate training launches
against evaluation noise alone, and launch-to-launch spread is larger (see *training is
not reproducible across launches*, below, and
[docs/act4u5_results.md](docs/act4u5_results.md)).

**~~The official Uno curriculum's intermediate stage does real work.~~ Not established: see
the qualification below.** *(The original Act IV-U4 text follows, kept as the record.)* Going 4 -> 6 -> 8
beat going 4 -> 8 directly on every aggregate at equal step count: held-out TV 1.3046
vs 1.3191, agreement 0.1992 vs 0.1869, mean accepted prefix 1.143 vs 1.099. The direct
arm's long offsets did not move at all. Confirmed from
`ifm-ai/uno@training/configs/uno_3epoch_curriculum.yaml` that IFM run six equal-token
stages `2,4,6,8,12,16`, and from `training/trainer.py` that a stage transition changes
only the block size -- optimizer, adapter and LR schedule all carry over.
*Qualified 2026-09-14:* the two arms were single training launches, and their 0.044
prefix difference is inside the launch-to-launch spread Act IV-U5 measured at K=8
(0.095). That the intermediate stage helps is not established. The reading of IFM's
source stands.

**Draftability does not become more predictive further out.** Pre-registered as a
hypothesis and refuted: mean `|r(gap, acceptance)|` is 0.068 at offsets +6..+8 against
0.128 at +2..+3. Pooled, the gap still beats entropy (-0.167 vs -0.071) with a clean
monotone quintile gradient, so it predicts acceptance -- just not increasingly with
horizon. Three evaluations of an *identical* adapter move these per-offset
correlations by +-0.05, so most of the offset structure is noise.

**A "repeat noise" figure can be dominated by the noise draw rather than the clock.**
Act IV-U3 measured 1.44-1.99 tok/s of apparent timing jitter by re-running each prompt
three times -- but each repeat drew fresh global-RNG draft noise, so the repeats
decoded *different token sequences*. With the stream keyed per cycle, repeats are
bit-identical decodes and the residual jitter is 0.05-0.22 tok/s, ~12x smaller. U3's
gate required beating a baseline by more than the inflated figure, so it was harder
than intended: **that conclusion stands and was understated.** The same keying makes
teacher-forced metrics bit-reproducible across sessions a day apart.

**Evaluation can silently corrupt a training run through a shared RNG.** Batch
construction drew its corruption rate and noise tokens from the *global* MLX stream,
which training also draws from. Changing only `eval_every` and `eval_batches` therefore
shifted every later training noise draw: two runs with identical seeds, data and
hyperparameters diverged in **all 256** adapter tensors. It was caught only because a
pre-registered gate compared the re-run against the archived checkpoint. Evaluation now
uses an explicit PRNG key.

**A frozen diffusion drafter gives a real end-to-end speedup at K=4, and K=4 is where every
slot still pays (Act IV-S).** Setup: adapter `runs/u4b1/step-16000` on frozen
Qwen3.5-4B-Base (bf16, MLX, M4 Max), greedy decoding, 27 prompts in 9 categories, 128
generated tokens, 3 repeats, arms interleaved. Result: **K=4 60.32 tok/s against native AR
48.46 = 1.245×**, and 1.230× on the loop-free subset. Two later sessions reproduced 1.248×
and 1.238×.

Other widths: K=2 1.129×, K=8 1.128×, K=16 0.528×. The prefix-survival curve does not
depend on decode width (about 0.60 at one slot, halving each slot), so three speculative
slots pay and a fourth does not. The K=1 control, with no speculative slot at all, runs at
0.814× AR: that is the cost of the draft→verify cycle itself. See
[docs/RESULTS_SPECULATIVE.md](docs/RESULTS_SPECULATIVE.md).

**Speculative greedy output equals native greedy output except at bf16 ties.** Across Act
IV-S, 759 positions diverged from native AR. Every one was within one bf16 ULP of a tie
(492 exact ties, 0 beyond one ULP). 1.24% of decoded positions carry an exact top-2 tie,
where a width-1 AR forward and a width-(K+1) verify forward break it differently and
neither is wrong. True byte-identity needs a width-invariant target argmax (fp32 logits,
or tie-breaking by token id), not a decoder change.

**Algorithmic metrics are not wall-clock metrics (Acts IV-S, IV-U5, IV-U6).** The project's
main systems lesson, observed five separate times:
* acceptance improved without speed;
* tokens per forward improved without speed (K=16: best TPF 1.580, slowest decoder 0.528×);
* refinement made predictions much better and decoding slower (0.776× AR);
* adaptive scheduling was more algorithmically efficient and no faster;
* staged verification committed more tokens per cycle and still lost (−1.58% to −9.01%).

A verify forward costs about 20.1 ms fixed plus 0.80 ms per token on this stack, so
anything that adds a forward pays more than its paper metrics suggest.

**Timing needs paired, interleaved comparisons; medians can invent differences.** In Act
IV-U6's calibration, two *identical* K=4 decoder arms had medians 2.3% apart (56.62 vs
57.95 tok/s) while their paired mean differed by 0.16% (95% CI −0.42% … +0.76%). Act IV-U4
had already found a 15-prompt median inflating two effects. Every comparison from Act IV-S
on interleaves arms within each prompt and decides on paired means.

**On MLX/Metal, training is not reproducible across launches, and the spread is larger
than the thresholds it was compared against.** Act IV-U5 launched one K=4 continuation
three times and one K=8 continuation twice, identical in start checkpoint, arguments,
seed, data position, code and environment, all proven from artifacts
(`results/act4u5/u5_replicate_provenance.json`). Loss matches at the first step and
diverges from the second, and every finished adapter differs.

* K=4 mean accepted prefix: 1.028 / 1.029 / 0.976, a spread of 0.053. That is 2× U5's
  preregistered floor of 0.0254; the TPF spread is 5× its floor.
* K=8: 0.988 vs 1.083.

A fixed adapter evaluates bit-identically across sessions, so this is training variance.
**Any comparison between two separately trained adapters is one draw from this spread.**
It qualifies two Act IV-U4 claims above, and it means every future arm comparison needs
replicate launches.

## Negative results

**Shared-weight bidirectional Gated DeltaNet does not improve held-out denoising.**
Act II's central hypothesis, tested with pre-registered criteria and killed by them.
Three independent measurements agreed:

- it lost to the causal baseline at every noise level (−1.06 points mean lift, needed
  +3.00) while costing 1.26× wall clock;
- it never beat its own **position-shuffled** control — the gap started at zero and
  stayed there (slopes ~0 across training), which means the reverse pass supplied a
  perturbation, not position-aligned information;
- given a learned gate, all 24 layers moved *away* from the reverse path
  (0.9500 → 0.9533).

FLARE's block-end readout, transplanted into the same harness, also failed to beat the
causal baseline in that configuration. Both known workarounds for DeltaNet causality
lost to simply relying on the periodic full-attention layers.

**A timestep conditioner can silently eat the canvas.** Act III run 1 aborted at step
125: the conditioner's additive bias reached norm ~14 against a token-embedding norm of
0.66 (**20×**), swamping token identity. Canvas-conditioning L1 collapsed 1.517 →
0.016. The bias was nearly identical at t = 0.1/0.5/0.9, so it was not encoding the
timestep — it had found "add a large constant to canvas positions". Under mask
corruption the module is unnecessary anyway (the `[MASK]` count *is* the noise level),
and FLARE carries no timestep embedding. Removed, plus a permanent norm bound.

**Uno-style block decoding is slower than autoregression on a hybrid recurrent
backbone, and the reason is one forward.** Act IV-U reached tokens-per-forward 0.980
at K=2 against the 1.000 that plain AR gets by definition, and 0.63–0.83× AR wall
clock. A Gated DeltaNet recurrence cannot be rewound into the middle of a forward, so
a *partially* accepted block costs a third forward to replay the accepted tokens.
Remove only that term — the counterfactual for a rewindable, pure-attention backbone —
and TPF is 1.20–1.28 at K=2/4/8. **This is the runtime failing to realise the
algorithm, not the algorithm failing**, and the two are reported separately.

**Low teacher entropy does not predict a longer parallel horizon; high entropy does
predict failure.** Bucketing K=4 agreement by teacher entropy gives 0.197 / 0.316 /
0.237 / 0.197 / 0.075 from lowest to highest — non-monotone, with only the top bucket
clearly collapsing. Near-zero entropy often means the next token is fixed by the
*immediately preceding* one (finishing a word or a name), which is exactly what a
noise-filled draft row cannot see. An argument against naive entropy-routed dynamic
block sizing, which the block-level correlation (r = −0.15 to −0.28) alone would have
hidden.

**Adaptive K does not beat fixed K=4 (Act IV-S, gate 5).** Entropy-threshold, survival and
expected-value schedulers, with scheduler cost charged to the wall clock: the best adaptive
policy reached 1.230× AR against fixed K=4's 1.238× (1.213× vs 1.215× on loop-free text).
* The entropy policy won tokens-per-forward (1.542, best of any arm) and converted none of
  it into wall-clock speed.
* The survival policy collapsed to a constant K=2, which is a defect in its rule.
* The drafter's own top-1 probability does predict prefix survival (a 7.5× spread across
  deciles). But it exists only after the draft forward, too late to choose that forward's
  width.

**More denoising improves the draft and destroys the decoder (Act IV-S).** Four refinement
passes raise mean accepted prefix from 1.069 to 2.784, and per-slot acceptance to 0.938.
They also drop throughput to 0.776× AR, because every pass is a full forward. Wider blocks
show the same dissociation: K=16 has the highest tokens-per-forward (1.580) and is the
slowest decoder (0.528×).

**Greedy base-model output loops, and loops flatter every drafting metric (Act IV-S).** The
mean 8-gram loop fraction is 0.18 at 128 tokens and 0.50 at 512, and it correlates with
acceptance at r ≈ +0.5. Three apparent results were artifacts of it:
* "speedup grows with generation length";
* "high-entropy prompts draft best";
* "a prompt-lookup n-gram drafter matches diffusion": 1.199× vs 1.248× pooled, but 1.077×
  vs 1.234× on loop-free text.

**Long context does not help speculative decoding on this hybrid backbone (Act IV-S).** K=4
decode-only speedup falls from 1.444× at 512 tokens of context to 1.118× at 16K. 24 of 32
layers keep constant-size recurrent state, so native decoding barely slows with context
(−11% over a 32× increase). The width-(K+1) verify pass grows in the attention layers.

**Training at a longer horizon does not make a better short-horizon drafter here
(Act IV-U5).** Four arms (K=4 control, K=6, K=8, curriculum 4→6→8) started from one
byte-identical checkpoint, trained 3200 matched steps each (1.26–1.27 h), and all
decoded at K=4, paired against the matched control on 15 prompts. Accepted-prefix change
vs control:

* B_k6: +0.009 [−0.108, +0.101]
* C_k8: −0.040 [−0.155, +0.071]
* D_curr: +0.000

No arm cleared a gate at any of five checkpoints: U5-1, U5-3 and U5-4 fail, and U5-5 fails by the letter on C_k8's
shortfall. The matched K=4 control itself gained +0.060 over the start, two thirds of
the U4 effect that motivated the experiment. An identical relaunch of C_k8 landed on
the other side of the control (+0.072 against the original's −0.023). Verdict
`U4 OBSERVATION WAS NOISE`. With 15 prompts the paired intervals are about ±0.10
prefix, so a true effect smaller than that is not excluded. See
[docs/act4u5_results.md](docs/act4u5_results.md).

**Decoupling draft width from verify width does not help on this stack (Act IV-U6).** On
the frozen Act IV-S drafter, four preregistered staged configurations ran against K=4,
paired over 81 units. They all committed more tokens per cycle and all were slower:

| config | vs K=4 [95% CI] |
|---|---|
| S(6,4) | −1.58% [−2.89, −0.36] |
| S(8,4) | −1.61% [−3.10, −0.17] |
| S(6,3) | −3.03% [−4.28, −1.91] |
| S(8,2) | −9.01% [−10.58, −7.51] |

The reason is mechanical:
* **At a fixed draft width, staging commits exactly what a coupled wide decoder commits.**
  Per-cycle commits were identical across verify widths on 81/81 units.
* **A verify forward costs 20.1 ms fixed plus 0.80 ms per token on the M4 Max**, so
  verifying 2 twice costs 45.1 ms against 24.5 ms for 4 once.
* **A wider draft canvas does not change the first draft slots.** The draft pass is causal,
  and slots 1–4 differ across widths no more than a noise re-draw makes them differ.

Verdict `VERIFY COST DOMINATES`, with every preregistered prediction within half a point.
See [docs/act4u6_results.md](docs/act4u6_results.md).

**One-batch overfit saturates at 4B and cannot rank mechanisms.** At 0.8B it
discriminated; at 4B both the causal and bidirectional configs hit loss ~1e-4 and 100%
corrupted-position accuracy. It is a plumbing check, not a benchmark.

---

## Reproduction result

**Act III produced a genuinely healthy *denoiser*, and by the letter of its own
pre-registered criteria it is recorded as a FAILURE: 6 of 7 passed.**

What passed (held-out WikiText-103 validation, 2500 steps, 148 min, 21.1 GB unified):

| | step 0 | step 2500 |
|---|---|---|
| masked accuracy @ t=0.10 | 1.4% | **58.2%** |
| masked accuracy @ t=0.50 | 1.6% | **41.8%** |
| masked accuracy @ t=0.90 | 0.8% | **13.0%** |
| canvas-conditioning L1 | 1.62 | **1.86** |
| AR reference perplexity | 8.91 | **5.11** |

Strong noise dependence (the Act II pathology was a flat ~6–7%), the canvas probe rising
rather than collapsing, and the AR path *improving* while the diffusion path learned.

What failed: **criterion 3, visible-token preservation ≥ 80%** — measured 2.8%. On
analysis the criterion was mis-specified for absorbing-state diffusion: the objective
(ours and FLARE's) supervises only positions masked in that view, so a visible token is
never a training target, and the sampler freezes committed positions so those
predictions are never used. It was a reasonable guard when written against Act II's
overwrite-everything pathology and is the wrong measurement here. **We did not rewrite
it.** Recorded as a specification error, with the run reported as 6/7.

Three honest qualifications:

0. **Free generation does not work yet.** From a fully-masked canvas the model produces
   high-frequency repetition. It is an infiller, not yet a generator.

1. **This is not a FLARE reproduction in the benchmark sense.** It is an independent
   implementation of the paper's methodological ideas at a much smaller scale, with
   documented deviations ([docs/ACT3_OBJECTIVE.md](docs/ACT3_OBJECTIVE.md)): one
   diffusion block per sequence instead of `K` blocks of 4, LoRA instead of full-weight
   conversion, WikiText-103 instead of their transfer mix, and **no logit shift** (the
   paper mentions one without giving the formula in the sections we could read).
2. **FLARE reports that data mix dominates algorithmic recipe.** Our corpus is a
   fraction of theirs, so absolute quality here is not evidence about their method.

---

## Not demonstrated

- Any claim about FLARE's published numbers. We did not reproduce their benchmarks.
- That the model produces *fluent* long-form text by block diffusion. Masked-position
  accuracy is not fluency.
- That LoRA is sufficient at scale. Everything here is 1.3% of a 4B model.
- Anything about Qwen3.8-27B. Analysis only ([docs/QWEN38_MIGRATION.md](docs/QWEN38_MIGRATION.md)).
- That self-conditioning or MTP initialisation help. Both implemented or analysed,
  neither tested.
- Any speed claim against optimised inference. Our sampler is unoptimised research
  Python/MLX; comparing it to GGUF or llama.cpp would be meaningless.
- Any speculative-decoding result for CUDA, vLLM, other Qwen models, other hardware,
  sampling-based decoding or long generations.
- The root cause of training-launch nondeterminism on MLX/Metal.
- True byte-identity with native greedy decoding. A width-invariant target argmax was
  never attempted.

---

## Open questions

**Further investigation is intentionally deferred.** The project closed on 2026-09-14.
The questions below are a record of what it raised, not a plan; pursuing any of them would
be a new project.

Raised by Acts I–III:

1. **Multi-block training** (FLARE's `K` blocks of `B=4` per sequence) — far more
   diffusion supervision per forward than our single 128-token block.
2. **The logit shift** FLARE mentions. The clearest known gap in our implementation.
3. **Does the bidirectional DeltaNet look different under a healthy objective?** Act II
   killed it under a pure-diffusion objective in a degenerate regime. Retesting it under
   the Act III recipe is cheap and is the fairest possible rematch.
4. **Visible-token supervision.** Our objective (like FLARE's) supervises only masked
   positions, so visible-token preservation stays at chance. It does not affect
   generation — the sampler freezes committed positions — but a small auxiliary term
   would make the model's canvas readout coherent everywhere, which matters if you ever
   want to *re*-mask.
5. Denoising-step sweep against quality; self-conditioning; scaling to 27B.

For the Act IV-U (Uno) track, after U4:

6. ~~**Does training at K=8 and decoding at K=4 beat training at K=4?**~~ **Answered by
   Act IV-U5: no transfer found**, within that experiment's power (±0.10 prefix).
7. **The official curriculum from the start** (`2 -> 4 -> 6 -> 8`) rather than bolted
   on after 12,800 fixed-K=4 steps.
8. ~~**Dynamic-K routing**~~ **Tested in Act IV-S: adaptive K failed its gate**
   ([docs/RESULTS_SPECULATIVE.md](docs/RESULTS_SPECULATIVE.md)).
9. **Replicated training comparisons.** Any future training-recipe question needs at
   least three launches per arm and more than 15 evaluation prompts. U5's paired
   intervals (±0.10 prefix) and launch spread (0.05–0.10) are both larger than every
   training effect Act IV-U has chased.
10. ~~**Decouple draft width from verify width.**~~ **Tested in Act IV-U6:
    `VERIFY COST DOMINATES`.** Recorded observation, not investigated: about 82% of a
    width-5 verify forward on this stack does not depend on width.
