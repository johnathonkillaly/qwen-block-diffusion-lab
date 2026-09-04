# Findings

Short enough to read without the journal. Everything here is backed by a run in
`runs/` with `run_metadata.json` recording the commit, package versions, dataset
revision and seeds. Longer narrative: [docs/ACT3_JOURNAL.md](docs/ACT3_JOURNAL.md)
and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

Model throughout: **Qwen3.5** (0.8B in Act I, 4B-Base in Acts II–III), BF16, LoRA,
Apple Silicon.

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

**The adapter learns a real, control-separated part of the AR trajectory.** Held-out
agreement at future slot +1 rose from 6.3% (untrained) to 39.1% at K=2, while a
shuffled-teacher control stayed at 3.9%. Free-running acceptance improved 2.9–4.5×
over untrained at every block size. See [docs/act4u_results.md](docs/act4u_results.md).

**The replay forward was the whole bottleneck, and it can be removed.** Act IV-U2
unrolls the Gated DeltaNet recurrence *inside the existing verify forward* so per-token
state is captured, making a prefix commit a state selection rather than a model pass.
With **no retraining, the same adapter and byte-identical output**, tokens-per-forward
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

**Evaluation can silently corrupt a training run through a shared RNG.** Batch
construction drew its corruption rate and noise tokens from the *global* MLX stream,
which training also draws from. Changing only `eval_every` and `eval_batches` therefore
shifted every later training noise draw: two runs with identical seeds, data and
hyperparameters diverged in **all 256** adapter tensors. It was caught only because a
pre-registered gate compared the re-run against the archived checkpoint. Evaluation now
uses an explicit PRNG key.

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

---

## Open questions

Worth doing now that a healthy baseline exists:

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
