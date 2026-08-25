# The diffusion objective

## Forward (noising) process

For a clean canvas `x0 ∈ V^C` and a noise level `t ~ Uniform(t_min, t_max)`, each
position is corrupted independently:

```
p        = schedule(t)                      # uniform schedule: p = t
u_i      ~ Uniform(0, 1)                    # i = 0 .. C-1, independent
xt_i     = r_i        if u_i < p
           x0_i       otherwise
```

with the replacement `r_i` drawn from one of two distributions:

| mode | `r_i` | notes |
|---|---|---|
| `uniform` (default) | `~ Uniform{0, …, V-1}` | The canvas always holds *real* tokens. The model never sees an out-of-distribution symbol, and `t = 1` is a genuine "no information remains" condition. |
| `mask` | `mask_token_id` | Absorbing state. Offered as a separate experimental mode. Qwen3.5 has no dedicated mask token, so this feeds the model an input distribution it has never seen. |

Implementation: `src/qdif/diffusion/corruption.py`.

### Determinism

All randomness goes through an explicit CPU `torch.Generator`. A given seed
reproduces the corrupted positions **and** the replacement token ids bit-identically,
on CPU, MPS or CUDA — the canvas is tiny, so keeping the RNG on CPU costs nothing and
buys device-independent reproducibility. Pinned by
`tests/test_corruption.py::test_seeded_corruption_is_exactly_reproducible`.

### Two different "corruption rates"

`targeted_fraction` is the fraction of positions the process chose to replace.
`changed_fraction` is the fraction whose token id actually differs — strictly lower
under uniform corruption, because a random replacement can coincide with the original.
Both are logged. Conflating them biases every noise-level diagnostic.

## Reverse (denoising) model

The model is an **x0 predictor**: given the noisy canvas, the prefix and the noise
level, it predicts the clean token at every canvas position simultaneously.

```
        [ prefix (clean, causal) | canvas xt (noisy, bidirectional within itself) ]
inputs_embeds = embed(concat(prefix, xt))
inputs_embeds[:, P:] += timestep_conditioner(t)                    # zero-init
inputs_embeds[:, P:] += self_conditioner(prev_logits)              # optional, zero-init
hidden        = QwenTextDecoder(inputs_embeds, mask_dict)
logits        = lm_head(hidden[:, P:])                             # [B, C, V]
```

## Readout alignment — the single most important convention

```
AR        : logits[:, i]  ->  x[i + 1]      (shift by one)
diffusion : logits[:, i]  ->  x0[i]         (NO shift)
```

We deliberately do not shift. The model reads the noisy token at position *i* and
must emit the clean token at position *i*.

This has a concrete, checkable consequence that the diagnostics surface on every run:

> An **unadapted** Qwen3.5 evaluated under this objective scores near-zero identity
> accuracy and high next-token accuracy, because it is still doing its old job.

That is the expected starting point, not a bug. Measured on the untrained 0.8B model
at `t = 0.5`: `identity_accuracy = 0.0%`, `next_token_accuracy = 27.4%`. If those ever
invert *before* training, the alignment has drifted and something is wrong. Pinned by
`tests/test_objective.py::test_readout_is_unshifted` and
`tests/test_model_integration.py::test_unadapted_model_behaves_autoregressively_under_the_diffusion_readout`.

Undoing this alignment mismatch is exactly the job we are asking LoRA to do.

## Loss

Token-level cross entropy over canvas positions, computed in fp32 regardless of the
model's compute dtype (a bf16 `logsumexp` over 248,320 logits distorts the reported
number enough to matter).

`diffusion.loss_on` selects which positions contribute:

- **`all`** (default) — every canvas position. The standard x0-prediction objective.
- **`corrupted`** — only positions the forward process targeted. At `t = 0` there are
  none, so the loss is a graph-connected zero rather than a NaN.

### Why `loss_on` is a first-class experimental knob

Under **uniform** corruption with `loss_on: all`, *copying the input is a strong
attractor*. At noise level `t`, a fraction `1-t` of positions are already correct, the
model cannot tell which ones those are, and echoing the input scores `1-t` for free.
Experiment 001 landed squarely in it at 60 steps.

It turned out to be a **transient**, not a terminal optimum: the same configuration run
for 150 steps (experiment 004) escapes it and reaches 95% accuracy on corrupted
positions. That correction matters — a run stopped early in this region looks like
success by loss and like failure by mechanism. See [EXPERIMENTS.md](EXPERIMENTS.md).

This is a structural difference from mask-based (absorbing-state) discrete diffusion,
where corrupted positions are *identifiable* by construction — the model can see
exactly which positions need repair, and copying a mask token scores zero. It is
likely why most published discrete-diffusion LMs use masking. Testing that here is
the obvious next experiment.

## Metrics, and why there are so many

A falling loss proves nothing on its own, so `diffusion_metrics` reports the numbers
needed to distinguish the interesting outcome from the three boring ones:

| metric | what it catches |
|---|---|
| `identity_accuracy` | the headline number — but see `lift_over_copy` |
| `corrupted_accuracy` | accuracy restricted to positions that were actually corrupted. **The genuinely hard subset.** |
| `clean_accuracy` | accuracy on untouched positions. High here alone only proves copying. |
| `copy_rate` | P(prediction == input). ~1.0 means the model has collapsed to echoing. |
| `copy_baseline_accuracy` | what a *pure copier* would score, i.e. `1 - changed_fraction`. |
| `lift_over_copy` | `identity_accuracy - copy_baseline_accuracy`. **Only a positive value means information is being recovered.** |
| `next_token_accuracy` | high for an unadapted AR model; falls as adaptation takes hold |
| `mean_top1_prob`, `mean_entropy` | confidence, used by the adaptive sampler |

Experiment 001 peaked at `identity_accuracy = 96.9%` with `lift_over_copy = 0.0%`.
Without the last two metrics that run reads as a success.

## Timestep conditioning

Sinusoidal features of `t` -> 2-layer MLP -> additive bias on the canvas hidden
states. The output layer is **zero-initialised**, so at initialisation the diffusion
forward pass is bit-identical to the unmodified autoregressive forward pass. That
makes the AR control an exact comparison rather than an approximate one
(`tests/test_model_integration.py::test_untrained_adapters_leave_the_ar_path_bit_identical`).

## Self-conditioning (optional, off by default)

Iteration *k* produces a distribution over clean tokens; that estimate is summarised
as an **expected token embedding** and projected back into the residual stream:

```
p_hat = softmax(logits)
e_hat = p_hat @ E                       # E = the tied input embedding matrix
delta = W_out(SiLU(W_in(RMSNorm(e_hat))))   # W_out zero-initialised
inputs_embeds[canvas] += delta
```

Chosen over a second embedding table or hidden-dim concatenation because Qwen3.5-0.8B
ties input embeddings to the output head, so `E` is already the shared token space and
no vocabulary-sized parameter is introduced. With 248,320 tokens the full `p_hat @ E`
matmul is expensive, so the expectation is taken over the top-32 tokens by default
(`top_k=0` gives the exact full-vocabulary expectation; the approximation is checked
in `tests/test_data_and_config.py::test_topk_expectation_approximates_the_full_one`).

Note for the 27B migration: **Qwen3.8-27B does not tie its embeddings**, so `E` there
is the input table only and no longer coincides with the readout basis. The module
still works; the justification changes.

Training uses the standard two-pass scheme: one no-gradient pass to produce the prior
estimate, one conditioned pass that carries the gradient, applied on a fraction of
steps (`self_conditioning_prob`).

## Sampler

```
1. start from a fully-noised canvas (t = 1)
2. predict clean-token logits for every position
3. score confidence (top-1 probability)
4. commit the most confident uncommitted positions
      by step k of K, ceil(C * (k+1)/K) positions are frozen
5. re-corrupt the uncommitted remainder at t_next = 1 - (k+1)/K
6. repeat; the last step commits everything
```

`strategy: all` rewrites every position each step with no commitment — a control. If
it matches `confidence`, the commitment machinery is not earning its keep.

The metric to watch is **`tokens_per_forward` = canvas_length / K**. Autoregressive
decoding is 1.0 by construction. A diffusion speed advantage exists only if quality
holds at values well above 1.0. `qdif compare` prints both.

Block-autoregressive generation commits a whole block, appends it to the prefix, and
starts a fresh canvas. Because the mask keeps prefix rows strictly causal, a committed
prefix is never disturbed by canvas noise — verified by
`tests/test_masks.py::test_canvas_never_leaks_into_the_prefix`.
