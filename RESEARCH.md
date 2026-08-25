# Research questions

Live tracker. Each question carries a status, the evidence, and what would move it.
Status vocabulary: **open** (no evidence), **partial** (evidence from a memorisation
test only), **answered** (evidence I would defend), **blocked**.

Evidence lives in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); architecture claims in
[docs/QWEN35_NOTES.md](docs/QWEN35_NOTES.md).

> **Global caveat.** Every training result so far is `overfit-one-batch` on two
> examples of Qwen3.5-0.8B. Nothing here speaks to generalisation. That is the single
> largest gap and the next thing to fix.

---

### 1. Can Qwen3.5 LoRA learn token denoising?

**Partial — yes, on a memorisation test.**

Experiment 004: `full_attn` LoRA, rank 16, 2.4M trainable parameters (0.32%), 150
steps → loss 9.96 → 0.087, corrupted-position accuracy 95.2%, +31.2 points above the
copy baseline. Experiment 003 with more capacity reached 100% / loss 0.006.

Two important qualifications. First, this is memorisation of two examples; a corpus
run has not been attempted. Second, experiment 001 (the same config at 60 steps)
converged to *copying its input*, with a falling loss and a rising identity accuracy
that meant nothing. Copy collapse is a state the optimisation passes through, and it is
easy to mistake for progress.

*To move it:* train on a real corpus and measure reconstruction on held-out text.

### 2. Which layer families require adaptation?

**Partial.**

Full-attention `q/k/v/o` alone is *sufficient* to cross from copying to denoising
(004). Adding the MLP and doubling rank improves corrupted-position accuracy from
68.1% to 94.7% (tail means) at ~25% throughput cost. No preset targeting DeltaNet has
been run.

`qdif arch-report` enumerates every adaptable module; `lora.target_preset` offers
`full_attn`, `full_attn_mlp`, `deltanet`, `deltanet_gates`, `all_attn`, `all_linear`.

*To move it:* the target ablation at fixed rank and steps.

### 3. Does DeltaNet causality prevent effective bidirectional diffusion?

**Answered for the architecture; open for the consequence.**

Gated DeltaNet layers are causal **by construction, not by masking** — a causal
depthwise `conv1d` plus hard-coded `tril`/`triu` inside `chunk_gated_delta_rule`. Their
`attention_mask` argument is a padding mask. No mask can make them bidirectional. In
Qwen3.5-0.8B that is 18 of 24 layers, including layers 0–2, which are purely causal
even inside the canvas.

So bidirectional canvas mixing exists **only** in the 6 full-attention layers. `qdif
probe-bidir` confirms this empirically: perturbing the last canvas token moves all 15
earlier canvas hidden states with the mask relaxed, and **exactly zero** with it causal.

Whether this *prevents effective* diffusion is unanswered — 003/004 denoise well with
only full-attention adaptation, but on memorised data where six bidirectional layers
may be plenty.

*To move it:* implement the two-direction DeltaNet pass and compare. Alternatives and
their trade-offs are tabulated in [QWEN35_NOTES.md](docs/QWEN35_NOTES.md).

### 4. Can diffusion work with only periodic full-attention layers adapted?

**Partial — yes, on the memorisation test.** Both passing runs adapted full-attention
layers only. Not yet a controlled comparison, and not yet on unseen text.

### 5. Does increasing LoRA rank materially improve reconstruction?

**Partial.** r16/`full_attn` → 68.1% corrupted-position accuracy; r32/`full_attn_mlp`
→ 94.7%. Rank and target set were changed together, so the rank contribution is not
isolated.

*To move it:* rank sweep 8/16/32/64 at a fixed target preset.

### 6. At what corruption level does pretrained knowledge stop helping?

**Open.** The noise sweep on checkpoint 003 shows ~98% reconstruction at `t = 1.0`,
but that is a memorised batch being regenerated from the prefix, not a language prior
recovering structure. The question requires held-out text.

### 7. How many denoising steps are needed?

**Open.** The sampler supports 1/2/4/8/16/32 and reports `tokens_per_forward`. At 8
steps on a 32-token canvas (4.00 tok/forward) the memorised passage came back
near-perfectly, and per-step confidence rose monotonically (0.646 → 0.977) while
entropy fell (1.21 → 0.11). A step sweep on non-memorised content has not been run.

### 8. Can AR ability remain unchanged while diffusion adapters are loaded?

**Answered — no.**

With adapters active, reference perplexity degrades **8.06 → 115.87 (14.4x)** and greedy
AR decoding emits `fortune fortune fortune …`. The failure is exactly what the readout
convention predicts: the adapter was trained to emit the token at position *i*, so run
autoregressively — where position *i* must predict *i+1* — it repeats the current token
forever.

The two modes are not simultaneously available in a single forward configuration.

*Caveat:* measured on a model overfit to two examples with `full_attn_mlp` at rank 32.
A lightly-trained, low-rank adapter might be less destructive. The metric to use is
`reference_perplexity`, not self-perplexity, which is degenerate when the output
degenerates.

### 9. Can the same base checkpoint switch between AR and diffusion mode?

**Answered — yes, by toggling.**

`DiffusionQwen.no_diffusion()` disables every `LoRALinear` in place; the AR path returns
to reference perplexity 8.06 with no reload and no second copy of the weights. Because
`lora_B` and the auxiliary output layers are zero-initialised, an *untrained* wrapper is
bit-identical to the base model
(`tests/test_model_integration.py::test_untrained_adapters_leave_the_ar_path_bit_identical`).

Switching is a boolean, not a reload — but per question 8, it is a genuine switch, not
coexistence.

### 10. What needs to change before moving to Qwen3.8-27B?

**Answered — very little in code, two real gaps.** Full analysis in
[docs/QWEN38_MIGRATION.md](docs/QWEN38_MIGRATION.md).

27B declares the same `model_type: qwen3_5`, the same 3:1 hybrid (48 L / 16 F), the same
248,320 vocabulary, the same head dim, the same mRoPE, the same MTP. Module names are
identical, so every LoRA preset resolves. The two substantive differences:

- **Untied embeddings** (0.8B ties, 27B does not) — changes the self-conditioning
  justification and makes `lm_head` a legitimate, arguably ideal, LoRA target.
- **DeltaNet value-head replication** (`linear_num_value_heads` 16 → 48) activates a
  `repeat_interleave` branch that is *dead code* at 0.8B and therefore untested here.

Blockers: no HF bf16 27B checkpoint locally (all local copies are MLX/GGUF quantized
exports), and QLoRA is unavailable on Apple silicon, which makes the MLX backend a
prerequisite rather than an optimisation.

### 11. Does MTP provide anything useful for diffusion initialisation or denoising?

**Open.** The checkpoint ships an Eagle-style MTP head (`mtp.fc`, one full-attention
layer, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `mtp.norm`) that
`Qwen3_5ForCausalLM` does not load.

The appealing hypothesis: MTP already predicts a token at a position from *a hidden
state plus a token embedding at that position*, which is structurally much closer to the
diffusion readout (`logits[i] → x0[i]`) than the base LM head is. That makes it a
candidate **initialisation for a denoising head**, not just a speculative-decoding
trick. Entirely untested.

### 12. Can self-conditioning reduce required denoising steps?

**Open.** Implemented and unit-tested (`SelfConditioner`, expected-token-embedding
representation, zero-initialised so it starts as a no-op), disabled in every run so
far (`self_conditioning: false`). Needs question 7 answered first to have a baseline
to reduce.

---

## Findings that were not questions

Things learned that nobody asked about, recorded because they shape everything else.

- **`Qwen3_5TextModel.forward` accepts `attention_mask` as a dict keyed by layer type.**
  This is the cleanest possible injection point: bidirectional canvas masks without
  monkeypatching, subclassing or vendoring transformers.
- **Text-only execution needs no surgery.** `Qwen3_5ForCausalLM` already ignores
  `model.visual.*` and `mtp.*` on load.
- **Uniform corruption makes the denoising task ill-posed in a way masking does not.**
  Corrupted positions are unidentifiable, so the model must detect *and* repair, and
  copying scores `1−t` for free. This is a plausible reason published discrete-diffusion
  LMs favour absorbing-state masking.
- **A falling loss and a rising identity accuracy are compatible with learning nothing.**
  `lift_over_copy` and `corrupted_accuracy` exist because experiment 001 passed a
  weaker check.
- **Self-perplexity is a degenerate metric for adapter damage.** A model that repeats
  one token scores ppl 1.013 on its own output. Use a fixed reference text.
