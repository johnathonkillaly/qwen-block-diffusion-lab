# Research questions

> Covers Acts I–III. From Act IV on, see [docs/RESULTS_INDEX.md](docs/RESULTS_INDEX.md) and
> [docs/PROJECT_SUMMARY_THROUGH_U5.md](docs/PROJECT_SUMMARY_THROUGH_U5.md).

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

---

# v0.2 questions — shared-weight bidirectional Gated DeltaNet

Branch `experiment/bidirectional-deltanet`, Qwen3.5-4B-Base on Unsloth/MLX.
Evidence: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) v0.2 section,
[docs/BIDIRECTIONAL_DELTANET.md](docs/BIDIRECTIONAL_DELTANET.md).

> **Global caveat, unchanged and now more important.** No held-out result exists in
> either version. v0.2 adds a 4B architecture that works and a one-batch test that
> **saturates**, which means the only discriminating evidence so far is a
> zero-training probe — and that probe is negative.

### 13. Can the same pretrained DeltaNet weights be evaluated in both directions?

**Answered — yes, mechanically, and it is cheap.**

24 of 32 layers wrapped, all sharing base tensors by object identity
(`test_weights_are_shared_not_duplicated`). Parameter count goes 4,205,751,296 →
4,216,111,104, and every added parameter is LoRA (3.1 M), timestep conditioner
(7.2 M) or fusion (0 for `mean`). The 4B backbone did not become 8B.

Cost 1.23× forward / 1.19× forward+backward — lower than the naive 2× because only
24/32 layers are DeltaNet and the reverse pass covers the canvas only.

### 14. Does the reverse recurrence actually carry information backwards?

**Answered — yes, and the leakage boundary holds.**

Per-layer probe: perturbing the last canvas token moves **0/15** earlier positions
under the native causal recurrence and **15/15** under bidirectional fusion, with
**0** prefix positions moving in either condition.

### 15. Is that backward information *useful* to the pretrained model?

**Answered for the untrained model — NO, not as position-aligned information.**

This is the sharpest negative result of v0.2. With no training at all, lowering the
fusion gate does reduce diffusion loss (8.18 → 7.33 at g = 0.25) and the
`forward_scaled_control` rules out mere attenuation. But
`shuffled_reverse_control` — the same reverse output with its canvas positions
randomly permuted — is **as good or better at every gate value** (6.89 vs 7.33).

Destroying positional alignment does not hurt, so the untrained gain is not
"position *i* learning about positions > *i*". It looks like an unstructured
perturbation that disrupts the model's autoregressive readout, which v0.1 showed is
the unadapted model's core problem under the unshifted x0 objective.

*Does not refute the hypothesis* — the premise is that adapters *learn* to use the
reverse direction, and the model has never seen a fused representation. But it
lowers the prior, and it means the untrained loss drop must not be quoted as
success.

*To move it:* Phase 3, and specifically re-running `shuffled_reverse_control`
**after** training. If shuffled stays competitive post-training, the reverse pass is
a perturbation, not information, and v0.2 should be killed.

### 16. Does bidirectional DeltaNet improve one-batch memorisation?

**Answered — no, because the test saturates at 4B.**

A (causal) and B (bidirectional, mean fusion) both reach loss ~1e-4, 100% identity
accuracy and 100% corrupted-position accuracy in 150 steps. Final metrics are
identical to the decimal. At 0.8B this test discriminated (v0.1 001 vs 004); at 4B
it does not.

Methodological consequence: **one-batch overfit is a plumbing check at 4B, not a
benchmark.** Only held-out evaluation can rank these configs.

### 17. Which fusion strategy should be preferred?

**Open.** Only untrained losses exist. `concat_proj` is an exact identity at init
(as designed), `scalar_gate` and `token_gate` sit ~0.05–0.17 nats below causal at
their 0.95 init, `mean` is far below but is the condition most confounded with the
shuffled control. A trained ranking requires Phase 3.

A cheap, decisive diagnostic once trained: **read the learned gate.** If
`scalar_gate` converges back to g ≥ 0.9, the model is choosing to ignore the reverse
path — that is a kill criterion, and it is recorded as one.

### 18. Can Unsloth Studio own a custom diffusion training loop?

**Answered — no.** `POST /api/train/start` has a closed `training_type` enum
(`LoRA/QLoRA`, `Full Finetuning`, `Continued Pretraining`) and no custom-script,
hook, callback or custom-loss field among its 72 parameters. Studio owns model
management, GGUF export and AR inference; the Unsloth **Python** engine (MLX + the
Qwen3.5 GatedDeltaNet custom VJP) runs the training. Details and the full capability
table: [docs/UNSLOTH_BACKEND.md](docs/UNSLOTH_BACKEND.md).

Also worth knowing: Studio's `/api/train/diffusion/*` endpoints are **image**
diffusion LoRA (datasets of images and captions), not discrete text diffusion.

### 19. What does Unsloth actually contribute on Apple silicon?

**Answered.** One component, and it is the right one:
`unsloth_zoo.gated_delta_vjp.patch_gated_delta()`, a memory-efficient custom VJP for
**Qwen3.5's** GatedDeltaNet with hand-written Metal chunk kernels, which recomputes
recurrent states in backward instead of holding all T intermediates. It is verified
active in every run header, and it serves both the forward and the reverse pass.

Triton, xformers and bitsandbytes never activate on this platform — which is *why*
v0.2 is BF16 base + BF16 LoRA rather than QLoRA.

## Updates to earlier questions

**Q3 (does DeltaNet causality prevent bidirectional diffusion?)** — the causality is
confirmed at 4B and now *worked around* rather than merely documented. Whether the
workaround helps is Q15, currently negative at initialisation.

**Q10 (Qwen3.5 → Qwen3.8-27B)** — one migration gap from v0.1 is now closed: 4B has
`linear_num_value_heads/num_key_heads = 2`, so the DeltaNet head-replication path
that was dead code at 0.8B is exercised. 27B has ratio 3 and untied embeddings, both
still untested.

---

## Phase 3 verdicts (held-out wikitext-2, 4B, Unsloth/MLX)

### 15 (revisited). Is the backward information useful *after training*?

**Answered — no.** The zero-training probe said the aligned reverse pass was
indistinguishable from a position-shuffled control. Training does not change that:

- aligned − shuffled at t ≥ 0.50 = **−0.41 points** after 600 steps, with fitted
  slopes of ~0 (+0.03 / −0.03 / −0.09 points per 100 steps at t = 0.50/0.75/0.90) and
  differences inside the 0.5–0.8 point within-arm evaluation noise;
- given a learned gate, all 24 layers move **away** from the reverse path
  (0.9500 → 0.9533).

The predicted signature — a gap that starts near zero and becomes positive — never
appears.

### 20. Does aligned dual recurrence beat the v0.1 causal hybrid on held-out text?

**Answered — no.** Arm B loses to arm A at all six noise levels
(6.84% → 5.44% corrupted-position accuracy at t = 0.50) and costs 1.26× wall clock.
The pre-registered continue criterion required ≥ +3.00 points; the observed margin is
**−1.06**. **Criterion not met; kill criterion fired.**

### 21. Does FLARE's block-end state readout do better in our harness?

**No, in this configuration** — arm D scores below A at every level. This says nothing
about FLARE: arm D is their *state-scheduling mechanism* transplanted under our
objective (pure diffusion, not `L_AR + L_diff`), our corruption (uniform, not masked)
and our data. See [docs/FLARE_COMPARISON.md](docs/FLARE_COMPARISON.md). Notably,
implementing it required **no kernel work** — `mlx_lm`'s `gated_delta_update` already
returns the block-end state, so the readout is one extra matmul.

### 22. Can this configuration learn generalising discrete denoising at all?

**Answered — no, and this is the most important absolute result.** Every arm
plateaued by step ~250 with held-out cross entropy ~6.57, corrupted-position accuracy
~6–7% *flat across all noise levels*, copy rate ~3.5%, and lift over copy ≈ −83% at
t = 0.10. That profile is a model that has stopped conditioning on the canvas and
emits a generic prefix-conditioned guess.

Rank-16 LoRA on 8 attention layers, 600 steps × batch 4 = 2,400 examples, is far too
little to relearn an unshifted readout convention on real text. v0.1 already showed
capacity is decisive (experiment 003 vs 004) and Phase 3 used the *lower*-capacity
setting.

*To move it:* a new experiment with new pre-registered criteria — more adapter
capacity (`full_attention + mlp`, plus `deltanet` so the reverse pass is adaptable at
all), many more steps, and plausibly `corruption: mask`. FLARE uses masking, and v0.1
experiment 001 showed uniform corruption makes the task ill-posed because corrupted
positions are unidentifiable. **That is a new experiment, not a re-roll of this one.**

### 3 (final update). Does DeltaNet causality prevent effective bidirectional diffusion?

The causality is real and confirmed at 4B. Two workarounds have now been built and
measured — our aligned dual recurrence and FLARE's block-end readout — and **neither
beat simply leaving the DeltaNet layers causal and relying on the periodic
full-attention layers** in a matched comparison. On current evidence the answer to
"does DeltaNet causality need fixing?" is: not at this scale, not with this objective,
and not before the data and capacity problems are addressed.
