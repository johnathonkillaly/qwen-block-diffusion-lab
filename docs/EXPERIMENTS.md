# Experiment log

Every run in this log is `overfit-one-batch` on **Qwen3.5-0.8B**, bf16, Apple M4 Max
(MPS), canvas 32, prefix 16, batch 2, seed 1234, `t ∈ [0, 0.5]`, deterministic first
batch. Reproduce with the config named in each entry.

Failures are recorded with the same weight as successes. Two of the four runs below
are failures, and one of them corrected a conclusion I had already written down.

---

## Summary table

| # | Config | LoRA targets | r | Steps | loss | id-acc | copy base | **lift** | **corr-acc** | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 001 | `qwen35_0_8b_lora_torch` | `full_attn` | 16 | 60 | 9.96 → 1.93 | 57.8% | 56.2% | **+1.6%** | **7.1%** | FAIL — copy collapse |
| 002 | `exp002_loss_on_corrupted` | `full_attn` | 16 | 60 | 6.95 → 3.55 | 14.1% | 53.1% | **−39.1%** | 13.3% | FAIL — mode collapse |
| 003 | `exp003_full_attn_mlp` | `full_attn_mlp` | 32 | 150 | 9.96 → **0.006** | **100%** | 67.2% | **+32.8%** | **100%** | **PASS** |
| 004 | `qwen35_0_8b_lora_torch` | `full_attn` | 16 | **150** | 9.96 → 0.087 | 98.4% | 67.2% | **+31.2%** | **95.2%** | **PASS** |

`lift` = identity accuracy − copy baseline. Only a positive lift means information is
being recovered. `corr-acc` is accuracy restricted to positions the noise actually
changed — the genuinely hard subset. Tail means (last quarter of steps) are in
`runs/*/summary.json`.

---

## 001 — Baseline: attention-only LoRA, 60 steps. **FAIL (copy collapse)**

```
loss 9.955 -> 1.931 (best 0.452)   identity 0% -> 57.8%   corrupted-position 7.1%
copy baseline 56.2%               lift +1.6% (tail -5.0%)  next-token 48.4% -> 6.5%
2,394,112 trainable (0.317%)      3.48 GB unified          199 tok/s
```

The loss fell 5x and identity accuracy climbed to 57.8%, and **both numbers are
worthless**: identity accuracy equals the copy baseline to the decimal, and accuracy on
corrupted positions is ~7%. The model learned to **echo its input**.

Two things did work, and they matter:

- `next_token_accuracy` collapsed from 48.4% to 6.5%, so the model genuinely abandoned
  autoregressive next-token prediction. The readout plumbing is correct.
- Gradients reached LoRA only; base weights stayed frozen.

**Why copying is attractive here.** Under *uniform* random-token corruption at noise
level `t`, a fraction `1−t` of positions are already correct and the model cannot tell
which ones. Echoing scores `1−t` for free with no understanding. This is a structural
difference from mask/absorbing-state diffusion, where corrupted positions are
*identifiable* by construction and copying a mask token scores zero. It is a plausible
reason most published discrete-diffusion LMs use masking.

**Instrumentation consequence.** The first version of the pass/fail check tested only
"did loss fall" and "is identity accuracy > 50%". It passed this run. `copy_baseline_accuracy`
and `lift_over_copy` were added specifically so this failure cannot be reported as a
success again.

---

## 002 — Score only corrupted positions. **FAIL (mode collapse)**

`diffusion.loss_on: corrupted`, `t ∈ [0.1, 0.5]`, otherwise identical to 001.

```
loss 6.951 -> 3.553   identity 14.1%   copy baseline 53.1%   lift -39.1% (tail -56.9%)
corrupted-position 13.3%   3.48 GB unified
```

Removing the copy reward did remove copying — and replaced it with something worse.
The model began emitting a few high-frequency tokens across the entire canvas
(`Rabbit Rabbit Rabbit …`), destroying clean positions along with corrupted ones.
Identity accuracy landed **39 points below** a pure copier.

Diagnosis: with `loss_on: corrupted`, nothing rewards leaving an uncorrupted token
alone, so the model stops distinguishing clean from corrupted at all. Corrupted-position
accuracy did improve slightly over 001 (13.3% vs 7.1%), so the shortcut removal
was doing *something*, but the objective as stated is not usable on its own.

Candidate follow-ups: a blended objective (weight corrupted positions higher rather
than exclusively), or `corruption: mask` where the distinction is observable.

---

## 003 — More adapter capacity, more steps. **PASS**

`lora.target_preset: full_attn_mlp`, `rank 32`, `alpha 64`, **150 steps**,
`loss_on: all`.

```
loss 9.955 -> 0.006 (best 0.001)   identity 100%   corrupted-position 100% (tail 94.7%)
copy baseline 67.2%                lift +32.8% (tail +24.2%)   next-token 0%
14,092,288 trainable (1.839%)      3.63 GB unified             152 tok/s
```

Complete memorisation of the batch, and genuinely *denoising* memorisation: every
corrupted position repaired, 33 points above the copy baseline.

Noise sweep on this checkpoint (`qdif reconstruct`), against the unadapted base model
on the same examples and the same seeded corruption:

| t | changed | base id-acc | base next-tok | **trained id-acc** | **trained corr-acc** | trained loss |
|---|---|---|---|---|---|---|
| 0.0 | 0% | 0.00% | 59.7% | **98.4%** | — | 0.017 |
| 0.1 | 9.4% | 0.00% | 51.6% | **100%** | **100%** | 0.001 |
| 0.5 | 50.0% | 0.00% | 27.4% | **100%** | **100%** | 0.005 |
| 0.9 | 95.3% | 0.00% | 1.6% | **98.4%** | **98.4%** | 0.072 |
| 1.0 | 100% | 0.00% | 0.0% | **98.4%** | **98.4%** | 0.099 |

The base model's row is the control that makes the readout convention visible: 0%
identity accuracy, high next-token accuracy, exactly as
[DIFFUSION_OBJECTIVE.md](DIFFUSION_OBJECTIVE.md) predicts.

Reconstruction stays ~98% even at `t = 1.0`, where **no** original information remains.
That is not generalisation — it is the correct signature of a memorised batch, which
can be regenerated from pure noise given the prefix. It is what the overfit test is
supposed to show, and it must not be read as evidence that the idea works on unseen
text.

---

## 004 — Ablation: was it capacity or step count? **PASS**

Experiment 003 changed two things at once. This run repeats **001's** configuration
(`full_attn`, rank 16, 2.4M trainable) for **150 steps**.

```
loss 9.955 -> 0.087   identity 98.4%   corrupted-position 95.2% (tail 68.1%)
copy baseline 67.2%   lift +31.2% (tail +16.5%)   3.48 GB unified   195 tok/s
```

**This corrects the conclusion drawn from experiment 001.** Attention-only LoRA at
rank 16 *can* learn to denoise; 60 steps was simply too few. Copy collapse is a
**transient state the optimisation passes through**, not the terminal local optimum I
first recorded it as.

Capacity still helps, clearly and measurably:

| | 004 (`full_attn`, r16) | 003 (`full_attn_mlp`, r32) |
|---|---|---|
| Trainable | 2.4M (0.32%) | 14.1M (1.84%) |
| Final loss | 0.087 | **0.006** |
| Corrupted-position accuracy (tail mean) | 68.1% | **94.7%** |
| Throughput | 195 tok/s | 152 tok/s |

So the ordering is: step count was the *decisive* factor for crossing from copying to
denoising; adapter capacity determines how far past that line the model gets, at ~25%
throughput cost.

---

## AR-vs-diffusion comparison (checkpoint 003)

```
qdif compare -c configs/exp003_full_attn_mlp.yaml \
  --checkpoint runs/exp003-full-attn-mlp-overfit/checkpoint \
  --prompt "It is a truth universally acknowledged, that a single man in possession of a good"
```

| mode | output | ref-ppl | tok/forward |
|---|---|---|---|
| AR, adapters **off** | `fortune, who is not a man of the world, is a man of the world.` | **8.06** | 1.00 |
| AR, adapters **on** | `fortune fortune fortune fortune fortune …` | **115.87** | 1.00 |
| Diffusion (8 steps) | `fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his of entering a neighbourhood, this truth` | — | **4.00** |

Three findings:

1. **The AR path survives intact when adapters are disabled** (research question 8/9,
   first half). Same weights, one boolean.
2. **The AR path is destroyed when diffusion adapters are active** — reference
   perplexity degrades **14.4x**. And the failure is *interpretable*: the adapter was
   trained to emit the token at position *i*, so run autoregressively (where position
   *i* should predict *i+1*) it repeats the current token forever. The two modes are
   not simultaneously available; they must be toggled.
3. **Diffusion decoding produced 32 tokens in 8 forwards** (4.00 tokens/forward) and
   near-perfectly reconstructed the memorised passage. On *memorised* content — this
   says the sampler works, not that quality holds on unseen text.

A note on the wall-clock numbers: the first `compare` run showed AR at 18 tok/s and
diffusion at 50 tok/s, but that reflected warm-up. Warm, AR ran at 45 tok/s vs
diffusion 51 tok/s. **`tokens_per_forward` is the structural claim; wall-clock at this
scale is noise.** Do not quote the speedup.

---

## What is not established

- **Nothing about generalisation.** Every result here is memorisation of two examples.
  The harness has never been trained on a corpus.
- The noise sweep at `t = 1.0` reflects memorisation, not a language prior recovering
  structure. Research question 6 is untouched.
- No result at 4B or 27B.
- No self-conditioning result (`self_conditioning: false` throughout).
- No result on whether DeltaNet layers *need* adaptation — 003 and 004 both adapted
  full-attention layers only, and both worked, which is suggestive for research
  question 4 but is not a controlled comparison against a `deltanet` preset.

## Next experiments, in priority order

1. **Train on a corpus, not one batch.** Everything above is a plumbing proof.
   `data.source: hf`, `max_steps` in the thousands, then re-run the noise sweep on
   *held-out* text. This is the first result that would mean anything.
2. **`corruption: mask` vs `uniform`, matched otherwise.** Directly tests the copy-shortcut
   explanation from 001/002.
3. **LoRA target ablation** across `full_attn`, `deltanet`, `deltanet_gates`,
   `all_attn` at fixed rank and steps — research question 2 and 4.
4. **`bidirectional_canvas: false` control.** If the causal control matches the
   bidirectional run, the 6 full-attention layers are not contributing what the design
   assumes.
5. Rank sweep 8/16/32/64 (research question 5); self-conditioning (12); denoising-step
   sweep 1/2/4/8/16/32 (7).

---
---

# v0.2 — Bidirectional Gated DeltaNet on Qwen3.5-4B (Unsloth/MLX)

Appended, not rewritten. The v0.1 false positive and its correction above remain
part of the record.

**Backend change.** v0.2 runs on Unsloth, which on Apple silicon is an MLX stack
(`DEVICE_TYPE == "mlx"`). See [UNSLOTH_BACKEND.md](UNSLOTH_BACKEND.md). The v0.1
torch numbers above are **historical reference only** — different framework,
different model. The controlled comparison is ablation **A vs B**, both inside MLX
on the same 4B checkpoint, differing by one boolean.

**Model.** `unsloth/Qwen3.5-4B-Base`, BF16, from `/Volumes/SHUTTLE`. 32 layers,
8 full-attention at [3,7,11,15,19,23,27,31], 24 Gated DeltaNet, hidden 2560, vocab
248,320, tied embeddings, `linear_num_value_heads/num_key_heads = 2` — so the
DeltaNet head-replication path that was **dead code at 0.8B** is live here.

## Hypothesis

A pretrained causal Gated DeltaNet can provide useful bidirectional diffusion
representations if the *same* weights are evaluated forward and reversed over the
canvas and the aligned directional representations are fused. Full design and the
fusion-point justification: [BIDIRECTIONAL_DELTANET.md](BIDIRECTIONAL_DELTANET.md).

## Phase 1 — architecture smoke test: **12/12 PASS**

`qdif smoke-bidir -c configs/v02_B_mean_fusion.yaml`

| # | check | result |
|---|---|---|
| 1 | 4B loaded via Unsloth/MLX | 4,205,751,296 params, 8.41 GB bf16, 0.9 s; `Unsloth: Patched GatedDeltaNet with memory-efficient custom VJP.` |
| 2 | architecture discovered at runtime | 32 layers, 8 full-attn / 24 DeltaNet, repeat factor 2, vision tower not loaded |
| 3 | forward DeltaNet output | `(1,24,32,128)` bf16, finite |
| 4 | reverse DeltaNet output | `(1,16,32,128)`, finite, `max|H_r − H_f| = 4.83e-02` |
| 5 | position alignment | `flip(flip(x)) == x`; perturbing `canvas[0]` leaves `H_r[-1]` **exactly** unchanged |
| 6 | fused output | canvas moves `1.90e+00`; prefix moves **0.000e+00** |
| 7 | shared parameter identity | id match across all 24 wrapped layers; total 4,216,111,104 = base + lora 3,145,728 + conditioner 7,214,080 + fusion 0 |
| 8 | bidirectionality probe | see below |
| 9 | finite diffusion loss | 10.890517 |
| 10 | gradients only where expected | 34 nonzero (32 `lora_b` + 2 conditioner), 34 zero (`lora_a` at init — expected), **0 base tensors touched** |
| 11 | peak unified memory | 9.59 GB (unified, not VRAM); active 8.45 GB |
| 12 | runtime ratio | 1.23× forward, 1.19× forward+backward |

### Directionality probe (layer 0, the flagship diagnostic)

| condition | earlier positions affected | mean L2 | max L2 | cosine change | prefix affected |
|---|---|---|---|---|---|
| native causal DeltaNet | **0 / 15** | 0.0000e+00 | 0.0000e+00 | 3.58e-08 | 0 |
| bidirectional DeltaNet | **15 / 15** | 2.1886e-02 | 7.2176e-02 | 4.66e-05 | 0 |

All three required properties hold: the recurrence really is causal by construction,
the reverse pass really does carry information backwards, and the leakage boundary
holds in both conditions.

## Phase 2 — one-batch overfit: **both pass, and the test is saturated**

150 steps, batch 2, canvas 32, prefix 16, t ∈ [0, 0.5], LoRA r=16 on full-attention
q/k/v/o only, 10,359,808 trainable (0.246%).

| | A — causal (v0.1 architecture) | B — bidirectional, mean fusion |
|---|---|---|
| first loss | 9.4776 | **7.6199** |
| final loss | 0.00011 | 0.00008 |
| identity accuracy | 3.1% → **100%** | 1.6% → **100%** |
| corrupted-position accuracy | **100%** (tail mean 100%) | **100%** (tail mean 100%) |
| lift over copy | +15.6% (tail +23.6%) | +15.6% (tail +23.6%) |
| next-token accuracy | 0.0% | 0.0% |
| peak unified memory | 10.16 GB | 10.41 GB |
| throughput | 263 tok/s | 214 tok/s |
| s/step | 0.243 | 0.299 |

**The one-batch test cannot discriminate A from B at 4B.** Both saturate: loss ~1e-4,
100% corrupted-position accuracy, identical lift. Final metrics are bit-identical
because both perfectly memorise the same batch. This is itself a methodological
result — at 0.8B the same test was discriminative (v0.1 experiments 001 vs 004);
at 4B it is not, and only held-out evaluation can decide anything.

The one real difference is the **initial** loss: 7.62 (B) vs 9.48 (A). At step 0 the
LoRA and conditioner are zero-initialised, so the fusion is the only difference. That
looked like the first positive signal — so it was tested properly.

## The measurement that matters so far: **negative for the hypothesis**

`qdif compare-fusion`. Zero training. 2 batches × 2, canvas 32, identical seeded
corruption in every condition, t = 0.5.

```
condition                      g=1.0    g=0.75     g=0.5    g=0.25     g=0.0
scalar_gate                   8.1838    7.8789    7.5729    7.3256    7.4303
forward_scaled_control        8.1838    8.1415    8.1928    8.4278   13.6508
shuffled_reverse_control      8.1807    7.8588    7.4774    6.8908    7.0739
```

Untrained loss by fusion, across noise levels:

| condition | t=0.1 | t=0.25 | t=0.5 | t=0.75 | t=0.9 |
|---|---|---|---|---|---|
| A_causal | 11.8233 | 9.6504 | 8.1838 | 8.8646 | 9.7234 |
| B_mean | 8.5182 | 7.8622 | 7.5729 | 7.9900 | 9.0420 |
| B_scalar_gate (g=0.95) | 11.7761 | 9.5916 | 8.0994 | 8.6925 | 9.5740 |
| B_concat_proj (identity init) | 11.8233 | 9.6504 | 8.1838 | 8.8646 | 9.7234 |

Two conclusions, and the second is the important one:

1. **Attenuation is ruled out.** `forward_scaled_control` (`g·H_f`, no reverse
   information at all) stays flat and collapses at g = 0. Merely scaling the
   pretrained forward path down does not reproduce the gain.
2. **But the gain is not position-aligned information.**
   `shuffled_reverse_control` — the same reverse output with canvas positions
   randomly permuted — is **as good or better at every gate value** (6.89 vs 7.33 at
   g = 0.25).

If destroying `H_r`'s positional alignment does not hurt, then at initialisation the
benefit is not "position *i* learning about positions > *i*". The most likely
mechanism is unstructured: adding a correctly-scaled, largely decorrelated signal to
the DeltaNet output disrupts the model's autoregressive readout — and v0.1 established
that AR-ness is exactly the unadapted model's problem under the unshifted x0 objective.

**This is a negative result for the v0.2 hypothesis at initialisation.** It does not
refute it: the whole premise is that adapters *learn* to exploit the reverse
direction, and the model has never seen a fused representation. It does mean the
lower untrained loss must **not** be reported as evidence that bidirectional DeltaNet
works, and it makes the held-out trained comparison the only thing that can decide.

## Kill / continue criteria — recorded BEFORE the held-out run

Fixed here so they cannot be moved after seeing the result. Phase 3: train on a small
real corpus, evaluate on entirely unseen text at t ∈ {0.10, 0.25, 0.50, 0.75, 0.90,
1.00}, matched steps / seeds / data, A vs B vs C vs D.

**Continue v0.2** (bidirectional DeltaNet earns its place) if *both*:
- held-out **lift over copy** for the best bidirectional config exceeds A by
  **≥ +3 percentage points** averaged over t ≥ 0.50, and
- held-out **corrupted-position accuracy** exceeds A at **t = 0.50 and t = 0.75**
  individually, not merely on average.

**Continue v0.1 instead** (periodic full attention is enough) if:
- A is within **1 point** of the best bidirectional config on held-out lift at
  t ≥ 0.50. The 1.19–1.23× compute and the extra machinery are not worth a tie.

**Kill bidirectional DeltaNet** if any of:
- held-out lift for every bidirectional config is **≤ A** at t ≥ 0.50, or
- the learned gate converges back to g ≥ 0.9 (the model choosing to ignore the
  reverse path), or
- training destabilises — gradient norm or loss diverging where A is stable, or
- `shuffled_reverse_control` remains competitive with real fusion **after** training,
  which would show the reverse pass is a perturbation and not information.

The last one is the sharpest test and it must be run at Phase 3, not skipped.

## Not established in v0.2

- **Nothing on held-out generalisation.** Phase 3 has not been run.
- No result for ablations C (scalar gate), D (DeltaNet LoRA) or E (token gate) under
  training — only their untrained losses.
- No sampler / generation results at 4B; no step-count sweep.
- AR-preservation was not re-measured at 4B (v0.1 measured a 14.4× reference
  perplexity degradation with adapters on, restored exactly by toggling them off).
- The reverse pass starts from a zero recurrent state mid-document, a regime the
  pretrained recurrence never saw. Whether that is the limiting factor is untested.

---

## Phase 3 — held-out denoising on wikitext-2. **v0.2 continue criterion FAILED; kill criterion FIRED.**

`unsloth/Qwen3.5-4B-Base`, BF16 base + BF16 LoRA r=16 on full-attention q/k/v/o only
(3,145,728 LoRA + 7,214,080 timestep conditioner = 10,359,808 trainable, 0.246%),
600 steps, batch 4, canvas 32, prefix 16, `t ~ U(0,1)`, seed 1234.

**Data.** `Salesforce/wikitext:wikitext-2-raw-v1`. Train = 3,000 windows from the
`train` split; held-out = 128 windows from the `test` split — a **different set of
articles**, so the claim is document-disjoint, not merely window-disjoint. Held-out
corruption uses a fixed seed (777) reset per noise level, so every arm sees
byte-identical noise.

All arms are identical except the DeltaNet mechanism.

### Held-out corrupted-position accuracy at step 600

| arm | t=0.10 | t=0.25 | t=0.50 | t=0.75 | t=0.90 | t=1.00 |
|---|---|---|---|---|---|---|
| **A** causal (v0.1) | **9.92%** | **6.51%** | **6.84%** | **6.70%** | **7.32%** | **7.08%** |
| **B** aligned fusion (v0.2) | 7.10% | 6.13% | 5.44% | 5.76% | 6.55% | 5.86% |
| **C** shuffled control | 7.97% | 6.07% | 6.18% | 5.87% | 6.92% | 6.64% |
| **D** FLARE block-end readout | 6.93% | 5.58% | 5.36% | 5.61% | 5.79% | 5.57% |

### Held-out cross entropy at step 600

| arm | t=0.10 | t=0.25 | t=0.50 | t=0.75 | t=0.90 | t=1.00 |
|---|---|---|---|---|---|---|
| A | **6.565** | **6.564** | **6.570** | **6.577** | **6.563** | **6.571** |
| B | 6.601 | 6.606 | 6.610 | 6.608 | 6.610 | 6.608 |
| C | 6.574 | 6.576 | 6.576 | 6.598 | 6.575 | 6.577 |
| D | 6.611 | 6.628 | 6.663 | 6.666 | 6.660 | 6.665 |

`t = 1.00` measures the **prior, not denoising**: at full corruption the particular
held-out continuation is not identifiable from the canvas, so a model can only produce
*a* plausible continuation. Exact reconstruction was 0.0% for every arm at every noise
level, as it should be on unseen text.

### The measurement the hypothesis lives or dies on

`corrupted_accuracy(aligned) − corrupted_accuracy(shuffled)`, tracked every 50 steps:

| t | mean diff | std | slope per 100 steps | step 0 | step 600 |
|---|---|---|---|---|---|
| 0.50 | **−0.77%** | 0.55% | +0.03% | −1.00% | −0.74% |
| 0.75 | **−0.61%** | 0.53% | −0.03% | −0.07% | −0.11% |
| 0.90 | **−0.58%** | 0.42% | −0.09% | +0.22% | −0.36% |

**It starts near zero and stays near zero.** The hypothesis predicted it should become
positive as the model learns to exploit the positional structure of the reverse
recurrence. The fitted slopes are ~0 and two of three are negative. For scale, the
within-arm step-to-step evaluation noise on arm A is 0.5–0.8 percentage points — the
*same magnitude as the differences* — so B and C are statistically indistinguishable
throughout, and B never gets ahead.

### Cost

| arm | tok/s | s/step | peak unified | wall |
|---|---|---|---|---|
| A causal | **191.1** | 0.670 | 11.65 GB | 565 s |
| B aligned | 151.1 | 0.847 | 12.14 GB | 698 s |
| C shuffled | 132.3 | 0.968 | 12.15 GB | 814 s |
| D FLARE | 156.6 | 0.817 | 11.95 GB | 700 s |

### Verdict against the pre-registered criteria (unmodified)

- Mean lift over t ≥ 0.50: **B −22.17% vs A −21.11% → margin −1.06 points.**
  Required ≥ +3.00. **NOT MET.**
- B beats A at both t = 0.50 and t = 0.75 individually: **False** (B loses at every
  level).
- Aligned − shuffled at t ≥ 0.50: **−0.41 points.** The kill criterion — "shuffled
  remains competitive with real fusion after training" — **FIRES**.

**Conclusion: the aligned dual-recurrence mechanism does not earn its 1.26× wall-clock
cost.** On this evidence the reverse pass supplies a perturbation, not position-aligned
information, exactly as the zero-training probe in
[BIDIRECTIONAL_DELTANET.md](BIDIRECTIONAL_DELTANET.md) predicted.

FLARE's block-end readout (D) also failed to beat A here. That is **not** evidence
against FLARE — arm D is a mechanism transplant under our objective, our uniform
corruption and our data, none of which are FLARE's; see
[FLARE_COMPARISON.md](FLARE_COMPARISON.md).

### The honest caveat: this comparison had low power

Every arm is in a degenerate regime and the run must not be read as "denoising works,
and mechanism X is best". Specifically:

- Held-out **lift over copy is strongly negative at low noise** for every arm (≈ −83%
  at t = 0.10). A pure copier would score ~90% identity accuracy at t = 0.10; these
  models score ~7%. They are *destroying clean tokens*.
- **Copy rate is ~3.5%** and corrupted-position accuracy is ~6–7% and **essentially
  flat across all noise levels**, which is the signature of a model that has stopped
  conditioning on the canvas and is emitting a generic prefix-conditioned guess.
- Training **plateaued by step ~250**: cross entropy fell 8.29 → 6.55 and then stopped.

So the arms were compared while all of them were weak. Two things follow, and both
matter:

1. The **relative** comparison is still valid and is what the criteria were written
   against — matched data, matched seeds, matched steps, matched capacity. B lost.
2. The **absolute** result says this configuration cannot learn generalising discrete
   denoising: rank-16 LoRA on 8 attention layers, 600 steps × batch 4 = 2,400 examples,
   is far too little for a model to relearn an unshifted readout convention on real
   text. v0.1 already showed capacity matters (experiment 003 vs 004), and this run
   used the *lower*-capacity setting.

A properly powered rerun would need, roughly in order of expected effect: more adapter
capacity (`full_attention + mlp`, and `deltanet` for arms B/C/D so the reverse pass is
adaptable at all), many more steps, and plausibly `corruption: mask` — v0.1 experiment
001 showed uniform corruption makes the task ill-posed because corrupted positions are
unidentifiable, and FLARE uses masking. **That rerun is a new experiment with new
pre-registered criteria, not a re-roll of this one.** The criteria above fired and are
not being reopened.

Note also FLARE's own reported finding, which anticipated this: "the attainable
transfer quality is governed primarily by the transfer data mix rather than by the
algorithmic recipe." Our axis — recurrence and mask design — is the one they found had
less headroom.

### Arm E — the gate test: the model chooses to use *less* reverse information

Arms B/C/D used `mean` fusion, which has **no gate**, so the gate kill criterion could
not be evaluated from them. Arm E repeats arm B with `fusion: scalar_gate`
(`gate_init = 0.95`, one learned scalar per DeltaNet layer, 24 trainable scalars),
everything else identical.

`g = 1.0` is exactly the causal path; `g = 0.5` is mean fusion; lower `g` means more
reverse information.

| step | gate mean | gate min | gate max |
|---|---|---|---|
| 0 | 0.9500 | 0.9500 | 0.9500 |
| 150 | 0.9518 | 0.9498 | 0.9529 |
| 300 | 0.9526 | 0.9499 | 0.9540 |
| 450 | 0.9530 | 0.9497 | 0.9547 |
| **600** | **0.9533** | 0.9497 | **0.9553** |

**The gate moved monotonically *away* from the reverse path** — 0.9500 → 0.9533 — and
not one of the 24 layers moved meaningfully towards it (min 0.9497). Given a free
parameter to decide how much of the reverse recurrence to admit, the model reduced it.

Held-out at step 600: corrupted-position accuracy 6.00% / 6.15% / 6.99% at
t = 0.50 / 0.75 / 0.90 — below arm A (6.84% / 6.70% / 7.32%), at 123 tok/s vs A's 191.

**Pre-registered kill criterion "the learned gate converges back to g ≥ 0.9" — FIRES**
(g = 0.9533).

Two caveats stated for fairness: the gate drift is small in absolute terms (+0.0033),
and it is a single scalar per layer trained for 600 steps in the same low-power regime
as the rest of Phase 3. But its *direction* is unambiguous and consistent across all 24
layers, and it agrees with every other measurement in this phase.

### Phase 3 summary

| arm | mechanism | t=0.50 | t=0.75 | t=0.90 | tok/s | verdict |
|---|---|---|---|---|---|---|
| **A** | causal + bidirectional full attention (v0.1) | **6.84%** | **6.70%** | **7.32%** | **191.1** | best on every axis |
| B | aligned dual recurrence (v0.2) | 5.44% | 5.76% | 6.55% | 151.1 | loses, costs 1.26× |
| C | shuffled-reverse control | 6.18% | 5.87% | 6.92% | 132.3 | indistinguishable from B |
| D | FLARE-style block-end readout | 5.36% | 5.61% | 5.79% | 156.6 | loses (not a FLARE baseline) |
| E | aligned + learned scalar gate | 6.00% | 6.15% | 6.99% | 123.1 | loses; gate moves away from reverse |

**v0.2's hypothesis is not supported.** Three independent measurements agree: the
aligned reverse recurrence does not beat its own position-shuffled control, does not
beat the causal baseline, and is actively down-weighted when the model is given the
choice.
