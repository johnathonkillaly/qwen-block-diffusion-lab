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
