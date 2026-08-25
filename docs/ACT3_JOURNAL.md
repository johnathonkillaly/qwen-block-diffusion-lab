# Act III research journal

Chronological. Failed runs and wrong assumptions stay in.

---

## Hypothesis

Acts I–II trained a *pure* diffusion objective under uniform random-token corruption
and ended in a canvas-ignoring regime (Act II Phase 3: held-out masked-position
accuracy ~6–7%, essentially flat in `t`). Act III tests whether reproducing FLARE's
recipe more faithfully produces a *healthy* denoiser:

1. combined `L_total = L_AR + λ·L_diff`,
2. absorbing-state MASK corruption (corrupted positions identifiable),
3. complementary mask views (every canvas token supervised once per step),
4. Gated DeltaNet block-end state scheduling,
5. more adapter capacity (r32 across attention + MLP + DeltaNet).

Criteria pre-registered in [ACT3_CRITERIA.md](ACT3_CRITERIA.md) **before** run 1.

---

## Setup decisions

**Mask token = 248,319.** Qwen3.5's model vocab (248,320) exceeds its tokenizer vocab
(248,044). Every embedding row from ~248,070 up shares norm 0.445587 — initialised,
never trained. The last row is therefore unreachable by the tokenizer (cannot collide
with real text) and carries no pretrained semantics.

**Corpus = WikiText-103-raw-v1** (rev `b08601e0…`), 315 MB, 1.8M paragraphs. Packed
into contiguous 192-token windows (64 prefix + 128 canvas): 20,000 train windows =
3.84M tokens; validation split held out, test untouched.

**Milestone check passed 15/15** before any training:
`L_AR` 2.549, `L_diff` 8.539, 63,050,752 trainable (1.477%), canvas-conditioning
L1 1.5046, peak 21.9 GB unified, 2.837 s/step (3 forwards/step).

---

## Run 1 — `act3-main`, 2500 steps planned. **ABORTED at step 125.**

Stop condition fired: *"canvas_l1 falls below 0.1 at any evaluation (canvas ignoring)"*.

| step | t=0.10 masked | t=0.50 masked | canvas L1 | AR ppl | next-tok |
|---|---|---|---|---|---|
| 0 | 1.4% | 1.6% | **1.517** | 8.91 | 56.2% |
| 125 | 2.4% | 3.6% | **0.016** | 5.38 | 60.6% |

The AR path was *fine* — perplexity improved 8.91 → 5.38 and next-token accuracy rose,
so the `L_AR` term was doing its job. The diffusion path had already stopped reading
the canvas.

### Diagnosis: the timestep conditioner ate the canvas

Measured directly from `ckpt-125`:

```
mean token-embedding norm             : 0.6611
timestep-conditioner bias norm @ init : t=0.1 0.0000  t=0.5 0.0000  t=0.9 0.0000
timestep-conditioner bias norm @ 125  : t=0.1 14.8998  t=0.5 13.1569  t=0.9 13.5280
RATIO bias / token-embedding          : 19.9x
```

The conditioner adds its output to `inputs_embeds` at canvas positions. At a **20×**
ratio to the token embeddings it simply swamps them, so the canvas *contents* stop
mattering — exactly what the probe measured.

Two details confirm it was a degenerate solution rather than useful conditioning:

- the learned bias is nearly **identical** at t=0.1 / 0.5 / 0.9 (14.90 / 13.16 / 13.53),
  so it was not encoding the noise level at all — it had found "add a large constant to
  canvas positions";
- its gradient dominated everything else from the start. At the milestone check,
  `timestep_conditioner.fc2.weight` had grad norm **1.685e+03** against ~1.4e+01 for the
  largest LoRA tensor. With global-norm clipping at 1.0, the conditioner consumed most
  of every update.

This was a **carry-over bug from Acts I–II**, where uniform corruption made the noise
level unobservable so *some* timestep signal was needed. It should have been removed
when the corruption changed.

### Fix

1. **`timestep_conditioning: false` for Act III.** Under absorbing-state masking the
   noise level is directly observable from the `[MASK]` count, and FLARE's formulation
   carries no timestep embedding. Removing it is both the fix and the more faithful
   choice. Drops trainable parameters 63.1M → 55.8M.
2. **Norm bound as a permanent safety rail.** `TimestepConditioner` now hard-clips its
   output norm to `max_relative_norm` (default 0.5) × the model's own mean
   token-embedding norm, measured at construction. The failure cannot recur silently
   for anyone who does enable it. `configs/act3_abl_timestep.yaml` keeps the bounded
   variant as an ablation.

### What this cost, and what it bought

~7 minutes of compute and one aborted run. The canvas-conditioning probe — added
because the user insisted it be a first-class metric — caught it at step 125 instead of
after a two-hour run ending in another inexplicable flat-accuracy result. Act II's
Phase 3 pathology may well have had a related cause that we never diagnosed, because
that probe did not exist yet.

---

## Run 2 — `act3-main`, 2500 steps, **completed**

Identical to run 1 except `timestep_conditioning: false`. 148 minutes, 83.3 canvas
tok/s, peak **21.12 GB unified**, 55,836,672 trainable (1.31%).

```
L_AR    2.6063 -> 1.8262
L_diff  8.4935 -> 3.1787
```

### Held-out (WikiText-103 **validation**, never trained on), step 2500

| t | masked-acc | visible-pres | canvas L1 | diffusion CE | copy rate | predicts [MASK] |
|---|---|---|---|---|---|---|
| 0.10 | **58.17%** | 2.77% | 1.949 | 2.036 | 2.49% | 0.00% |
| 0.25 | **51.76%** | 2.41% | 1.926 | 2.389 | 1.81% | 0.00% |
| 0.50 | **41.80%** | 2.73% | 1.856 | 3.065 | 1.37% | 0.00% |
| 0.75 | **24.15%** | 2.34% | 1.590 | 4.533 | 0.59% | 0.00% |
| 0.90 | **12.99%** | 1.44% | 1.256 | 5.830 | 0.15% | 0.00% |
| 1.00 | 5.37% | n/a | 0.000 | 6.676 | 0.00% | 0.00% |

Step-0 baseline was 1.4–1.6% masked accuracy, **flat**, with diffusion CE 8.2–9.4.
AR reference perplexity **8.91 → 5.11**; next-token accuracy 56.2% → 61.8%.

`canvas_l1 = 0.000` at t=1.00 is correct, not a pathology: with the whole canvas
masked there are no visible tokens for the probe to scramble.

### Learning dynamics

| step | t=0.10 | t=0.50 | t=0.90 | canvas L1 (t=0.50) | AR ppl |
|---|---|---|---|---|---|
| 0 | 1.4% | 1.6% | 0.8% | 1.623 | 8.91 |
| 125 | 38.0% | 25.5% | 9.2% | 1.613 | 5.38 |
| 500 | 49.5% | 34.3% | 10.9% | 1.761 | 5.16 |
| 1000 | 51.9% | 39.4% | 12.3% | 1.855 | 5.20 |
| 1500 | 55.3% | 41.0% | 12.4% | 1.833 | 5.11 |
| 2000 | 57.7% | 41.3% | 12.6% | 1.845 | 5.11 |
| 2500 | 58.2% | 41.8% | 13.0% | 1.856 | 5.11 |

Most of the transition happens in the first ~125 steps, then slow improvement to a
plateau by ~1500. Canvas conditioning *rises* (1.62 → 1.86) rather than collapsing —
the opposite of run 1.

### Verdict against the pre-registered criteria

| # | criterion | result | |
|---|---|---|---|
| 1 | noise dependence, gaps ≥ 3 pts | 58.2% > 41.8% > 13.0%, gaps 16.4 / 28.8 | **PASS** |
| 2 | masked-acc(0.10) ≥ 25% | 58.2% | **PASS** |
| 3 | visible preservation ≥ 80% | 2.8% / 2.7% | **FAIL** |
| 4 | masked-acc(0.50) ≥ 2× step 0 | 26.8× | **PASS** |
| 5 | canvas L1 ≥ 0.5 | 1.856 | **PASS** |
| 6 | AR ppl ≤ 3× step 0 | 5.11 vs limit 26.73 | **PASS** |
| 7 | held-out text | validation split | **PASS** |

**6 of 7 pass. By the written outcome definition — "FAILURE if 1, 3 or 5 fails" — this
run is a FAILURE.** The criteria are not being rewritten.

### Why criterion 3 was mis-specified (analysis, not a goalpost move)

Criterion 3 was written against the Act II pathology, where the model *overwrote every
canvas position* and destroyed known-correct tokens. Under absorbing-state masking that
threat does not exist in the same form, for two structural reasons:

1. **The objective never supervises visible positions.** Both noisy views compute
   cross entropy only at the positions masked *in that view's input*
   (`M` for view 1, `M^c` for view 2). A visible token is never a target. This is
   FLARE's construction, not an accident of ours — their `L_diff` sums over `ℓ ∈ M_b`.
   So visible-token accuracy staying at its ~2.5% baseline is the *expected* outcome,
   not degradation.
2. **The sampler never uses those predictions.** Absorbing-state decoding is monotone:
   once a position is committed it is frozen and never rewritten. Predictions at visible
   positions are discarded by construction.

Evidence that it does not gate quality: infilling on unseen text works well (below)
while visible preservation sits at 2.7%.

The criterion was a reasonable guard when written and is a wrong measurement for this
objective. It is recorded as a **specification error on our part**, and the run is
reported as a by-the-letter failure with 6/7 passing.

### What the model can and cannot do

**Infilling with real context — works.** Held-out WikiText validation, 128-token canvas,
t = 0.25 (32/128 masked), masked accuracy 46.9%:

```
INPUT : ▁▁ ) long and weigh▁▁0 @.@ 7 – ▁▁▁▁.@ 2 kg ( 1 @▁▁ 5 – 4 @▁▁▁▁9▁▁ )▁▁ Like other
        crustace▁▁ , lobsters have a hard exoskeleton which they must shed▁▁▁▁ to grow▁▁
FILLED:  in ) long and weigh 0 @.@ 7 – 1 @.@ 2 kg ( 1 @.@ 5 – 4 @.@ 9 lb ) . Like other
        crustaceans , lobsters have a hard exoskeleton which they must shed it in to grow .
TRUTH :  in ) long and weigh 0 @.@ 7 – 2 @.@ 2 kg ( 1 @.@ 5 – 4 @.@ 9 lb ) . Like other
        crustaceans , lobsters have a hard exoskeleton which they must shed in order to grow ,
```

It recovers `in )`, `weigh`, the `@.@` numeric formatting, `lb )`, `Like other
crustaceans`, `moulting )`, and the exact phrase `young lobsters , but decreases to once
every ... years for`. That is real denoising on text it has never seen.

**Free generation from a fully-masked canvas — does not work.** 8 steps, 64-token
canvas, prompt "The rain had stopped by morning":

```
, and the the the the the sun the sun was the the the the the moon was the the the the
stars , the the moon was the the the the sun was the the ...
```

High-frequency token collapse, with top-1 confidence only 0.059 → 0.263 across the
trace. This is consistent with the health table rather than a surprise: masked accuracy
at t = 1.00 is 5.37% and diffusion CE 6.676, barely better than a unigram prior.

The model learned **conditional infilling**, not **unconditional block generation**.
Plausible causes, untested: only ~640k canvas tokens of diffusion supervision; a single
128-token block conditioned on a 64-token prefix is a very long span to invent; errors
compound as the sampler commits its own low-confidence tokens; and `t ~ U(0.05, 1.0)`
spends most of its mass in a regime where there is almost no context to learn from.

### Cost

| | |
|---|---|
| Wall clock | 148.1 min |
| Throughput | 83.3 canvas tok/s, ~135 sequence tok/s |
| Peak memory | 21.12 GB unified |
| Diffusion supervision | 2500 × 2 views × 128 = 640,000 canvas tokens |
| AR supervision | 2500 × 2 × 191 = 955,000 tokens |
