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
