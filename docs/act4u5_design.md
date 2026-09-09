# Act IV-U5 — Train Long, Decode Short: design

Written during **U5-PREP**, with heavy execution blocked by an unrelated BoothGPT
pretraining job. Nothing in this document was informed by a U5 result, because no U5
training has run.

Pre-registered criteria: [`act4u5_preregistered_criteria.md`](act4u5_preregistered_criteria.md).

---

## 1. The question

Act IV-U4 concluded `K4 IS THE PRACTICAL FRONTIER`: K=4 is the best static *inference*
block size, because only three speculative slots ever pay for their 1.5 ms. It also
produced an observation nobody had pre-registered — that training at K=6/K=8 improved
**K=4** decoding more than 9600 further steps of K=4 training had.

U5 asks whether that is real:

> Under an equal training budget, does training the adapter against longer future
> blocks produce a better K=4 decoder than training directly at K=4?

Provisional name for the phenomenon, claimed as a label and not as established
literature: **cross-horizon distillation transfer**.

## 2. Why the U4 observation is not yet a result

The U4 comparison was, in its own session and paired by noise stream:

| K=4 decoding | accept | mean prefix | TPF | tok/s (median) | tok/s (mean) |
|---|---|---|---|---|---|
| `step-12800`, after 9600 K=4 steps | 0.3226 | 0.9679 | 1.4035 | 56.17 | 56.04 |
| `+3200` steps at K=8 | 0.3525 | 1.0574 | 1.4314 | 59.08 | 57.61 |
| `+1600` at K=6 then `+1600` at K=8 | 0.3484 | 1.0451 | 1.4314 | 58.60 | 57.74 |

Three reasons that is not yet evidence for the hypothesis:

1. **There was no matched control.** Nothing in U4 trained K=4 for the same 3200 extra
   steps. The comparison was against the *starting point*, so "3200 more steps of
   anything" is not excluded. U4-A's plateau makes it unlikely, but unlikely is not
   controlled.
2. **It was not pre-registered**, so the choice of which numbers to look at was made
   after seeing them.
3. **It rests on a fragile estimator.** The median over 15 prompts said +5.2%; the mean
   said +2.8%; the cost model, which knows only acceptance and cycle cost, predicted
   +2.8%. U5 must not repeat that — hence per-prompt pairing (§7).

U5 exists to supply the missing control and to decide the question before looking.

## 3. What is genuinely matched, and what cannot be

This is the part that decides whether U5 answers anything.

**Compute is matched by construction, not by tuning.** In `teacher.py` both forwards
are always `window - 1` tokens wide, whatever the block size: the student is
`prefix + noise tail` of total width `(W - L) + (L - 1) = W - 1`. So every arm pushes
**identical token counts** through the model per step. U4's measured cost confirms it:
1.400 s/step at K=4, 1.432 at K=6, 1.435-1.444 at K=8 — a ~2.5% spread, entirely from
the wider supervised slice, not from more sequence.

Matched across arms:

| | matched? | how |
|---|---|---|
| optimizer steps | yes | 3200 each |
| training examples and their order | yes | same `WindowSource`, same seed, same start position |
| tokens through the model | yes | `2 x B x (W-1)` per step, independent of `L` |
| learning-rate exposure | yes | constant 1e-5, no warmup on resume |
| starting adapter and AdamW moments | yes, byte-identical | every arm `--resume-from` the same directory (§5) |
| corruption realisation on shared positions | yes | right-aligned noise draw (§6) |
| wall clock | ~2.5% apart | reported, not corrected |

**Not matched, because it is the treatment:** the number of supervised future
positions. K=4 supervises 4 positions per step (3 adapted); K=8 supervises 8 (7
adapted). You cannot lengthen the training horizon without supervising more positions.

### 3.1 The confound U5-A cannot separate, stated in advance

Because of the above, a positive result is consistent with **two** mechanisms:

* **horizon** — the extra positions are *further out*, forcing the adapter to model
  dependence on predecessors it cannot see, which improves the near positions too;
* **density** — the extra positions are simply *more supervision per step*, and any
  extra supervised positions would have done as well.

U5-A cannot distinguish them, and will not claim to. The experiment that does is named
now so it cannot be invented afterwards: **train at K=8 but mask the loss to the first
four supervised positions**. Same block, same noisy context, same number of supervised
positions as K=4. If that arm keeps the gain, the mechanism is horizon; if it loses it,
the mechanism is density. That is U5-B, and it is out of scope here (brief §8).

## 4. Arms

Four, per brief §7. All resume from one checkpoint, all decode at K=4.

| arm | training horizon | note |
|---|---|---|
| **A_k4** | K=4 for 3200 steps | the control U4 never had |
| **B_k6** | K=6 for 3200 steps | |
| **C_k8** | K=8 for 3200 steps | the U4 observation, now controlled |
| **D_curr** | K=4 → K=6 → K=8, ~1067 steps each | IFM's curriculum shape |

Checkpoints at +400, +800, +1600, +2400, +3200 (absolute 13200 … 16000), because the
brief asks for transfer *curves*, not endpoints.

Arm D follows the released Uno curriculum, re-confirmed from source during U4:
`uno_3epoch_curriculum.yaml` runs six equal-token stages `2,4,6,8,12,16`, and
`training/trainer.py` shows a stage transition changes **only** the block size —
optimizer, adapter and LR schedule all carry over. Equal token budget per stage is
equal steps here, since sequence length and batch size are fixed. We reproduce the
*shape* at four orders of magnitude less data, and say so.

## 5. Shared starting point

`runs/u4a/step-12800` — the K=4-saturated adapter, which is exactly where U4's
horizon-specific divergence began. Verified with `scripts/u5_start_check.py`, whose
output is written to `results/act4u5/u5_start_check.json`:

```
step               12800
config hash        846fa021a8294cbe6d69a1af31b6445380ea801d88447325beb3f1bd183443a6
adapter sha256     8332986b8a411f589f8f05d2a2f067a60ae7d7e34d8c2261ec22a89ac5460750
adapter tensors    256  (combined 35ac2d460b7eeefb…)
optimizer sha256   e93eabcb67e9ed1202a21ba2265a21161e6b83d7710518dda25dbc16d0a5da7e
optimizer tensors  514
data position      12800 train draws, cursor 3200
```

Every arm passes `--resume-from` that directory, so identity of the starting state is a
property of the mechanism rather than something to verify afterwards. The same script
re-runs after training to confirm the arms **diverged** — two arms with identical final
adapters would mean the horizon changed nothing, which is far more likely to mean a
misconfiguration than a discovery.

Arms B/C/D change `block_size`, so their config hash differs from the checkpoint's and
the trainer prints a warning. That is correct and expected: the weights carry over, the
objective's horizon does not.

## 6. Making block size the only variable

Two changes were needed, both pinned by `tests/test_uno_horizon_pairing.py`.

### 6.1 Right-aligned corruption

Draft rows for block size `L` occupy absolute positions `W-L … W-2`, so **every** arm's
rightmost draft row is at `W-2` and the arms overlap on their last positions. But an
MLX draw of shape `(B, 3)` is not a sub-array of one of shape `(B, 7)` — the row stride
differs — so under one key each arm still saw a different corruption realisation. Two
arms would then differ in their noise *as well as* their horizon.

`build_uno_batch(..., noise_align_width=A)` draws at width `A-1` and keeps the
rightmost `L-1` columns. Arms at K=4, K=6 and K=8 then see **identical noise tokens and
identical keep decisions at every position they share**. The per-sequence corruption
rate already had shape `(B, 1)` and needed no alignment. `None` reproduces U3/U4 exactly,
so those runs still replay.

### 6.2 Curriculum staging on a resumed run

`block_size_at` divided the *absolute* step axis into stages. A curriculum arm resumed
at 12,800 of 16,000 would therefore have spent every step in the final stage — a
curriculum arm that trains only at K=8 and is not a curriculum at all. Stages now
divide the resumed span, and a test asserts each stage gets equal steps.

## 7. Pairing, and why it is the whole design

U4's calibration measured, for K=4 tok/s, a **between-session sd of 2.36** against a
**within-unit repeat noise of 0.15**. Almost all of the spread is prompt difficulty and
the noise realisation — both of which cancel when the same prompt with the same draft
noise is compared across arms.

So `u5-eval` keeps per-unit data that `u4-eval` aggregated away: the tok/s, TPF and
accepted counts of every prompt, and the accepted prefix of every held-out row. Arms
are then compared by a **paired bootstrap over differences** (10,000 resamples), which
reports a 95% interval on the mean of `arm − control`.

`tests/test_uno_paired_stats.py` includes the case this exists for: a consistent +1
effect buried in a between-prompt spread of 100 is invisible unpaired and exact paired;
and a ±10 inconsistent effect with a zero mean is correctly reported as not significant.

## 8. Evaluation

Two sessions, both with an in-session AR baseline.

* **Primary** — every arm at every checkpoint, `K_decode = 4` only. This is where the
  verdict comes from, per brief §12: the question is not whether K=8 *decoding*
  improved, it is whether K=8 *training* produced a better K=4 drafter.
* **Endpoint diagnostics** — final checkpoints at `K_decode = 4, 6, 8`, plus the
  untrained zero-init adapter as a floor. Recorded, never decisive.

Metrics per brief §13, with U4's offset convention: **future offset `+j` is supervised
slot `j-1`**, and `+1` is the adapter-off seed row whose agreement is 1.0 by
construction and is an integrity check, not a learning signal.

## 9. Transfer metrics

For training horizon `h`, all at `K_decode = 4`:

```
ΔA_h  = A(h→4)   − A(4→4)      accepted-prefix transfer
ΔT_h  = TPF(h→4) − TPF(4→4)    algorithmic transfer
ΔS_h  = S(h→4)   − S(4→4)      wall-clock transfer
```

Each reported as a point estimate *and* a paired bootstrap interval. Positive means
useful cross-horizon transfer.

Transfer efficiency (`ΔA_h` per training hour) is reported, with the caveat that since
the arms are compute-matched to ~2.5% it ranks identically to `ΔA_h` itself. The
near-equality is the finding, not a limitation: a longer training horizon is close to
free here.

## 10. Resource safety

`src/qdif/uno/resource_guard.py` blocks heavy execution whenever a BoothGPT pretraining
process is visible, and is wired into `load_model()` in `scripts/uno.py` — the single
place that instantiates Qwen3.5-4B, so no subcommand can forget it. Three properties,
each tested:

1. **It never signals.** A test tokenises the module and asserts that `kill`, `signal`,
   `terminate`, `renice`, `Popen` and friends do not appear as code tokens, and that
   `subprocess.run` occurs exactly once with the fixed read-only argument list
   `["ps", "-axo", "pid=,command="]`.
2. **It fails closed.** An unreadable process table blocks.
3. **It cannot see itself.** `grep boothgpt`, the guard's own tests, and this repo's own
   `train.py` do not trip it.

On this machine it currently detects **two** protected processes, one more than the
brief named:

```
PID 35477  python train.py --config configs/v2/pretrain_h1_4k_working_set_v1.yaml
PID 35478  caffeinate -is .../boothgpt-qwen-shape-gate/.venv-mlx025/bin/python train.py …
```

`scripts/u5_launch.sh` waits on that gate — polling read-only, every 5 minutes — and
runs the whole experiment when the machine is free. It contains no code that could
stop, signal or reconfigure another process.

## 11. Cost

Training 4 × 3200 steps ≈ 5.0 h at U4's measured 1.40–1.44 s/step. Primary evaluation
≈ 2 h (21 checkpoints, K=4 only). Endpoint diagnostics ≈ 1 h. Total ≈ 8 h of exclusive
machine time.

## 12. Out of scope

No dynamic-K routing of any kind (brief §24): no entropy router, no draftability
router, no acceptance-history router, no learned router. No mixed-horizon objective
`L₄ + λL₈` (brief §8) — that adds a second variable before the first question is
answered. U4 already established K=4 as the best static runtime; U5 asks only how good
a K=4 drafter training can make.
