# State recovery — 2026-09-13

Written before any new experiment was run, per the brief's §2 ("recover before
changing"). Every claim below is backed by a file in this repository or by a command
re-run today. Where the incoming brief's recollection disagrees with the artifacts,
**the artifacts win** and the disagreement is stated.

---

## 0. Headline: the experiment the brief describes already exists and has results

The brief asks whether a diffusion-trained adapter can draft tokens that a frozen
autoregressive target verifies, producing identical output faster. That is **Act IV-U**,
run across four numbered stages between 2026-09-03 and 2026-09-05. It is lossless, it
beats AR wall-clock, and its frontier is measured.

> **Best previously measured configuration: K=4, 59.08 tok/s vs 49.45 tok/s AR =
> 1.195×, greedy, bit-exact identical output, frozen 4B backbone.**
> — `docs/act4u4_results.md`, arm B1 +3200 at `K_decode=4`.

So the mission is not to build a speculative decoder. It is to **finish** one: several
things the brief asks for were never measured. Those are listed in §5.

---

## 1. Where the code actually lives

Two git worktrees share one repository and one `runs/` directory:

| path | branch | role |
|---|---|---|
| `/Users/johnathonkillaly/code/diffusion_project` | `public-release` | curated publication branch; Acts I–III + RPRM only |
| **`/Users/johnathonkillaly/code/diffusion-u5`** | **`rprm-diffusion-stage1`** | **the live research branch — Act IV-U…U5 lives here** |

`diffusion-u5/runs` is a **symlink** to `diffusion_project/runs`, so both see the same
checkpoints. `diffusion-u5` has no venv of its own and uses
`diffusion_project/.venv-unsloth` by absolute path.

`public-release` is **behind** on 13 files (3184 insertions): all of `scripts/u3_report.py`,
`u4_report.py`, `u5_*`, `src/qdif/uno/resource_guard.py`, the `teacher.py`/`trainer.py`
horizon-pairing changes, and four test modules. Act IV-U's *results documents*
(`docs/act4u*.md`) exist **only** on `rprm-diffusion-stage1`.

**Consequence for this session: all new work happens in `diffusion-u5`.** The editable
install in `.venv-unsloth` points at `diffusion_project/src`, so every command must set
`PYTHONPATH=/Users/johnathonkillaly/code/diffusion-u5/src` or it silently imports the
older `public-release` copy of `qdif`. Without it, two test modules fail to import.

There is also one stash (`stash@{0}`, "Act IV-N paused work in progress") — an unrelated
paused track, left untouched.

---

## 2. Completed stages, with artifacts

| stage | verdict | artifacts |
|---|---|---|
| **Act I** | denoising plumbing; discovered copy collapse | `runs/exp00{2,3,4}*`, `docs/EXPERIMENTS.md` |
| **Act II** | bidirectional Gated DeltaNet **REJECTED** by pre-registered criteria | `runs/p3-{A..E}`, `docs/BIDIRECTIONAL_DELTANET.md` |
| **Act III** | healthy denoiser, **6/7 criteria → recorded FAILURE** | `runs/act3-main`, `docs/ACT3_JOURNAL.md` |
| **Act IV-U** | Uno distillation reproduced on frozen Qwen3.5-4B | `runs/uno-*`, `docs/act4u_results.md` |
| **Act IV-U2** | transactional verification removes the replay forward | `runs/uno-bench/u2_transactional.json`, `docs/act4u2_results.md` |
| **Act IV-U3** | `ACCEPTANCE AND SPEED SCALE` — K=4 to 1.168× AR | `runs/u3a/`, `results/act4u3/`, `docs/act4u3_results.md` |
| **Act IV-U4** | `K4 IS THE PRACTICAL FRONTIER` — 1.195× AR; K=8 loses | `runs/u4a/`, `runs/u4b*/`, `results/act4u4/`, `docs/act4u4_results.md` |
| **RPRM Stage 1** | **STOP** — denoiser entropy too weak for adaptive early exit | `results/rprm_stage1/`, `RPRM_DIFFUSION_STAGE1_RESULTS.md` |

### Checkpoint inventory (all present on disk, verified today)

| checkpoint | what it is |
|---|---|
| `runs/u3a/step-3200` | U3 endpoint, K=4-trained |
| `runs/u4a/step-12800` | K=4 trained to **saturation** (9600 further steps bought +1.2% TPF) |
| **`runs/u4b1/step-16000`** | **arm B1: +3200 steps at K=8. The best K=4 decoder measured (1.195× AR).** |
| `runs/u4b2-k8/step-16000` | arm B2: curriculum K=6→K=8. 1.185× AR at K=4 |
| `runs/u4b2-k6/step-14400` | B2's intermediate K=6 stage |
| `runs/u4b2-k8` (16000) | also the **RPRM primary** adapter |
| `runs/uno-k4-true` | RPRM replication adapter |

---

## 3. What Act IV-U already establishes (constraints, not assumptions)

These are measured and should not be re-derived:

1. **Losslessness holds.** Greedy speculative output is bit-exact against native AR at
   K=2/4/8 (gate U4-B4, `runs/u4b/losslessness.json`). Re-verified today at K=4 on
   `runs/u4b1/step-16000`: 48/48 token IDs identical.
2. **The cycle costs 2 forwards, or 3 on partial rejection.** 24 of 32 layers are a
   Gated DeltaNet recurrence that *cannot be rewound mid-forward*, so the verify pass
   re-sends the seed and a partial acceptance replays. This is architectural
   (`decode.py` docstring, `cache_utils.py`), not a bug.
3. **Committed tokens per cycle = accepted + 2, exactly**, for every K. So the horizon
   survival curve's area *is* the expected accepted prefix.
4. **Cost model v2 is measured, not fitted**: 2.36% mean |error| over 48 points, and it
   still leans +2.3% optimistic. Marginal price of one draft slot: **1.28–1.52 ms**
   (R² ≥ 0.993).
5. **Only three speculative slots ever pay for themselves**, at any K tested. K=8's
   slots 4–7 cost 1.5 ms each and return 0.059, 0.013, 0.004, 0.000 tokens.
6. **K=8 beats AR (1.074×) but loses to K=4 (0.910×).** It buys +2.4% TPF for +11.9%
   cycle cost.
7. **Between-session wall-clock noise is ±4% and is *not* thermal** — it is the
   draft-noise realisation. Within-session, keyed, repeats are bit-identical decodes
   with 0.05–0.22 tok/s jitter. **Every comparison must be in-session with the noise
   stream keyed.**
8. **The median over 15 prompts is a fragile estimator** and inflated two U4 effects
   (3.9% vs the mean's 1.4%; 5.2% vs 2.8%). The mean is the defensible one.
9. **Draftability gap predicts acceptance, but not increasingly with horizon** — the
   pre-registered U4 hypothesis was refuted.

### RPRM Stage 1 — what its STOP does and does not forbid

RPRM asked whether the **denoiser's own per-token entropy** predicts the verifier's
accept/reject decision well enough to justify **adaptive early exit inside a block**.
Verdict **FAIL → STOP**: +0.022 AUROC over a progress-only baseline against a
pre-registered ≥0.05 bar, replicated on a second adapter. It is frozen.

Two separate reasons it does not block this session's adaptive-K work, stated so the
distinction cannot blur later:

* RPRM scored the **drafter's** uncertainty about **its own already-drafted tokens**.
  Choosing `K` for the *next* cycle is a different predictor with a different target.
* RPRM's own documents record that this adapter's decoding is **single-shot**, so there
  is no per-token denoising loop for early exit to save work from *regardless of signal
  strength*. That is a statement about the early-exit mechanism, not about block sizing.

Anything measured here is reported under its own name. RPRM's thresholds are not
touched and its diagnostic top-1 result is not treated as a pass.

---

## 4. Implemented but NOT evaluated: Act IV-U5

`docs/act4u5_design.md` and `docs/act4u5_preregistered_criteria.md` are **frozen**, the
rig is built and tested (`scripts/u5_launch.sh`, `u5_report.py`, `u5_start_check.py`,
`uno.py u5-eval`, `tests/test_uno_horizon_pairing.py`, `tests/test_uno_paired_stats.py`),
and **no U5 training step has ever run.**

`results/act4u5/` contains exactly one file: `u5_start_check.json`. There is no U5
training output, no U5 eval, no U5 result.

U5's question — does training at K=6/K=8 produce a better *K=4* decoder than training at
K=4, under a matched budget? — comes from an **unplanned** U4 observation
(`docs/act4u4_results.md` § "The result nobody pre-registered"). It cost ~8 h of
exclusive machine time and was blocked by an unrelated BoothGPT pretraining job.

**That job is no longer running** (process table checked today: no `boothgpt`, no
`train.py`). The guard would now permit U5. It remains un-run, and this session does
not start it (see §6).

---

## 5. Planned or implied by the brief, with NO artifact proving it ran

This is the actual gap, and it is where new work goes:

| brief § | asked for | status |
|---|---|---|
| §13 | acceptance stratified by **prompt class** | **never reported.** `PROMPT_SUITE` has 5 domains (prose/factual/code/math/structured) and every Act IV-U document aggregates over them. No per-domain acceptance table exists. No `reasoning`, `dialogue` or `high-entropy` category exists at all. |
| §19 | **context scaling** — 2K / 8K / 16K / 32K | **never run.** Every Act IV-U number is at a ~10–25 token prompt and 48 generated tokens. There is no long-context measurement anywhere in the repository. |
| §17 | a **conventional speculative baseline** (n-gram / prompt-lookup / small AR drafter) | **never built.** The only controls are the untrained zero-init adapter and shuffled targets. Nothing answers "would a cheap non-diffusion drafter do as well?" |
| §11–12, Gate 5 | **adaptive K** | **explicitly out of scope for U5** (`act4u5_design.md` §12 rules out every router). Never implemented. |
| §14 | **refinement steps** 1 vs 2 vs 4 | **not possible as built** — the draft pass is single-shot. No iterative-refinement decode path exists. |
| §5 | **K=1 control** and **K=16** | K=1 never run; K=16 never run (K=2/4/8 only). |
| §4 | 128 / 256 / **512**-token generations | all Act IV-U runs are 48 tokens. |
| §16 | full acceptance-prefix **distribution** | partially present (`u4_acceptance_histogram.csv`, `u4_survival.csv`) but only at K≤8 and pooled over domains. |

One more thing found while reading, worth fixing rather than inheriting:

**`scripts/uno.py bench` reports best-of-N.** It keeps the row with the highest
`tokens_per_second` across repeats (`if row["tokens_per_second"] > best[...]`). The
brief's §20 forbids exactly that. The U3/U4 eval commands do not do this — they report
median and mean — so **no published Act IV-U number is affected**; `bench` is the older
U2-era command. New work uses its own harness with median + dispersion and interleaved
native/speculative runs.

---

## 6. What this session does, and what it deliberately does not

**Does:** the §5 gaps, all of which are *evaluation* on frozen checkpoints and need no
training — consistent with the brief's §15 ("Do not retrain immediately. First test
existing checkpoints.").

**Does not:** launch U5's ~8 h training. `AGENTS.md` §8 says "Do not launch a long
training run unless asked", U5's criteria are frozen and deserve to be run as their own
pre-registered experiment rather than folded into another brief's session, and the
brief's §15 gates retraining behind evidence from existing checkpoints. If the
evaluation below shows short-horizon acceptance is the binding constraint, U5 is the
experiment already written and waiting.

---

## 7. Reproduction commands verified today

```bash
cd /Users/johnathonkillaly/code/diffusion-u5
PYTHONPATH=$PWD/src /Users/johnathonkillaly/code/diffusion_project/.venv-unsloth/bin/python -m pytest -q -m "not model"
```

**383 passed, 40 deselected** (2026-09-13). Without `PYTHONPATH` this is
`2 errors during collection`.

Model load + losslessness spot-check, `runs/u4b1/step-16000`, K=4, 48 tokens:

```
AR  49.20 tok/s  forwards=48
UNO 53.01 tok/s  forwards=37  acceptance=0.222
IDENTICAL: True
```

Backbone: `Qwen3.5-4B-Base`, 4,205,751,296 params, bf16, 8.41 GB, 8 full-attention +
24 Gated DeltaNet layers, Unsloth custom VJP active. Adapter r=16, 128 modules,
21,233,664 trainable (0.5023%).

---

## Addendum — 2026-09-14

This document is a snapshot of 2026-09-13, written before that day's experiments. Two
statements in it have since been refined or superseded:

* **§0 and §3 item 1** say greedy speculative output is "bit-exact" against native AR.
  Later the same day, Act IV-S found it is exact **up to bf16 ties**
  ([`RESULTS_SPECULATIVE.md`](RESULTS_SPECULATIVE.md) §2). The U4-B4 check it relied on
  used a narrower regime and prompt set.
* **§4 and §6** say Act IV-U5 was never run. It ran on 2026-09-13 and was scored on
  2026-09-14, with verdict `U4 OBSERVATION WAS NOISE`
  ([`act4u5_results.md`](act4u5_results.md)).
