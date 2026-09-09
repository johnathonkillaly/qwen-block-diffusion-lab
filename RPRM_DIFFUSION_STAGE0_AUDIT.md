# RPRM diffusion — Stage 0 audit

**Date:** 2026-09-08
**Purpose:** locate and understand the prior Uno drafting work *before* any new
measurement, so that the proposed "denoiser uncertainty predicts acceptance"
experiment is not a rediscovery of something already run under another name.

**Nothing in this audit modifies prior results.** No training was started, no
checkpoint was written, no pre-registered criteria document was edited.

---

## 1. Repo and commit

| | |
|---|---|
| Canonical repo | `/Users/johnathonkillaly/code/diffusion_project` |
| Remote | `https://github.com/johnathonkillaly/qwen-block-diffusion-lab.git` |
| Branch | `main` |
| HEAD at audit time | `13e9d0594eb62a715fe5214b9d5270a9791620a3` — *"Act IV-U4: the useful parallel horizon is four tokens"* |
| Working tree | dirty — untracked Act **IV-N** (structured-noise) files only; the Act IV-U tree is clean |
| Backbone | `unsloth/Qwen3.5-4B-Base` (cached at `/Volumes/SHUTTLE/hub/models--unsloth--Qwen3.5-4B-Base`) |
| Env | `.venv-unsloth` (Python 3.13.12, MLX 0.32.1), `HF_HOME=/Volumes/SHUTTLE` |

### Search performed

I searched `~/code` and `/Volumes/SHUTTLE` for `uno`, `draft*`, `accept*`,
`diffus*`, `verif*`, `qwen`, `spec*` by directory name and by recently-modified
Python files. Two candidates surfaced:

1. **`~/code/diffusion_project`** — HEAD dated 2026-09-05, contains phases U, U2,
   U3, **and U4**. This is the most recent work.
2. **`~/code/uno-deltanet-mlx`** — a single-commit (`bd87845`, 2026-09-04)
   *public-release extraction* of the U and U2 phases only. It is a strict subset
   with files renamed (`scripts/run.py` ← `scripts/uno.py`), and it contains no
   U3/U4 material. Its `docs/draftability_gap.md` is a good prose summary and I
   cite it below, but it is **not** newer work.

**Conclusion: `diffusion_project` @ `13e9d05` is the canonical source.** No newer
Uno experiment exists elsewhere on the searched drives.

> Note the repo contains **two unrelated "Act IV"s** (per `CLAUDE.md`). Act IV-**N**
> is a paused structured-noise-alphabet track. Act IV-**U** is the Uno drafting
> track and is the only one relevant here. They share no code.

---

## 2. Relevant files

### Core library — `src/qdif/uno/`

| File | Role |
|---|---|
| `teacher.py` | **Batch construction.** Defines the teacher/student layout, the noise schedule, and `UnoBatch` (which exposes `corrupted` and `corruption_rate`). The single most important file for this experiment. |
| `noise.py` | Uniform random-token noise in `[1, mask_token_id)`; `build_draft_block`, `draft_lora_mask`. |
| `verifier.py` | `greedy_accept` — the decode-time acceptance rule. |
| `model.py` | `UnoModel.ar_logits` (adapter off) and `UnoModel.draft_logits(ids, lora_mask)` (adapter gated on). |
| `gated_lora.py` | The U-adapter: per-position gated LoRA, zero-init `lora_b`. |
| `losses.py` | TV objective. |
| `trainer.py` | Training loop, `load_adapter`, held-out teacher-forced eval. |
| `decode.py` | Free-running `uno_greedy_generate`, cycle accounting. |
| `metrics.py` | `pearson`. |
| `cost_model.py`, `transaction.py`, `recurrence.py` | U2/U3/U4 throughput machinery — not needed here. |

### Harness

| File | Role |
|---|---|
| `scripts/uno.py` | The CLI. **`cmd_draftability` (lines 767–892)** is the existing draftability/entropy analysis. `cmd_u3_eval` (~line 915) and `cmd_u4_eval` (~line 1226) embed it as `_draftability_quintiles` (line 1127) and `_draftability_by_offset` (line 1535). |
| `scripts/u3_report.py`, `scripts/u4_report.py` | Turn run JSON into the CSVs and plots under `results/`. |

### Results already on disk (do not overwrite)

| Path | Contents |
|---|---|
| `runs/uno-draftability/draftability.json` | K=4, 288 positions |
| `runs/uno-draftability/draftability_k8.json` | **K=8, 2,688 positions** — this is the "~2,688 evaluated positions" referenced in the brief |
| `results/act4u3/u3_draftability.csv` | 7 checkpoints × 5 quintiles, K=4, n=384/checkpoint |
| `results/act4u4/u4_draftability_by_offset.csv` | per-offset gap/entropy correlations, K=8, n=256 per (arm, checkpoint, offset) |
| `docs/act4u_results.md` §"Entropy analysis" | the original entropy finding |
| `~/code/uno-deltanet-mlx/docs/draftability_gap.md` | prose write-up of the gap |

> **On the number 2,688.** Two different quantities coincidentally land near it.
> The K=8 draftability run evaluated **2,688 speculative positions**
> (96 eval batches × batch 4 × 7 kept slots). Separately, `u3_draftability.csv`
> holds 7 checkpoints × 384 positions = 2,688 rows. The brief's "2,688
> evaluated examples/positions" refers to the **former**.

---

## 3. Checkpoints

All adapters are LoRA-only (`rank 16`, full-attention + MLP, DeltaNet layers
untouched), trained on frozen `Qwen3.5-4B-Base`.

| Run | Block size K | Steps | Adapter | Notes |
|---|---|---|---|---|
| `runs/uno-k4-true/` | 4 | 400 | `adapter.safetensors` | Act IV-U reproduction; the adapter used for the original draftability runs |
| `runs/uno-k4-shuffled{,2}/` | 4 | 400 | `adapter.safetensors` | shuffled-target control (must fail) |
| `runs/u3a/` | 4 | 3200 | `adapter.safetensors` + `step-{400…3200}/` | U3 acceptance-scaling ladder |
| `runs/u4a/` | 4 | 12800 | `adapter.safetensors` + `step-{4800…12800}/` | U4 arm A |
| `runs/u4b1/` | 8 | 16000 | `adapter.safetensors` + `step-{13200…16000}/` | U4 arm B1 |
| `runs/u4b2-k6/`, `runs/u4b2-k8/` | 6, 8 | 16000 | `adapter.safetensors` | U4 arm B2 — **the most-trained K=8 adapters** |

Common training config (`runs/u4b2-k8/result.json`):

```
block_size 8, window 128, batch 4, lr 1e-5, tv_weight 1.0, ce/kl 0.0,
noise_mode "random_uniform", corruption "uniform", lora_rank 16, seed 20260903
```

---

## 4. What the existing "draftability predictor" actually is

**Definition** (`cmd_draftability`, `scripts/uno.py:806–819`):

```
P_full    (j) = p(y_{t+j} | true causal predecessors)        -> model.ar_logits, adapter OFF
P_corrupt (j) = p(y_{t+j} | predecessors replaced by noise)  -> model.draft_logits, adapter ZEROED
draftability_gap(j) = TV(P_full, P_corrupt) = Σ_v |P_full(v) − P_corrupt(v)|
```

Four properties that matter for the new experiment:

1. **It is a property of the frozen model and the text — not of anything trained.**
   The adapter is explicitly reset to its zero-init no-op (`lora_b ← 0`) before
   both distributions are measured. This is why `mean_gap` is *bit-identical
   across every checkpoint* in `u4_draftability_by_offset.csv`.
2. **TV is the unhalved L1 distance**, so it lies in `[0, 2]`, matching the
   training objective's convention. Observed values are compressed into
   `≈[1.6, 2.0]` — the noised context is almost always far from the clean one.
3. **It measures predecessor dependence, not uncertainty.** The stated hypothesis
   is that what hurts a parallel draft is not "how unpredictable is this token"
   but "how much does it depend on the tokens a parallel draft cannot see."
4. **It is not computable at decode time.** `P_corrupt` is available (it is the
   draft pass we already run), but `P_full` requires the *true* intervening
   predecessors — which are exactly the tokens being drafted. The existing write-up
   is careful about this: it calls an online approximation "plausible but
   unproven." **This is an important asymmetry for the new experiment** and is
   recorded here, before results, so it cannot later be used as a rescue.

**Result achieved** (K=8, 2,688 held-out positions):

```
pearson(gap,     acceptance) = −0.2114
pearson(entropy, acceptance) = −0.0968
pearson(gap,     entropy)    = −0.2535
```

with monotone quintiles: acceptance 0.263 → 0.149 → 0.155 → 0.123 → 0.074.

---

## 5. What the existing "predictor entropy" actually is

**This is the single most important finding of this audit.**

The entropy already tested is:

```
H_teacher(j) = H( P_full(j) ) = H( p(y_{t+j} | TRUE causal predecessors) )
```

computed at `scripts/uno.py:822` as `−Σ exp(lf)·lf` where `lf` is the
log-softmax of **`full`**, i.e. the **frozen AR teacher's** distribution given
clean context.

It is therefore:

* **the *teacher's* uncertainty, not the *denoiser's*;**
* a property of the frozen model and the text, invariant to the adapter and to
  training step (again: identical `mean_entropy` across all checkpoints);
* conditioned on the **true** predecessors — the information a parallel draft
  explicitly *does not have*.

**Prior finding** (`docs/act4u_results.md`, "Entropy analysis — partially
supported"): the relationship is **non-monotone**. High teacher entropy strongly
predicts failure (top quintile 0.075 agreement, ~4× worse than any other), but
**low teacher entropy does not predict success** — the lowest-entropy bucket
(0.197 at K=4 / 0.149 at K=8) scores *below* the second bucket (0.316 / 0.207).
The stated explanation is that near-zero teacher entropy often marks a token
determined by the *immediately preceding* token (finishing a multi-token word, a
proper noun, a closing bracket) — precisely the information the draft row sees as
noise. The prior work concluded this is "a real argument against naive
entropy-routed dynamic block sizing."

---

## 6. How target acceptance is measured

There are **two** acceptance measurements in the codebase. They are not the same
and the distinction must be kept.

### (a) Teacher-forced per-position agreement — used by the draftability analysis

`scripts/uno.py:835` and `:1579`:

```python
hit = (mx.argmax(student, axis=-1) == mx.argmax(full, axis=-1))
```

* `student` = `model.draft_logits(batch.student_ids, batch.lora_mask)` with the
  **trained** adapter loaded — the proposal distribution on the noised block.
* `full` = `model.ar_logits(batch.teacher_ids)` — the frozen target model given
  true predecessors.
* Restricted to `batch.supervised_slice`.
* **Slot 0 is excluded**: it is the seed row, runs with the adapter off, and is
  therefore trivially a hit.
* This is a **per-position, independent** binary. It does *not* apply the
  prefix-truncation rule.

This is the rule the existing draftability predictor is scored against, and it is
the rule the new experiment must reuse unchanged.

### (b) Decode-time greedy prefix acceptance — used for throughput

`src/qdif/uno/verifier.py:greedy_accept`. One causal forward over
`context + [t₀ … t_{L−1}]` yields `p(· | context, t₀ … t_i)` for every prefix, so
L tokens are verified in one pass. Acceptance walks forward and **stops at the
first mismatch**:

```python
while accepted < num_specs and tokens[accepted+1] == spec_targets[accepted]:
    accepted += 1
```

A rejected position is *replaced* by the verifier's own token (so a cycle never
stalls and always commits ≥2 tokens); a fully-accepted block of L commits **L+1**
tokens, taking the free lookahead.

**Relationship between (a) and (b).** (b) is (a) plus a prefix constraint:
position *j* is decode-accepted only if *every* position `< j` also agreed. So (a)
is the per-position ceiling and (b) is what throughput actually sees. Measured
K=8 per-offset agreement under (a) falls 0.44 (+2) → 0.04 (+8).

---

## 7. Has denoiser entropy already been tested?

**No.** This is the pivotal question for the brief and the answer is clean.

| Quantity | Distribution | Adapter | Conditioned on | Tested? |
|---|---|---|---|---|
| Existing "predictor entropy" | `P_full` | **off** | **true** predecessors | **Yes** — weak, non-monotone, r ≈ −0.10 |
| Existing "draftability gap" | TV(`P_full`, `P_corrupt`) | **off** | both | **Yes** — monotone, r ≈ −0.21 |
| **Proposed denoiser uncertainty** | **`P_student`** | **trained, on** | **noised** predecessors | **No — never computed** |

The proposed H1 concerns `p_θ(x_{0,i} | x_t, t)` — the *adapter's own* output
distribution on the corrupted block. In this codebase that is exactly
`model.draft_logits(batch.student_ids, batch.lora_mask)` with the trained adapter
loaded. The existing code **computes this tensor** — at `scripts/uno.py:832` and
`:1577` — but then immediately reduces it with `argmax` and throws the
distribution away. Its entropy, top-1 probability and top-1/top-2 margin have
never been recorded, bucketed or correlated with anything.

So the new experiment is **adjacent to, but genuinely distinct from**, prior work.
Two consequences follow, and both are recorded now rather than after results:

* **The prior negative result is a real prior, not a duplicate.** Teacher entropy
  was weak and non-monotone. Denoiser entropy is a different variable, but they
  are related enough that a null result would be unsurprising.
* **The deployability asymmetry runs the other way.** The gap needs `P_full` and
  is *not* computable at decode time. Denoiser entropy is a free by-product of the
  draft forward that already happens. This does not make a weak result strong, but
  it is why "WEAK PASS" is worth distinguishing from "FAIL" here.

Two further uncertainty variables are nearly free and will also be recorded, so
that "denoiser uncertainty" is not silently conflated with one arbitrary choice:

* `H(P_corrupt)` — frozen model, noised context, **no** adapter. The
  adapter-free ablation of denoiser entropy.
* `H(P_student)` — trained adapter. **The primary H1 variable.**

---

## 8. Architectural finding: there is no diffusion timestep at inference

The brief's design assumes a diffusion schedule with timesteps `t`, within-timestep
controls, and Stage 2 "adaptive diffusion depth / early exit". The audit found the
following, which materially changes what those can mean.

### 8a. Inference is single-shot

`uno_greedy_generate` (`decode.py:243`) runs, per cycle:

```
draft block = [seed, ν₁ … ν_{K−1}]        (uniform random token ids)
  1 adapter forward  -> all K−1 proposals at once
  1 verify forward   -> greedy_accept
```

There is **no iterative refinement**: no loop over decreasing noise levels, no
per-position update schedule, no denoising depth. Every evaluation entry point in
`scripts/uno.py` hardcodes `corruption="full"` (lines 143, 380, 551, 809, 989,
1145, 1330, 1560), i.e. every draft row holds noise. Deployment sees exactly one
corruption condition.

**Consequence for Stage 2:** "percentage of token-position denoising updates
avoided" has no referent in this architecture — there are no per-position updates
to skip, and freezing a position saves nothing, because one forward produces all
K−1 proposals regardless. Stage 2 as literally specified is **not runnable here**.
The nearest real operational lever is *adaptive block width* (how many speculative
slots to offer), which is a different quantity with a different cost model. This
is flagged now, before Stage 1 results exist, so that Stage 2 is not quietly
redefined afterwards to fit whatever Stage 1 produces.

### 8b. But a genuine corruption-level axis exists in training

`build_uno_batch(corruption=...)` (`teacher.py:71–174`) supports:

* `"full"` — every draft row is noise; `corruption_rate = 1.0`. **The only state
  that occurs at inference.**
* `"uniform"` — Uno/SDAR-faithful: a per-sequence rate **`t ~ U(0, 1)`** decides
  how many of the `K−1` draft rows hold noise. `UnoBatch` exposes both
  `corruption_rate` (per sequence, `[B]`) and `corrupted` (per position, `[B, L]`).

**Every U3/U4 adapter was trained with `corruption="uniform"`** (confirmed in all
three `result.json` files). So the adapter has seen the whole schedule, and `t` is
a legitimate, in-training-distribution diffusion-progress axis — it is simply
never exercised at eval time by the existing harness.

This gives the confound controls the brief asks for, with one honest caveat:

| Brief's variable | Available analogue here |
|---|---|
| diffusion timestep `t` | **`corruption_rate` `t ~ U(0,1)`** — real, in-distribution for training, but **off-manifold for deployment** (deployment is always `t = 1`) |
| corruption level | per-position `corrupted` flag, and the count of corrupted predecessors before position *j* |
| — (no brief analogue) | **future offset `j`** — the structural distance from the seed. This is the dominant driver of acceptance (0.44 at +2 → 0.04 at +8) and the existing code already conditions on it (`_draftability_by_offset`). |

**Both `t` and `j` must be conditioned on.** Offset is arguably the more important
control of the two, because it is the one that varies at deployment and the one
that already demonstrably drives acceptance. The pre-registration will treat
"survives the progress control" as requiring survival within **both**.

---

## 9. Data and split hygiene

* Corpus: **WikiText-103-raw-v1**, cached locally.
* `build_sources` (`data.py:159`) draws train from the `train` split and
  validation from the `validation` split — **separate dataset splits, not slices
  of one stream**. There is no leak path from training text into the analysis.
* Windows are contiguous 128-token blocks; the analysed block is always the
  suffix. One window = one example. All `K−1` analysed positions within a window
  share a context, so **the split for any fitted model must be at the window
  level**, never at the token level. This is exactly the hazard the brief warns
  about and it is real here: 7 correlated positions per window at K=8.
* Reproducibility: `WindowSource` is a pure function of `(seed, draws)`;
  `make_noise` accepts an explicit PRNG `key` so evaluation consumes no global RNG
  state. Pre-registered gate U3-0b exists specifically because a shared global
  stream once made two identically-seeded runs diverge. **Any new analysis must
  pass explicit keys.**

---

## 10. Summary of what Stage 1 must respect

1. Reuse the existing acceptance rule **unchanged**: per-position teacher-forced
   `argmax(student) == argmax(teacher)`, slot 0 excluded. (§6a)
2. Reuse the existing draftability gap **unchanged** as the incumbent predictor,
   and score against it. (§4)
3. Keep teacher entropy as a *separate* variable from denoiser entropy — the
   former is the already-tested one. (§5, §7)
4. Condition on **both** corruption rate `t` **and** future offset `j`. (§8b)
5. Split at the **window** level. (§9)
6. Record that the gap is not decode-time computable while denoiser entropy is —
   as context for interpreting a WEAK PASS, not as a rescue. (§4)
7. Do **not** carry Stage 2 forward in its literal form; there are no per-position
   denoising updates in this architecture. (§8a)

**Nothing in this audit has looked at any correlation between denoiser
uncertainty and acceptance.** No such number has been computed. Pre-registration
follows in `RPRM_DIFFUSION_PREREG.md`.
