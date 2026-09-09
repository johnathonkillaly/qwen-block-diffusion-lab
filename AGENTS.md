# AGENTS.md — shared instructions for Claude Code and Codex

Read this first. It is the handoff contract for this repository: either agent should
be able to pick the project up from here without re-deriving the state.

Keep it current. If you change the layout, the commands, or the experiment plan,
update this file **in the same commit**.

---

## 1. What this repository is

`qdif` — "Qwen Diffusion Lab". A local research lab on Apple Silicon investigating
whether an autoregressive **Qwen3.5-4B-Base** can be adapted into a **discrete text
diffusion** model, using LoRA on a mostly-frozen backbone.

> ### ⚠️ There are two different "Act IV"s. Do not confuse them.
>
> | | | |
> |---|---|---|
> | **Act IV-N** | structured **N**oise alphabet | `configs/act4/`, `src/qdif/mlx_backend/act4*.py`, `docs/ACT4_CRITERIA.md` (UPPERCASE) |
> | **Act IV-U** | **U**no diffusion distillation | `src/qdif/uno/`, `scripts/uno.py`, `docs/act4u_*.md` |
> | **Act IV-U2** | transactional recurrent verification | `recurrence.py`, `transaction.py`, `docs/act4u2_*.md` |
> | **Act IV-U3** | acceptance scaling | `cost_model.py`, `scripts/u3_report.py`, `docs/act4u3_*.md` |
> | **Act IV-U4** | horizon scaling | `rng.py`, `scripts/u4_report.py`, `docs/act4u4_*.md` |
>
> Act IV-N was paused at "ready to run Stage 2" and is **untouched**. Act IV-U is a
> separate track added later at the user's request, reproducing IFM's Uno. They share
> no code. The lowercase/uppercase doc split is the disambiguator.

It is a research notebook, not a product. **Negative results are the main output**,
and they are kept in deliberately. Aborted runs, a mis-specified success criterion and
a rejected hypothesis are all recorded rather than cleaned up.

### The prime directive

> This project is run as an experiment intended to **disprove itself**. Controls and
> clean comparisons matter more than producing an exciting metric.

Concretely, and non-negotiably:

- **Pre-register criteria before running.** Never invent or adjust a threshold after
  seeing results. `docs/ACT3_CRITERIA.md`, `docs/ACT4_CRITERIA.md`,
  `docs/act4u_preregistered_criteria.md`, `docs/act4u2_preregistered_criteria.md`,
  `docs/act4u3_preregistered_criteria.md`, `docs/act4u4_preregistered_criteria.md` and
  `RPRM_DIFFUSION_PREREG.md` are frozen once committed.
- **A falling loss is not a result.** Act I had loss fall 5× and accuracy reach 96.9%
  while the model had learned only to echo its input.
- **A degenerate model cannot rank a mechanism.** Act II compared five arms that had
  all stopped reading the canvas. Health gates exist to prevent that recurring.
- **Every arm of a comparison must differ on exactly one axis.** Act II's controls ran
  at a different mixing gate than the arm they controlled for, which invalidated the
  comparison for a whole phase.

---

## 2. Environment

| | |
|---|---|
| Working venv | **`.venv-unsloth`** (Python 3.13.12) — use this one |
| Other venv | `.venv` — legacy torch path, currently broken (see §6) |
| Model checkpoint | `/Volumes/SHUTTLE/hub/models--unsloth--Qwen3.5-4B-Base/...` |
| **Required env var** | **`export HF_HOME=/Volumes/SHUTTLE`** for anything that loads the model |
| Backend | MLX (`mlx 0.32.1`, `mlx_lm 0.31.3`), Unsloth Gated-DeltaNet custom VJP active |
| Hardware | Apple M4 Max, 128 GB unified. Act IV Stage 2 peaks at ~16.3 GB |

```bash
VIRTUAL_ENV=.venv-unsloth uv pip install -e ".[mlx,dev]" unsloth
```

Model weights, `runs/`, and `*.safetensors` are **git-ignored**. Never commit them.

---

## 3. Layout

```
src/qdif/
  cli.py                     all commands (argparse; one cmd_* per subcommand)
  config.py                  typed YAML config; unknown keys are a hard error
  diffusion/
    corruption.py            v0.1 torch corruption (uniform / mask)
    noise_alphabet.py        ACT IV: the structured corruption alphabet
    structured_corruption.py ACT IV: the forward process over that alphabet
    clustering.py            ACT IV: spherical k-means (cosine)
    bucket_builder.py        ACT IV: build + analyse the D/E bucket tables
    resolve.py               ACT IV: config -> alphabet; reverse arm -> fusion
  mlx_backend/               THE REAL BACKEND — all Act II/III/IV experiments
    build.py                 assemble a run: load -> wrap -> LoRA -> freeze
    model.py                 MLXDiffusionQwen; embed() splits clean vs noise ids
    deltanet.py              bidirectional Gated DeltaNet + fusions + controls
    noise_embedding.py       ACT IV: E_noise side table, split_embed
    act3.py / act3_trainer.py    Act III objective + trainer
    act4.py / act4_trainer.py    Act IV metrics + trainer
    act4_stage0.py           Act IV Stage 0 sanity gate
    act4_report.py           Act IV Stage 2 scorer (encodes the frozen criteria)
  models/                    v0.1 torch path (legacy, currently broken)
  uno/                       ACT IV-U — see below. Independent of mlx_backend.
configs/act4/                GENERATED — do not hand-edit (see §5)
artifacts/buckets/           committed bucket tables + cluster report
docs/                        research journal and design docs
runs/                        run outputs (git-ignored)
```

### `src/qdif/uno/` — Act IV-U

```
gated_lora.py   token-conditional LoRA + routing context + freezing
model.py        UnoModel; ar_logits / draft_logits; backbone byte-fingerprint
noise.py        uniform / deterministic / mask draft noise
teacher.py      training-batch layout, teacher and student logits
losses.py       TV (IFM's default), reverse KL, CE
verifier.py     greedy acceptance + a slow sequential reference to check it
decode.py       AR baseline and the Uno draft->verify cycle, cached and uncached
cache_utils.py  hybrid-cache snapshot/restore (KV offset + DeltaNet state)
recurrence.py   ACT IV-U2: DeltaNet forward that records per-token state
transaction.py  ACT IV-U2: begin/commit_prefix/rollback; replay|snapshot|rewind
cost_model.py   ACT IV-U3/U4: acceptance -> forwards/token -> tok/s; v2 adds a
                measured cycle-overhead term, survival curves and slot economics
rng.py          ACT IV-U4: named, separated PRNG streams keyed on (seed, stream, step)
metrics.py      per-slot agreement, entropy buckets
trainer.py      training loop, saturation guard, adapter save/load, exact resume
data.py         wikitext windows + the held-out prompt suite
tiny.py         a small *real* hybrid model for fast tests
```

**Deliberately independent of `mlx_backend/`.** Act IV-U never wraps the backbone,
never makes a DeltaNet bidirectional and never applies a custom mask — the AR path is
exercised exactly as pretrained. Sharing code with Acts I–IV-N would risk changing
runs those Acts own.

---

## 4. Tests

```bash
.venv-unsloth/bin/python -m pytest -q -m "not model"
```

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python -m pytest -q tests/test_act3.py tests/test_bidirectional_deltanet.py tests/test_structured_noise_model.py
```

**Current state (run both before and after any change):**

| set | result |
|---|---|
| `-m "not model"` (fast; must always pass) | **423 passed** (240 pre-Act-IV-U + 183 Uno/U2/U3/U4) |
| MLX model tests | **64 passed** |
| torch v0.1 (`test_model_integration.py`) | **19 failed, 9 passed — pre-existing** |

The 19 torch failures are **not yours to fix unless asked**. `transformers 5.5.0` is
installed but `pyproject.toml` declares `>=5.8`; upstream `masking_utils.py` now
requires `attention_mask` to be a tensor while the v0.1 path passes a dict of
per-layer masks (`AttributeError: 'dict' object has no attribute 'ndim'`). The MLX
path, where every real experiment runs, is unaffected. Baseline was measured before
the Act IV work and is unchanged after it.

Do not run the whole suite in one process on a memory-constrained machine — loading
the 4B model in both torch and MLX in one session can exhaust unified memory and
produce spurious collection errors. Run the two groups separately, as above.

---

## 5. Experiment conventions

- **Configs are the interface.** Everything is driven by a typed YAML config; unknown
  keys raise at load time rather than silently defaulting.
- **`configs/act4/*.yaml` are generated.** Edit `scripts/make_act4_configs.py` and
  regenerate. `tests/test_act4_config.py::test_generated_configs_are_up_to_date` fails
  on hand-edits. This exists because hand-edited near-identical YAML is exactly how
  the Act II control ended up at a different gate from its arm.
- **Bucket tables are committed artifacts, not run-time computations.** A designated
  corruption mode without a `mapping_path` is refused, so two arms can never disagree
  about what bucket 7 means.
- **Old configs must keep reproducing old runs.** New behaviour goes behind a new
  config field with a legacy-preserving default. `configs/p3_*.yaml` and
  `configs/act3_main.yaml` are frozen artifacts.
- **`diffusion.timestep_conditioning` stays `false`.** It caused the Act III run-1
  canvas collapse (bias norm 20× the token-embedding norm). A norm bound now exists,
  but under identifiable corruption the module is unnecessary anyway.

---

## 6. Where the project is now

### Completed

- **Act I** — denoising plumbing; discovered copy collapse; established that a falling
  loss proves nothing.
- **Act II** — shared-weight bidirectional Gated DeltaNet. Rejected by pre-registered
  criteria. **Re-audited in this session: the rejection does not hold up** — all five
  arms were degenerate and two controls were mis-constructed. Treat as *untested*, not
  *refuted*. See `docs/previous_experiment_reconstruction.md` §6–7.
- **Act III** — FLARE-inspired `L_AR + λ·L_diff` with absorbing-state masking. Produced
  a genuinely healthy denoiser (58.2% masked accuracy at t=0.10 on held-out
  WikiText-103) while *improving* AR perplexity. 6/7 criteria → recorded as FAILURE by
  its own definition.

### In progress — Act IV-U: Uno diffusion distillation *(the active track)*

**Hypothesis.** A frozen AR backbone plus a small *token-conditional* LoRA can propose
blocks of K tokens that the frozen model itself verifies and accepts — preserving
output exactly while cutting sequential decoding steps. This is a from-source
reproduction of IFM's **Uno** (`github.com/ifm-ai/uno`, released 2026-09-03); the
training code, runtime and a released adapter are all public, so the objective is
transcribed rather than guessed.

**Architecture.** Backbone 100% frozen and byte-hashed. Trainable: gated LoRA
r=16/α=256 on `q,k,v,o` (the 8 full-attention layers) and `gate,up,down` (all 32
MLPs) = **21,233,664 params, 0.5023%** of the backbone. The adapter contributes
**only on diffusion-noise rows**; prefill, the seed row, verification and every plain
AR call run the bare backbone. That gating is the whole method — an always-on adapter
is not Uno.

**Objective.** IFM's default: total variation between the noisy student's and the
clean teacher's full-vocabulary distributions, `α=β=0, γ=1`. The teacher is the same
frozen weights with the adapter off.

Docs: [`docs/act4u_uno_source_notes.md`](docs/act4u_uno_source_notes.md) (what IFM
actually published, tagged confirmed / inferred / our approximation),
[`docs/act4u_design.md`](docs/act4u_design.md),
[`docs/act4u_preregistered_criteria.md`](docs/act4u_preregistered_criteria.md) (frozen),
[`docs/act4u_results.md`](docs/act4u_results.md).

**Status: Act IV-U, U2, U3 and U4 all complete.** U4 (horizon scaling) is the most
recent: [`docs/act4u4_preregistered_criteria.md`](docs/act4u4_preregistered_criteria.md)
(frozen), [`docs/act4u4_results.md`](docs/act4u4_results.md), `results/act4u4/`.

### Act IV-U4 verdict: `K4 IS THE PRACTICAL FRONTIER`

K=4 reached pre-registered saturation at step 6400 — 9600 further steps (3x U3's whole
budget) moved teacher-forced agreement +0.008 against a 2 sd of 0.011. K=8 learns and
commits more tokens per forward (TPF 1.4664 vs K=4's 1.4314) and beats AR at 1.074x,
but is **0.910x K=4** on wall clock: it pays +11.9% per cycle for +2.4% TPF. Only
**three speculative slots pay for themselves** at any block size tested. 11 of 13
gates pass; the two failures are the result.

Best measured: **K=4, 59.08 tok/s, 1.195x AR**, lossless greedy, frozen backbone.

**The finding that was not pre-registered, and the one worth chasing:** 3200 steps of
K=8/K=6 training improved *K=4* decoding by more (accept +0.030, ~7 sd) than 9600
further steps of K=4 training did (+0.010). A longer block may be a better *training*
objective than it is an *inference* configuration. That is U5's first experiment, with
the obvious control being 3200 more K=4 steps.

Two methodological corrections U4 makes to earlier Acts, both in the conservative
direction:

* **U3's "+1/+2 offsets" were speculative-slot indices.** Future offset `+j` is
  supervised slot `j-1`, and `+1` is the adapter-off seed row (agreement 1.0 by
  construction). U3's "+1" is U4's "+2". The U3 finding is unchanged; the labels move.
* **U3's "repeat noise 1.95 tok/s" was ~12x too large.** It drew fresh global-RNG
  draft noise per repeat, so repeats decoded different sequences. Keyed
  (`src/qdif/uno/rng.py`), repeats are bit-identical and the jitter is 0.05-0.22
  tok/s. Gate U3-4 was therefore harder than intended: **U3's conclusion stands and
  was understated.**

### Act IV-U verdict: `ALGORITHMIC SIGNAL, NO SPEEDUP`

| gate | result |
|---|---|
| 0 — integrity | **PASS** — backbone byte-identical throughout; stage 0 26/26 |
| 1 — learning (K=2) | **PASS** — 0.391 vs 0.063 untrained, shuffled control 0.039 |
| 2 — prediction (K=4) | **WEAK** — 0.203 mean spec agreement (MODERATE needed 0.25) |
| 3 — lossless decoding | **PASS** — 100% token equality, K=1,2,4, reference regime |
| 4 — sequential steps | **FAIL** — TPF 0.980, needed > 1.000 |
| 5 — wall clock | **FAIL** — 0.63–0.83× AR |

**The next experiment is #1 in `act4u_results.md`: eliminate the replay forward.** It is
worth more than any amount of extra training — removing that single term alone takes
measured TPF 0.980 to a counterfactual 1.20–1.28.

**Three facts worth not re-deriving:**

1. **The single-forward training layout does not survive Gated DeltaNet.** IFM's
   `block_diff_mask` needs an arbitrary attention mask; 24/32 of our layers are causal
   *by construction* and take only a padding mask. We use one noisy block per window
   at the suffix and two plain causal forwards, which is exactly the computation the
   draft pass performs at inference.
2. **The backbone is not chunking-invariant.** Greedy decoding with a KV cache and
   with full recomputation produce *different token sequences* on Qwen3.5-4B **with no
   adapter involved**. So exact-losslessness is asserted in the cacheless reference
   regime; the cached path is reported against AR-vs-AR disagreement as its noise floor.
3. **TV has a vanishing gradient under saturation.** At lr 3e-4 the student's logits
   blow up (absmax 24 → 76), the softmax goes one-hot, `|g|` hits 0 and the run
   freezes at constant loss — which *reads as a null result*. Use IFM's 1e-5. A
   saturation guard in `trainer.py` now raises instead of letting this pass silently.

### Complete — RPRM diffusion, Stage 1: denoiser uncertainty *(STOP)*

A separate, narrower pre-registered test built **on top of** the frozen Act IV-U4
checkpoints (`runs/u4b2-k8` primary, `runs/uno-k4-true` replication) — not a numbered
Act IV-U stage and does not revise any Act IV-U verdict. Asks whether the denoiser's
own per-token uncertainty predicts the frozen verifier's accept/reject decision well
enough to justify adaptive early exit.

**Verdict: FAIL on the pre-registered criterion → STOP.** Denoiser entropy is real,
monotone, token-specific, survives progress controls and a shuffled-permutation
control, and is better-behaved than teacher entropy (no low-end inversion) — but adds
only **+0.022 AUROC** over a progress-only baseline against a pre-registered **≥0.05**
bar. Replicates independently on a second adapter. Stage 2 (the early-exit mechanism)
was **not run**, per the pre-registered stop rule — and separately, Uno decoding is
single-shot (one adapter forward yields all `K−1` proposals), so there is no per-token
denoising loop for early exit to save work from, regardless of signal strength. A
diagnostic top-1-probability variant (+0.0358) is recorded as a diagnostic only, not a
rescue.

Docs: [`RPRM_DIFFUSION_STAGE0_AUDIT.md`](RPRM_DIFFUSION_STAGE0_AUDIT.md),
[`RPRM_DIFFUSION_PREREG.md`](RPRM_DIFFUSION_PREREG.md) (frozen),
[`RPRM_DIFFUSION_STAGE1_RESULTS.md`](RPRM_DIFFUSION_STAGE1_RESULTS.md). Code:
`scripts/rprm_stage1_extract.py`, `scripts/rprm_stage1_analyze.py`. Data:
`results/rprm_stage1/`.

Do not reopen this experiment, move its thresholds, or treat the diagnostic top-1
result as a pass.

### Paused — Act IV-N: structured corruption alphabet

Replace the single `[MASK]` with a small corruption vocabulary
`<NOISE_00>…<NOISE_31>` that lives **outside** the clean vocabulary, and ask whether
noise states that carry information about the source token help — and whether they
make genuinely aligned reverse information more useful.

Design: `docs/structured_noise_diffusion_plan.md`.
Frozen criteria: `docs/ACT4_CRITERIA.md`.

**Status: ready to run Stage 2, PAUSED while Act IV-U is active. Do not start it
without reading the criteria.**

- Stage 0 (implementation sanity): **30/30 PASSED**
- Stage 1 (each condition can learn): **PASSED** for all five conditions
- Stage 2 (held-out separation): **not started**

---

## 7. Commands

### Act IV-U (Uno) — the active track

Everything goes through one runner. Every subcommand writes a JSON artifact under
`runs/` carrying the git SHA, versions, seed and full config.

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py stage0
```

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py baseline --decode-tokens 48
```

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py overfit --block-size 4 --steps 120 --lr 1e-5
```

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py train --block-size 4 --steps 400 --lr 1e-5 --out runs/uno-k4-true
```

Control 2 is the same command plus `--shuffle-targets`. Control 1 is `bench` with no
adapter. Benchmark a trained adapter against the AR baseline in one harness:

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py bench --adapter runs/uno-k4-true/adapter.safetensors --block-sizes 1,2,4,8
```

Act IV-U2 — transactional decoding (the fast path) and the draftability analysis:

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py bench --adapter runs/uno-k4-true/adapter.safetensors --block-sizes 2,4 --transaction-modes replay,snapshot,rewind
```

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py draftability --adapter runs/uno-k4-true/adapter.safetensors --block-size 8 --eval-batches 96
```

Act IV-U3 — scaling run, then evaluate every checkpoint in ONE session:

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py train --block-size 4 --steps 3200 --lr 1e-5 --checkpoint-steps 400,800,1200,1600,2400,3200 --out runs/u3a
```

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py u3-eval --checkpoint step-400=runs/u3a/step-400 --checkpoint step-3200=runs/u3a/step-3200 --block-sizes 2,4 --out runs/u3a/eval.json
```

```bash
.venv-unsloth/bin/python scripts/u3_report.py --eval runs/u3a/eval.json --out results/act4u3
```

Act IV-U4 — horizon scaling. Resume is **exact**: the adapter, the AdamW moments, the
step, the data position and the noise stream all continue, and
`tests/test_uno_resume.py` proves a split run is bitwise-identical to an
uninterrupted one.

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py train --block-size 4 --steps 12800 --resume-from runs/u3a/step-3200 --checkpoint-steps 4800,6400,8000,9600,11200,12800 --eval-every 100 --eval-batches 8 --out runs/u4a
```

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py u4-eval --checkpoint step-3200=runs/u3a/step-3200 --checkpoint step-12800=runs/u4a/step-12800 --block-sizes 2,4,8 --eval-batches 64 --out runs/u4a/eval.json
```

```bash
.venv-unsloth/bin/python scripts/u4_report.py --eval runs/u4a/eval.json --training runs/u4a/result.json --out results/act4u4
```

Harness calibration (three replicates of one frozen checkpoint, differing only in the
noise seeds) is how U4's thresholds were set — rerun it before changing any of them:

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python scripts/uno.py u4-eval --checkpoint step-3200=runs/u3a/step-3200 --block-sizes 2,4,8 --eval-batches 64 --noise-stream-seed 77000001 --eval-noise-seed 44000001 --out runs/u4-calib/rep2.json
```

Act IV-U tests (no checkpoint needed — they use `qdif.uno.tiny`):

```bash
.venv-unsloth/bin/python -m pytest -q tests/test_uno_gated_lora.py tests/test_uno_verifier.py tests/test_uno_decode.py tests/test_uno_teacher.py tests/test_uno_losses.py
```

### Act IV-N (structured noise) — paused

Stage 0 — implementation sanity (fast, no training):

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/qdif act4-stage0 -c configs/act4/stage1_E_semantic_designated.yaml
```

Rebuild the bucket tables (only if the model or `k` changes):

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/qdif build-buckets -c configs/act4/stage2_E_semantic_designated.yaml --num-states 32
```

Stage 1 — one condition, ~1.7 min:

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/qdif act4-train -c configs/act4/stage1_E_semantic_designated.yaml
```

Stage 2 — the full 11-run matrix, ~2 h (**the next real step**):

```bash
./scripts/run_act4_stage2.sh
```

Score Stage 2 mechanically against the frozen criteria:

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/qdif act4-report --runs runs --out runs/act4-logs/report.json
```

Regenerate configs after editing the generator:

```bash
.venv-unsloth/bin/python scripts/make_act4_configs.py
```

Prior work: `qdif act3-train`, `act3-check`, `heldout`, `p3-report`, `probe-bidir`,
`unsloth-report`, `generate-trace`, `inspect`, `memory`.

---

## 8. Act IV design facts worth not re-deriving

- **`tie_word_embeddings = True`.** The readout *is* the embedding table. This is why
  noise states need a side table (`E_noise`) rather than spare embedding rows: a spare
  row would be a legal output. Noise ids start at `noise_base = 248,320 = model vocab`,
  outside the readout's index range, so they are structurally unemittable.
- **All conditions share one `E_noise` initialisation seed and distribution** (Gaussian
  rescaled to the mean token-embedding norm, 0.661). Initialising semantic states at
  their cluster centroids would confound "semantic buckets help" with "semantic
  initialisation helps".
- **Condition D is a *permutation* of condition E's assignment.** Bucket sizes and
  therefore `H(bucket)` are identical to machine precision (measured `|ΔH| = 0.00e+00`,
  4.1163 of 5.0 bits). This is what makes E-vs-D a test of structure rather than of
  information quantity. **The comparison that matters is E vs D, never E vs A.**
- **Two RNG streams**, one for the corrupted set and one for the noise states, so all
  five conditions corrupt *exactly the same positions* of the same examples.
- **Reverse arms all run at the same gate (0.5)** and the shuffled control resamples
  its permutation every forward pass. Both fix Act II defects; legacy defaults are
  preserved so the old configs still reproduce.
- **Cluster caveat:** clusters 21 and 23 hold the same lemmas split by capitalisation
  and leading whitespace, so "semantic" buckets also encode surface form. Any E-over-D
  gain may be partly orthographic. Recorded as an interpretation limit.

---

## 9. Working agreements

- Do **not** launch a long training run unless asked. Stage 2 is ~2 hours.
- Report failures plainly, with the output. If a criterion fails, it failed.
- Prefer adding a control over adding a metric.
- When you find a flaw in past work, fix it **and** write down what it invalidates —
  do not quietly correct the code and leave the old conclusion standing.
- Keep `docs/` as the source of truth for *why*; keep this file as the source of truth
  for *how to run it*.
