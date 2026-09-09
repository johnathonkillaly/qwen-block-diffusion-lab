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

`src/qdif/uno/` is a from-source reproduction of IFM's public **Uno**
(`github.com/ifm-ai/uno`, Apache-2.0): a token-conditional LoRA over a frozen,
byte-hashed backbone that proposes blocks of tokens which the same frozen weights
then verify. It exists in this repository as **supporting infrastructure** — it
produces the adapter checkpoints that the RPRM experiment below analyzes. Its own
results are not part of this release.

It is a research notebook, not a product. **Negative results are the main output**,
and they are kept in deliberately. Aborted runs, a mis-specified success criterion and
a rejected hypothesis are all recorded rather than cleaned up.

### The prime directive

> This project is run as an experiment intended to **disprove itself**. Controls and
> clean comparisons matter more than producing an exciting metric.

Concretely, and non-negotiably:

- **Pre-register criteria before running.** Never invent or adjust a threshold after
  seeing results. `docs/ACT3_CRITERIA.md` and `RPRM_DIFFUSION_PREREG.md` are frozen
  once committed.
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
| Other venv | `.venv` — legacy torch path, currently broken (see §4) |
| Model checkpoint | `/Volumes/SHUTTLE/hub/models--unsloth--Qwen3.5-4B-Base/...` |
| **Required env var** | **`export HF_HOME=/Volumes/SHUTTLE`** for anything that loads the model |
| Backend | MLX (`mlx 0.32.1`, `mlx_lm 0.31.3`), Unsloth Gated-DeltaNet custom VJP active |
| Hardware | Apple M4 Max, 128 GB unified |

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
  mlx_backend/               THE REAL BACKEND — Act II/III experiments
    build.py                 assemble a run: load -> wrap -> LoRA -> freeze
    model.py                 MLXDiffusionQwen; embed() splits clean vs noise ids
    deltanet.py              bidirectional Gated DeltaNet + fusions + controls
    act3.py / act3_trainer.py    Act III objective + trainer
  models/                    v0.1 torch path (legacy, currently broken)
  uno/                       supporting infra for RPRM — see §1. Independent of
                             mlx_backend; never wraps the backbone, never makes a
                             DeltaNet bidirectional, never applies a custom mask.
    gated_lora.py   token-conditional LoRA + routing context + freezing
    model.py        UnoModel; ar_logits / draft_logits; backbone byte-fingerprint
    noise.py        uniform / deterministic / mask draft noise
    teacher.py      training-batch layout, teacher and student logits
    losses.py       TV, reverse KL, CE
    verifier.py     greedy acceptance + a slow sequential reference to check it
    decode.py       AR baseline and the draft->verify cycle, cached and uncached
    cache_utils.py  hybrid-cache snapshot/restore (KV offset + DeltaNet state)
    recurrence.py   DeltaNet forward that records per-token state
    transaction.py  begin/commit_prefix/rollback; replay|snapshot|rewind
    cost_model.py   acceptance -> forwards/token -> tok/s
    rng.py          named, separated PRNG streams keyed on (seed, stream, step)
    metrics.py      per-slot agreement, entropy buckets
    trainer.py      training loop, saturation guard, adapter save/load, exact resume
    data.py         wikitext windows + the held-out prompt suite
    tiny.py         a small *real* hybrid model for fast tests
docs/                        research journal and design docs
runs/                        run outputs (git-ignored) — checkpoints referenced by
                             RPRM (e.g. `runs/u4b2-k8`, `runs/uno-k4-true`) are
                             produced with scripts/uno.py and are not included here;
                             see RPRM_DIFFUSION_STAGE0_AUDIT.md for their exact config
results/rprm_stage1/         committed RPRM data, plots and summary JSON
scripts/
  uno.py                     trains/benchmarks the Uno-style adapter (supporting
                             infra; see §1)
  rprm_stage1_extract.py     RPRM: run the adapter, record denoiser entropy per position
  rprm_stage1_analyze.py     RPRM: models, criteria, plots — runs on the committed
                             CSVs alone, no model needed
```

---

## 4. Tests

```bash
.venv-unsloth/bin/python -m pytest -q -m "not model"
```

```bash
HF_HOME=/Volumes/SHUTTLE .venv-unsloth/bin/python -m pytest -q tests/test_act3.py tests/test_bidirectional_deltanet.py
```

**Current state (run both before and after any change):**

| set | result |
|---|---|
| `-m "not model"` (fast; must always pass) | **324 passed** |
| MLX model tests | **45 passed** |
| torch v0.1 (`test_model_integration.py`) | **19 failed, 9 passed — pre-existing** |

The 19 torch failures are **not yours to fix unless asked**. `transformers` installed
is newer than the v0.1 path's assumptions; upstream `masking_utils.py` now requires
`attention_mask` to be a tensor while the v0.1 path passes a dict of per-layer masks
(`AttributeError: 'dict' object has no attribute 'ndim'`). The MLX path, where every
real experiment runs, is unaffected.

Do not run the whole suite in one process on a memory-constrained machine — loading
the 4B model in both torch and MLX in one session can exhaust unified memory and
produce spurious collection errors. Run the two groups separately, as above.

---

## 5. Experiment conventions

- **Configs are the interface.** Everything is driven by a typed YAML config; unknown
  keys raise at load time rather than silently defaulting.
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
  criteria. See `docs/BIDIRECTIONAL_DELTANET.md`.
- **Act III** — FLARE-inspired `L_AR + λ·L_diff` with absorbing-state masking. Produced
  a genuinely healthy denoiser (58.2% masked accuracy at t=0.10 on held-out
  WikiText-103) while *improving* AR perplexity. 6/7 criteria → recorded as FAILURE by
  its own definition.

### Completed — RPRM diffusion, Stage 1: denoiser uncertainty *(STOP)*

A separate, narrowly pre-registered test built **on top of** the Uno-style adapter
checkpoints in `runs/` (`runs/u4b2-k8` primary, `runs/uno-k4-true` replication) —
not itself a claim about that adapter's own performance. Asks whether the denoiser's
own per-token uncertainty predicts the frozen verifier's accept/reject decision well
enough to justify adaptive early exit.

**Verdict: FAIL on the pre-registered criterion → STOP.** Denoiser entropy is real,
monotone, token-specific, survives progress controls and a shuffled-permutation
control, and is better-behaved than teacher entropy (no low-end inversion) — but adds
only **+0.022 AUROC** over a progress-only baseline against a pre-registered **≥0.05**
bar. Replicates independently on a second adapter. Stage 2 (the early-exit mechanism)
was **not run**, per the pre-registered stop rule — and separately, this adapter's
decoding is single-shot (one forward yields every proposal in a block), so there is no
per-token denoising loop for early exit to save work from, regardless of signal
strength. A diagnostic top-1-probability variant (+0.0358) is recorded as a diagnostic
only, not a rescue.

Docs: [`RPRM_DIFFUSION_STAGE0_AUDIT.md`](RPRM_DIFFUSION_STAGE0_AUDIT.md),
[`RPRM_DIFFUSION_PREREG.md`](RPRM_DIFFUSION_PREREG.md) (frozen),
[`RPRM_DIFFUSION_STAGE1_RESULTS.md`](RPRM_DIFFUSION_STAGE1_RESULTS.md). Code:
`scripts/rprm_stage1_extract.py`, `scripts/rprm_stage1_analyze.py`. Data:
`results/rprm_stage1/`.

Do not reopen this experiment, move its thresholds, or treat the diagnostic top-1
result as a pass.

---

## 7. Commands

```bash
.venv-unsloth/bin/qdif act3-train -c configs/act3_main.yaml
.venv-unsloth/bin/qdif generate-trace --checkpoint runs/act3-main/checkpoint --prompt "The rain had stopped by morning" --steps 16
.venv-unsloth/bin/qdif unsloth-report
```

Other `qdif` commands: `inspect`, `arch-report`, `corrupt`, `act3-check`,
`probe-bidir`, `act3-compare`, `p3-report`, `memory`.

`scripts/uno.py` trains and benchmarks the adapter checkpoints RPRM analyzes
(`--help` lists subcommands: `stage0`, `baseline`, `overfit`, `train`, `bench`,
`draftability`). It is supporting infrastructure for RPRM, not itself part of this
release's results — see §1.

RPRM Stage 1:

```bash
export HF_HOME=/Volumes/SHUTTLE
.venv-unsloth/bin/python scripts/rprm_stage1_extract.py \
    --adapter runs/u4b2-k8/adapter.safetensors --block-size 8 --batches 320 --tag primary
.venv-unsloth/bin/python scripts/rprm_stage1_analyze.py --tag primary
```

Analysis alone (seconds, no model, operates on the committed CSVs):

```bash
.venv-unsloth/bin/python scripts/rprm_stage1_analyze.py --tag primary
```

---

## 8. Working agreements

- Do **not** launch a long training run unless asked.
- Report failures plainly, with the output. If a criterion fails, it failed.
- Prefer adding a control over adding a metric.
- When you find a flaw in past work, fix it **and** write down what it invalidates —
  do not quietly correct the code and leave the old conclusion standing.
- Keep `docs/` as the source of truth for *why*; keep this file as the source of truth
  for *how to run it*.
