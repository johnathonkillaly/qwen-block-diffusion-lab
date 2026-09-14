# Results index

Navigation only. Each stage lists its dates, question, verdict, report, where its data
lives, and the checkpoint it used or produced. Narrative:
[PROJECT_SUMMARY_THROUGH_U5.md](PROJECT_SUMMARY_THROUGH_U5.md). Claims and caveats:
[../FINDINGS.md](../FINDINGS.md).

**Committed data** lives in `results/` and `plots/`. **Raw run directories** live in
`runs/`, which is git-ignored and exists only on the development machine. **Weights** are
never in git; their SHA-256 digests are in `results/checkpoint_manifest.json` (§ Checkpoints).

---

## Stages

| stage | dates | question | verdict | report | committed data | raw runs (not committed) | checkpoint |
|---|---|---|---|---|---|---|---|
| Act I | 2026-08-24 | Can Qwen3.5 learn token denoising with LoRA? | plumbing works; copy collapse; AR destroyed without an AR term | [EXPERIMENTS.md](EXPERIMENTS.md) | — | `runs/exp002-*`, `exp003-*`, `exp004-*`, `dev-overfit` | — |
| Act II / v0.2 / Phase 3 | 2026-08-25 | Does a shared-weight bidirectional Gated DeltaNet help? | **rejected** | [BIDIRECTIONAL_DELTANET.md](BIDIRECTIONAL_DELTANET.md), [../RESEARCH.md](../RESEARCH.md) | — | `runs/p3-A…E`, `v02-*` | — |
| Act III | 2026-08-25 | Does a FLARE-style AR + diffusion objective give a healthy denoiser? | 6/7 criteria → **recorded failure** (healthy denoiser) | [ACT3_JOURNAL.md](ACT3_JOURNAL.md), [ACT3_CRITERIA.md](ACT3_CRITERIA.md) | — | `runs/act3-main` | `runs/act3-main/checkpoint` |
| Act IV-N | 2026-09-01 | Does a structured corruption alphabet help? | **paused** before Stage 2 | its documents and code are in `stash@{0}`, not in this working tree (see `AGENTS.md`) | — | `runs/act4-stage1-*` | — |
| Act IV-U | 2026-09-03 | Can a Uno-style adapter on a frozen 4B model draft blocks the model accepts? | `ALGORITHMIC SIGNAL, NO SPEEDUP` | [act4u_results.md](act4u_results.md), [criteria](act4u_preregistered_criteria.md) | — | `runs/uno-stage0`, `uno-baseline`, `uno-k4-true`, `uno-k4-shuffled*`, `uno-compare`, `uno-bench` | `runs/uno-k4-true` |
| Act IV-U2 | 2026-09-03/04 | Does transactional verification remove the replay forward? | `TRANSACTIONAL SPEEDUP ACHIEVED` | [act4u2_results.md](act4u2_results.md), [criteria](act4u2_preregistered_criteria.md) | — | `runs/uno-bench/u2_transactional.json`, `runs/uno-draftability` | `runs/uno-k4-true` (the Act IV-U adapter; recorded in `provenance.adapter`) |
| Act IV-U3 | 2026-09-04 | Does acceptance, then speed, scale with training? | `ACCEPTANCE AND SPEED SCALE` | [act4u3_results.md](act4u3_results.md), [criteria](act4u3_preregistered_criteria.md) | `results/act4u3/` | `runs/u3a` | `runs/u3a/step-3200` |
| Act IV-U4 | 2026-09-04/05 | How far can the parallel horizon go? | `K4 IS THE PRACTICAL FRONTIER` | [act4u4_results.md](act4u4_results.md), [criteria](act4u4_preregistered_criteria.md) | `results/act4u4/` | `runs/u4a`, `u4b`, `u4b1`, `u4b2-k6`, `u4b2-k8`, `u4-calib` | `runs/u4b1/step-16000` (best K=4); `runs/u4a/step-12800` |
| RPRM Stage 1 | 2026-09-08 | Does denoiser uncertainty justify adaptive early exit? | **STOP** | [../RPRM_DIFFUSION_STAGE1_RESULTS.md](../RPRM_DIFFUSION_STAGE1_RESULTS.md), [prereg](../RPRM_DIFFUSION_PREREG.md) | `results/rprm_stage1/` | — | `runs/u4b2-k8/step-16000` (primary), `runs/uno-k4-true` (replication) |
| Act IV-S | 2026-09-13 | The speculative decoder: baseline, K sweep, correctness, controls, context, adaptive K | `K=4 STANDS. ADAPTIVE K FAILS. TWO OF THE APPARENT WINS WERE ARTIFACTS.` | [RESULTS_SPECULATIVE.md](RESULTS_SPECULATIVE.md), [design](SPECULATIVE_DESIGN.md), [state recovery](STATE_RECOVERY.md) | `results/speculative/` (raw rows, CSVs, divergence audits), `plots/speculative/` | `runs/spec-*.log` | `runs/u4b1/step-16000` (frozen) |
| Act IV-U5 | 2026-09-13 (run), 2026-09-14 (scored) | Does longer-horizon training make a better K=4 drafter? | `U4 OBSERVATION WAS NOISE`; `NO CROSS-HORIZON TRANSFER` also holds | [act4u5_results.md](act4u5_results.md), [criteria + A1–A4](act4u5_preregistered_criteria.md), [design](act4u5_design.md) | `results/act4u5/` (incl. `raw/`, `provenance/`) | `runs/u5/` | start `runs/u4a/step-12800`; arms `runs/u5/{A_k4,B_k6,C_k8,D_curr}/step-16000`; replicates `runs/u5/{A_k4_r2,A_k4_r3,C_k8_r2}/step-16000` |
| Act IV-U6 | 2026-09-14 | Does decoupling draft width from verify width speed up the decoder? | `VERIFY COST DOMINATES` | [act4u6_results.md](act4u6_results.md), [criteria](act4u6_preregistered_criteria.md), [design](act4u6_design.md) | `results/act4u6/`, `plots/act4u6/` | `runs/u6-*.log` | `runs/u4b1/step-16000` (frozen) |

---

## Where each headline number comes from

| claim | artifact |
|---|---|
| K=4 1.245× native AR (60.32 vs 48.46 tok/s) | `results/speculative/main_ksweep.json` (raw rows); `main_ksweep_summary.csv` |
| K=4 1.230× on loop-free text | `results/speculative/main_ksweep_summary_clean.csv` |
| reproduced 1.248× and 1.238× | `main_controls_summary.csv`, `main_adaptive_summary.csv` |
| K=1/2/8/16 speedups, survival curve | `main_ksweep_summary.csv`, `main_ksweep_survival.csv` |
| 759 divergences, all within one bf16 ULP | `results/speculative/main_*_divergence_audit.json` |
| 1.24% of positions are exact bf16 ties | `results/speculative/main_baseline.json` → `tie_rate` |
| loop fractions 0.18 / 0.50 | `results/speculative/main_degeneration.json` |
| n-gram and refinement controls | `results/speculative/main_controls.json`, `main_controls_summary(_clean).csv` |
| context scaling 1.444× → 1.118× decode-only | `results/speculative/main_context.json`, `main_context_summary.csv` |
| adaptive K best 1.230× vs fixed 1.238× | `results/speculative/main_adaptive.json`, `main_adaptive_summary(_clean).csv` |
| U5 endpoint paired changes | `results/act4u5/raw/eval_endpoints.json` → `transfer`; `u5_transfer.csv` |
| U5 every checkpoint vs matched control | `results/act4u5/u5_matched_step_pairing.json` |
| U5-5 cacheless losslessness (13/15) | `results/act4u5/u5_losslessness.json` |
| launch spread (A 0.976–1.029; C 0.988 / 1.083) | `results/act4u5/raw/eval_replicates.json`; `u5_supplementary_robustness.json` |
| replicate launches identical | `results/act4u5/u5_replicate_provenance.json`, `u5_replicate_check.json`, `provenance/` |
| A1 launch divergence at step 12810 | `results/act4u5/provenance/a1_launch_overlap.txt` |
| U5 training record and stop conditions | `results/act4u5/u5_training_summary.json` |
| U4 K=4 1.195×, K=8 0.910× K=4 | `results/act4u4/u4_summary.json` and CSVs |
| U3 1.056× → 1.168× | `results/act4u3/u3_summary.json` and CSVs |
| RPRM +0.022 AUROC | `results/rprm_stage1/analysis_primary.json` |
| U6 staged arms −1.6% to −9.0% vs K=4 | `results/act4u6/u6_scores.json` → `arms`; raw rows `results/act4u6/decisive.json` |
| U6 floor 1.0% | `results/act4u6/floor.json` (from `calibrate.json`) |
| verify forward 20.1 ms + 0.80 ms/token | `results/act4u6/cost_curve.json` → `verify_fit_2_to_9` |
| wider draft does not change slots 1–4 | `results/act4u6/diagnostic_wide_draft.json` |
| staging commits identical per-cycle tokens across V | `results/act4u6/decisive.json` rows `committed_per_cycle` |

Act I–III and Act IV-U/U2 numbers are backed by run directories in `runs/`, each with a
`run_metadata.json` or equivalent. They were never committed; that is a gap for
publication and is recorded as one.

---

## Checkpoints

Full digests: `results/checkpoint_manifest.json`. Base model:
`unsloth/Qwen3.5-4B-Base`, snapshot `61541fe4aed37b6a1e615edd5cbf5a3f660312b1`, bf16,
4,205,751,296 parameters, backbone digest `7cd0f5a9…` (not redistributed).

| id | path | role | adapter sha256 (prefix) |
|---|---|---|---|
| `act4s_u6_drafter` | `runs/u4b1/step-16000` | best K=4 drafter; frozen for Act IV-S and Act IV-U6 | `8200c339d3d47363` |
| `u5_shared_start` | `runs/u4a/step-12800` | K=4 saturated; Act IV-U5 start | `8332986b8a411f58` |
| `rprm_primary` | `runs/u4b2-k8/step-16000` | U4 arm B2; RPRM primary | `284f00ed247d85cf` |
| `rprm_replication` | `runs/uno-k4-true` | Act IV-U K=4; RPRM replication | `37cc5f8e6c258d47` |
| `act4u3_endpoint_k4` | `runs/u3a/step-3200` | U3 endpoint | `abc82b17ffc3bc39` |
| `u5_A_k4`, `u5_B_k6`, `u5_C_k8`, `u5_D_curr` | `runs/u5/*/step-16000` | U5 arms | `177f4b0c…`, `ef8a2feb…`, `e38e960e…`, `f0465a5a…` |
| `u5_A_k4_r2`, `u5_A_k4_r3`, `u5_C_k8_r2` | `runs/u5/*/step-16000` | U5 replicate launches | `5958319a…`, `a55cce21…`, `e2588a5c…` |

**Release strategy.** Each adapter is 85 MB. For a GitHub release:
1. Attach `adapter.safetensors` for the checkpoints the release needs as release assets.
   At minimum that is `act4s_u6_drafter`, the one every speed claim uses.
2. Quote the SHA-256 values from the manifest in the release notes.
3. Tell users to verify with `shasum -a 256`.

Weights never go into git history. Optimizer state is only needed to resume training and
stays local.
