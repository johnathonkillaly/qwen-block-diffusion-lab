#!/bin/bash
# Act IV-U5 — Train Long, Decode Short. The whole heavy run, gated on the machine
# being free.
#
# This script waits; it never clears the way. If a BoothGPT pretraining job is running
# it polls, read-only, until that job exits on its own. There is no code path here that
# signals, suspends or reconfigures another process, and `scripts/uno.py` re-checks the
# guard itself before every model load, so a pretraining job that *starts* mid-run
# stops the next phase rather than racing it.
#
#   bash scripts/u5_launch.sh              # wait for the machine, then run everything
#   bash scripts/u5_launch.sh --dry-run    # print the plan and the gate state, run nothing
#
# Roughly 5 h of training (4 arms x 3200 steps) then ~3 h of evaluation.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${QDIF_PYTHON:-/Users/johnathonkillaly/code/diffusion_project/.venv-unsloth/bin/python}"
export HF_HOME="${HF_HOME:-/Volumes/SHUTTLE}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

START="runs/u4a/step-12800"          # the shared, byte-identical starting point
STEPS=16000                          # 12800 + 3200 matched steps per arm
CKPT="13200,13600,14400,15200,16000" # +400 +800 +1600 +2400 +3200
ALIGN=8                              # widest arm, so all arms share their noise
POLL="${U5_POLL_SECONDS:-300}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

COMMON=(--lr 1e-5 --rank 16 --window 128 --batch-size 4 --corruption uniform
        --train-windows 4000 --val-windows 64 --eval-every 100 --eval-batches 8
        --seed 20260903 --noise-align-width "$ALIGN"
        --resume-from "$START" --checkpoint-steps "$CKPT")

gate_state() {  # prints "clear" or a one-line reason, never touches anything
  "$PY" - <<'PYEOF'
from qdif.uno.resource_guard import heavy_execution_allowed
allowed, signals = heavy_execution_allowed()
print("clear" if allowed else "; ".join(s.describe()[:90] for s in signals))
PYEOF
}

wait_for_machine() {
  local phase="$1" state waited=0
  while true; do
    state="$(gate_state)"
    [[ "$state" == "clear" ]] && break
    if (( waited == 0 )); then
      echo "[u5] $phase blocked, waiting for the machine (polling every ${POLL}s)."
      echo "[u5]   $state"
      echo "[u5]   Not touching it. This script only waits."
    elif (( waited % 3600 < POLL )); then
      echo "[u5] still waiting after $((waited / 60)) min: $state"
    fi
    sleep "$POLL"
    waited=$((waited + POLL))
  done
  (( waited > 0 )) && echo "[u5] machine free after $((waited / 60)) min; starting $phase"
  return 0
}

train_arm() {  # name, then the horizon flags
  local name="$1"; shift
  if [[ -f "runs/u5/$name/step-16000/adapter.safetensors" ]]; then
    echo "[u5] arm $name already complete, skipping"
    return 0
  fi
  wait_for_machine "arm $name"
  echo "===== U5 arm $name ====="
  "$PY" -u scripts/uno.py train "${COMMON[@]}" --steps "$STEPS" "$@" --out "runs/u5/$name"
}

echo "[u5] Act IV-U5 — Train Long, Decode Short"
echo "[u5] shared start   $START"
echo "[u5] matched budget 3200 steps/arm, checkpoints $CKPT"
echo "[u5] noise align    $ALIGN (arms share corruption on shared positions)"
echo "[u5] gate           $(gate_state)"
if (( DRY_RUN )); then
  echo "[u5] --dry-run: nothing executed"
  exit 0
fi

mkdir -p runs/u5 results/act4u5

# 1. Shared-start evidence (no model load; safe at any time).
"$PY" scripts/u5_start_check.py --start "$START" --out results/act4u5/u5_start_check.json

# 2. Four matched arms. Every arm resumes from the same checkpoint, so the adapter and
#    the AdamW moments are identical at step 12800 by construction.
train_arm A_k4    --block-size 4
train_arm B_k6    --block-size 6
train_arm C_k8    --block-size 8
train_arm D_curr  --curriculum 4,6,8

# 3. Confirm the arms actually diverged.
"$PY" scripts/u5_start_check.py --start "$START" \
  --arm A_k4=runs/u5/A_k4/step-16000 --arm B_k6=runs/u5/B_k6/step-16000 \
  --arm C_k8=runs/u5/C_k8/step-16000 --arm D_curr=runs/u5/D_curr/step-16000 \
  --out results/act4u5/u5_start_check.json

# 4. PRIMARY: every arm decodes at K=4, all checkpoints, one session, paired.
wait_for_machine "primary evaluation"
ARMS=()
for arm in A_k4 B_k6 C_k8 D_curr; do
  for step in 13200 13600 14400 15200 16000; do
    ARMS+=(--arm "$arm@$step=runs/u5/$arm/step-$step")
  done
done
"$PY" -u scripts/uno.py u5-eval "${ARMS[@]}" \
  --arm "A_k4@12800=$START" --control A_k4@12800 \
  --primary-block 4 --block-sizes 4 --eval-batches 64 \
  --noise-align-width "$ALIGN" --out runs/u5/eval_curve.json

# 5. Endpoints with K=6/K=8 diagnostics, plus the untrained floor. Diagnostic only:
#    brief sec.12 keeps the verdict at K_decode=4.
wait_for_machine "endpoint diagnostics"
"$PY" -u scripts/uno.py u5-eval \
  --arm untrained= \
  --arm "start@12800=$START" \
  --arm A_k4@16000=runs/u5/A_k4/step-16000 \
  --arm B_k6@16000=runs/u5/B_k6/step-16000 \
  --arm C_k8@16000=runs/u5/C_k8/step-16000 \
  --arm D_curr@16000=runs/u5/D_curr/step-16000 \
  --control A_k4@16000 --primary-block 4 --block-sizes 4,6,8 --eval-batches 64 \
  --noise-align-width "$ALIGN" --out runs/u5/eval_endpoints.json

# 6. Tables and the nine plots.
"$PY" scripts/u5_report.py \
  --eval runs/u5/eval_curve.json --eval runs/u5/eval_endpoints.json \
  --training runs/u5/A_k4/result.json --training runs/u5/B_k6/result.json \
  --training runs/u5/C_k8/result.json --training runs/u5/D_curr/result.json \
  --out results/act4u5

echo "[u5] U5-RUN complete. Score the gates in docs/act4u5_preregistered_criteria.md."
