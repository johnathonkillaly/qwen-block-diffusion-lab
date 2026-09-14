#!/bin/bash
# Act IV-U5, criteria §11 A1-A4: replicate launches for the supplementary robustness check.
#
# Two launches of arm A_k4 from one start diverged within 40 steps (A1), so an
# arm-vs-control difference contains launch-to-launch training noise that the frozen
# floors do not measure. This script measures it: two more launches of the control (A2)
# and one independent launch of C_k8 (A4). It does not change the primary run, and it
# never feeds a replicate into the frozen gates.
#
#   1. While the primary run trains, capture each primary arm's exact command line and
#      environment from the process table (read-only), so identity can be proven later.
#   2. Wait for scripts/u5_launch.sh to exit completely, including its timed evaluations.
#   3. Refuse to start if the primary run did not finish.
#   4. Train A_k4_r2, A_k4_r3, then C_k8_r2, recording start digests, command line, code and
#      environment immediately before each launch.
#   5. Check divergence (u5_start_check.py) and launch identity (u5_replicate_provenance.py).
#   6. Evaluate all seven endpoints in ONE session at K_decode=4.
#   7. Write the supplementary robustness section (u5_replicate_report.py).
#
# It polls read-only and never signals, suspends or reconfigures anything.
#
#   bash scripts/u5_replicates.sh
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${QDIF_PYTHON:-/Users/johnathonkillaly/code/diffusion_project/.venv-unsloth/bin/python}"
export HF_HOME="${HF_HOME:-/Volumes/SHUTTLE}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# Byte-identical to u5_launch.sh, including the checkpoint schedule: Act IV-U3 gate U3-0b
# found that changing an evaluation schedule changed the training trajectory, so a
# replicate with a different schedule would not be a replicate.
START="runs/u4a/step-12800"
STEPS=16000
CKPT="13200,13600,14400,15200,16000"
ALIGN=8
POLL="${U5_POLL_SECONDS:-300}"
COMMON=(--lr 1e-5 --rank 16 --window 128 --batch-size 4 --corruption uniform
        --train-windows 4000 --val-windows 64 --eval-every 100 --eval-batches 8
        --seed 20260903 --noise-align-width "$ALIGN"
        --resume-from "$START" --checkpoint-steps "$CKPT")

PROV="runs/u5/provenance"
PRIMARY_ARMS=(A_k4 B_k6 C_k8 D_curr)
# replicate name : K_train of the primary arm it replicates
REPLICATES=("A_k4_r2:4" "A_k4_r3:4" "C_k8_r2:8")

trap 'echo "[u5-rep] !! FAILED at line $LINENO (exit $?)  $(date "+%Y-%m-%d %H:%M:%S")"' ERR

stamp() { date '+%Y%m%d-%H%M%S'; }

gate_state() {  # prints "clear" or a one-line reason, never touches anything
  "$PY" - <<'PYEOF'
from qdif.uno.resource_guard import heavy_execution_allowed
allowed, signals = heavy_execution_allowed()
print("clear" if allowed else "; ".join(s.describe()[:90] for s in signals))
PYEOF
}

wait_for_machine() {
  local phase="$1" state
  while true; do
    state="$(gate_state)"
    [[ "$state" == "clear" ]] && return 0
    echo "[u5-rep] $phase waiting on the resource guard: $state"
    sleep "$POLL"
  done
}

snapshot() {  # code and environment digest; compared by u5_replicate_provenance.py
  local label="$1"
  {
    echo "label $label"
    echo "info git_head $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "HF_HOME $HF_HOME"
    echo "PYTHONPATH $PYTHONPATH"
    "$PY" -c 'import sys, mlx.core as mx, mlx_lm; print("python", sys.version.split()[0], "mlx", mx.__version__, "mlx_lm", getattr(mlx_lm, "__version__", "?"))'
    shasum -a 256 scripts/u5_launch.sh scripts/uno.py scripts/u5_start_check.py \
      scripts/u5_report.py src/qdif/uno/*.py
  } > "$PROV/env_$label.txt"
  echo "[u5-rep] snapshot $label"
}

capture_primary() {  # read-only: the primary arms' command lines and environments
  local arm pid cmd environ
  for arm in "${PRIMARY_ARMS[@]}"; do
    [[ -s "$PROV/primary_argv_$arm.txt" && -s "$PROV/primary_environ_$arm.txt" ]] && continue
    pid="$(ps -axww -o pid=,command= | awk -v want="--out runs/u5/$arm" '
      { line = $0; sub(/[ \t]+$/, "", line) }
      index(line, "scripts/uno.py train ") &&
      substr(line, length(line) - length(want) + 1) == want { print $1; exit }')"
    [[ -n "$pid" ]] || continue
    cmd="$(ps -ww -p "$pid" -o command= 2>/dev/null || true)"
    environ="$(ps -wwE -p "$pid" -o command= 2>/dev/null | tr ' ' '\n' \
               | grep -E '^(HF_HOME|PYTHONPATH)=' | sort || true)"
    [[ -n "$cmd" ]] && printf '%s\n' "$cmd" > "$PROV/primary_argv_$arm.txt"
    [[ -n "$environ" ]] && printf '%s\n' "$environ" > "$PROV/primary_environ_$arm.txt"
    echo "[u5-rep] captured primary $arm (pid $pid): argv $([[ -n "$cmd" ]] && echo yes || echo NO), environment $([[ -n "$environ" ]] && echo yes || echo NO)"
  done
}

echo "[u5-rep] Act IV-U5 supplementary replicates (criteria §11 A2, A4)  $(date '+%Y-%m-%d %H:%M:%S')"
echo "[u5-rep] replicates: ${REPLICATES[*]}"

# Replicate outputs can never land on a primary arm's directory.
for spec in "${REPLICATES[@]}"; do
  for arm in "${PRIMARY_ARMS[@]}"; do
    [[ "${spec%%:*}" != "$arm" ]] || { echo "[u5-rep] !! replicate name collides with primary arm $arm"; exit 2; }
  done
done

mkdir -p "$PROV" results/act4u5
snapshot "orchestrator_start_$(stamp)"

# 1-2. Capture provenance while waiting for the primary run to exit on its own.
capture_primary
if pgrep -f "scripts/u5_launch.sh" >/dev/null; then
  echo "[u5-rep] primary run in progress; polling every ${POLL}s, read-only"
  while pgrep -f "scripts/u5_launch.sh" >/dev/null; do
    capture_primary
    sleep "$POLL"
  done
  echo "[u5-rep] primary run exited  $(date '+%H:%M:%S')"
fi
snapshot "primary_exit_$(stamp)"

# 3. Only proceed if the primary run actually finished.
missing=0
for arm in "${PRIMARY_ARMS[@]}"; do
  [[ -f "runs/u5/$arm/step-16000/adapter.safetensors" ]] || { echo "[u5-rep] missing runs/u5/$arm/step-16000"; missing=1; }
done
for f in runs/u5/eval_curve.json runs/u5/eval_endpoints.json; do
  [[ -f "$f" ]] || { echo "[u5-rep] missing $f"; missing=1; }
done
if (( missing )); then
  echo "[u5-rep] !! primary run incomplete; not starting replicates. Re-run u5_launch.sh first."
  exit 2
fi
for arm in A_k4 C_k8; do
  [[ -s "$PROV/primary_argv_$arm.txt" ]] || echo "[u5-rep] !! primary $arm command line was not captured; provenance will report it unverified"
done

# 4. The replicate launches.
for spec in "${REPLICATES[@]}"; do
  name="${spec%%:*}"; block="${spec##*:}"; out="runs/u5/$name"
  if [[ -f "$out/step-16000/adapter.safetensors" ]]; then
    echo "[u5-rep] $name already complete, skipping"
    continue
  fi
  if [[ -e "$out" ]]; then
    aside="runs/u5/$name.partial-$(stamp)"
    echo "[u5-rep] $out exists without step-16000 (an interrupted launch); moving it to $aside"
    mv "$out" "$aside"
  fi
  wait_for_machine "$name"
  snapshot "prelaunch_$name"
  "$PY" scripts/u5_start_check.py --start "$START" \
    --out "results/act4u5/u5_replicate_prelaunch_$name.json"
  cmd=("$PY" -u scripts/uno.py train "${COMMON[@]}" --steps "$STEPS" --block-size "$block" --out "$out")
  printf '%s\n' "${cmd[*]}" > "$PROV/replicate_argv_$name.txt"
  printf 'HF_HOME=%s\nPYTHONPATH=%s\n' "$HF_HOME" "$PYTHONPATH" | sort > "$PROV/replicate_environ_$name.txt"
  echo "===== U5 replicate $name  K_train=$block  $(date '+%H:%M:%S') ====="
  "${cmd[@]}"
done

# 5. Divergence and identity. Identical adapters are a legitimate outcome for replicates
#    (u5_start_check's "horizon cannot have been the variable" wording does not apply).
"$PY" scripts/u5_start_check.py --start "$START" \
  --arm A_k4=runs/u5/A_k4/step-16000 \
  --arm A_k4_r2=runs/u5/A_k4_r2/step-16000 \
  --arm A_k4_r3=runs/u5/A_k4_r3/step-16000 \
  --arm C_k8=runs/u5/C_k8/step-16000 \
  --arm C_k8_r2=runs/u5/C_k8_r2/step-16000 \
  --out results/act4u5/u5_replicate_check.json
if "$PY" scripts/u5_replicate_provenance.py; then
  echo "[u5-rep] provenance: every replicate is an identical launch"
else
  echo "[u5-rep] !! PROVENANCE: a replicate failed an identity check; see results/act4u5/u5_replicate_provenance.json."
  echo "[u5-rep] !! Evaluation continues so the data exist; the report marks the replicate invalid."
fi

# 6. One supplementary paired session, seven endpoints, K_decode=4.
wait_for_machine "supplementary evaluation"
"$PY" -u scripts/uno.py u5-eval \
  --arm A_k4@16000=runs/u5/A_k4/step-16000 \
  --arm A_k4_r2@16000=runs/u5/A_k4_r2/step-16000 \
  --arm A_k4_r3@16000=runs/u5/A_k4_r3/step-16000 \
  --arm B_k6@16000=runs/u5/B_k6/step-16000 \
  --arm C_k8@16000=runs/u5/C_k8/step-16000 \
  --arm C_k8_r2@16000=runs/u5/C_k8_r2/step-16000 \
  --arm D_curr@16000=runs/u5/D_curr/step-16000 \
  --control A_k4@16000 --primary-block 4 --block-sizes 4 --eval-batches 64 \
  --noise-align-width "$ALIGN" --out runs/u5/eval_replicates.json

# 7. Supplementary section. Never the frozen scorecard.
"$PY" scripts/u5_replicate_report.py

echo "[u5-rep] complete  $(date '+%Y-%m-%d %H:%M:%S'). Read with criteria §11 A3-A4."
