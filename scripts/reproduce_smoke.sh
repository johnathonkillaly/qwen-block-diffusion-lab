#!/usr/bin/env bash
# Small end-to-end reproduction: verifies the whole Act III path without the
# multi-hour transfer run. ~5 minutes on an M-series Mac with the 4B checkpoint
# already downloaded.
#
# Usage:  ./scripts/reproduce_smoke.sh [config]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/act3_smoke.yaml}"
PY="${QDIF_PYTHON:-.venv-unsloth/bin/qdif}"
export HF_HOME="${HF_HOME:-/Volumes/SHUTTLE}"

echo "=============================================================="
echo " qdif smoke reproduction"
echo " config: $CONFIG"
echo " python: $PY"
echo "=============================================================="

echo; echo ">>> 1/4  backend capability report (verified, not assumed)"
$PY unsloth-report | sed -n '1,40p'

echo; echo ">>> 2/4  Act III milestone checks (objective, masking, mechanisms, gradients)"
$PY act3-check -c "$CONFIG" --json-out runs/smoke/act3-check.json

echo; echo ">>> 3/4  short transfer run"
$PY act3-train -c "$CONFIG"

echo; echo ">>> 4/4  diffusion trace on the resulting adapters"
$PY generate-trace -c "$CONFIG" \
    --checkpoint runs/act3-smoke/checkpoint \
    --prompt "The rain had stopped by morning" \
    --steps 8

echo; echo "Smoke reproduction complete. Artifacts in runs/act3-smoke/."
echo "For the full experiment see docs/ACT3_JOURNAL.md."
