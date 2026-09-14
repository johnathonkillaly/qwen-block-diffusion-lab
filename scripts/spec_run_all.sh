#!/usr/bin/env bash
# Act IV-S: run the measurement suite back to back, in one machine state.
#
# Sequential on purpose. Two of these running at once would contend for the GPU and
# every wall-clock number in both would be worthless (brief §20: competing processes).
# Each stage reloads the 4B backbone, which costs ~13 s and is far cheaper than the
# risk of sharing one process across an hour of measurement.
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_HOME=/Volumes/SHUTTLE
export PYTHONPATH="$PWD/src"
PY=/Users/johnathonkillaly/code/diffusion_project/.venv-unsloth/bin/python
TAG="${TAG:-main}"
REPEATS="${REPEATS:-3}"

run () {
  local name="$1"; shift
  echo "=============================================================="
  echo "[run_all] $name  ($(date '+%H:%M:%S'))"
  echo "=============================================================="
  "$PY" scripts/spec_decode.py --tag "$TAG" --repeats "$REPEATS" "$@" \
    2>&1 | tee -a "runs/spec-${name}.log"
}

run ksweep     --tokens 128 ksweep --block-sizes 1,2,4,8,16
run controls   --tokens 128 controls --k 4
run confidence --tokens 128 confidence --k 4
run context    --tokens 128 context --lengths 512,2048,8192,16384 --block-sizes 2,4,8

echo "[run_all] done $(date '+%H:%M:%S')"
