#!/bin/sh
# Reference curve: the AUTHORS' stream_q_minatar.py (branch 2024 @ 407dca7) on MinAtar 1.0.15
# Breakout-v1, official defaults except total_steps (small-scale budget; eps schedule = 20 % of total_steps as in code).
# Runs isolated in the official venv on CPU, 1 torch thread, one seed after another, from the official runs dir.
# Usage: run_official_q.sh <env_name> <total_steps> <outdir> <seed...>
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/external/official}   # see setup.sh
ENVN=$1; T=$2; OUT=$3; shift 3
mkdir -p "$OUT" && cd "$OUT" || exit 1
for s in "$@"; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OFFICIAL_ROOT"/.venv/bin/python \
    "$OFFICIAL_ROOT"/repo/stream_q_minatar.py --env_name "$ENVN" --seed "$s" \
    --total_steps "$T" --debug > "log_seed$s.txt" 2>&1
done
