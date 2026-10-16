#!/bin/sh
# Reference curve: the AUTHORS' stream_ac_continuous.py (branch 2024 @ 407dca7) on real MuJoCo
# (gymnasium 0.29.1, mujoco 2.3.7), official defaults (= paper App. F). CPU, 1 torch thread per process, one seed after another.
# Usage: run_official_ac.sh <env_name> <total_steps> <outdir> <seed...>
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/external/official}   # see setup.sh
ENVN=$1; T=$2; OUT=$3; shift 3
mkdir -p "$OUT" && cd "$OUT" || exit 1
for s in "$@"; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OFFICIAL_ROOT"/.venv/bin/python \
    "$OFFICIAL_ROOT"/repo/stream_ac_continuous.py --env_name "$ENVN" --seed "$s" \
    --total_steps "$T" --debug > "log_seed$s.txt" 2>&1
done
