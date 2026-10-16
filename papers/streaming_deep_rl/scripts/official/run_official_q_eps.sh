#!/bin/sh
# As run_official_q.sh but with an explicit exploration fraction (paper text: 5 %; code default 20 %).
# Usage: run_official_q_eps.sh <env_name> <total_steps> <exploration_fraction> <outdir> <seed...>
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/external/official}   # see setup.sh
ENVN=$1; T=$2; EF=$3; OUT=$4; shift 4
mkdir -p "$OUT" && cd "$OUT" || exit 1
for s in "$@"; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$OFFICIAL_ROOT"/.venv/bin/python \
    "$OFFICIAL_ROOT"/repo/stream_q_minatar.py --env_name "$ENVN" --seed "$s" \
    --total_steps "$T" --exploration_fraction "$EF" --debug > "log_seed$s.txt" 2>&1
done
