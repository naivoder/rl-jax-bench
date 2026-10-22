#!/bin/sh
# Third-party reference runs (100 000 online slots each, 22-45 min per run on one CPU core), sequential.
# Arms: default; nlown = --no_leak --own_reward; nlownsamp20 = nlown + --sample_beta 20; paper = --paper_scalars.
# Needs sh scripts/official/setup.sh first. Output runs/official/shk_<arm>N<n>K<k>_s<seed>/ (trace.npz, summary.json).
set -e
cd "$(dirname "$0")/../.."
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
TFPY=${TFPY:-$OFFICIAL_ROOT/.venv/bin/python}
export OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
run() {  # tag users channels seed extra-flags...
  tag=$1 n=$2 k=$3 s=$4; shift 4
  out=runs/official/shk_${tag}N${n}K${k}_s${s}
  [ -f "$out/summary.json" ] && return 0
  mkdir -p runs/official
  $TFPY scripts/official/run_shk.py --users $n --channels $k --seed $s --slots 100000 --out "$out" "$@" > "$out.log" 2>&1
}
for s in 0 1 2; do run "" 2 2 $s; run "" 4 2 $s; done
for s in 0 1 2 3 4; do run "" 3 2 $s; done
for s in 0 1 2; do run nlown_ 2 2 $s --no_leak --own_reward; run nlown_ 3 2 $s --no_leak --own_reward; done
for s in 0 1 2; do run nlownsamp20_ 2 2 $s --no_leak --own_reward --sample_beta 20; done
for s in 0 1; do run nlownsamp20_ 3 2 $s --no_leak --own_reward --sample_beta 20; done
for s in 0 1 2; do run paper_ 3 2 $s --paper_scalars; done
${PY:-uv run python} scripts/official/aggregate_shk.py
