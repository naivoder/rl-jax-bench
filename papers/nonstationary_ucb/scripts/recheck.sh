#!/bin/sh
# Recheck runs (see README). Usage: scripts/recheck.sh identity | sweepd | tunedd | gapsweep | all
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
case "$1" in
  identity)  # RC3: eta_mult default on the GPU vs the stored Full-scale EXP3.S per-run regrets (seed 0 x 2000)
    for e in e1 e2; do $PY -m src.train --config src/configs/$e.yaml --seeds 0 --reps 2000 --policies exp3s --out runs/recheck/identity_$e; done ;;
  sweepd)
    for e in e1 e2; do $PY -m src.train --seeds 0,1,2,3,4 --batch-seeds --config src/configs/sweepd_$e.yaml --reps 1000 --out runs/recheck/sweepd_$e; done ;;
  tunedd)
    $PY scripts/make_recheck_configs.py tuned
    for e in e1 e2; do $PY -m src.train --seeds 20,21,22,23,24 --batch-seeds --config src/configs/tunedd_$e.yaml --reps 2000 --out runs/recheck/tunedd_$e; done ;;
  gapsweep)
    $PY scripts/x1_gapsweep.py ;;
  all)
    for s in identity sweepd tunedd gapsweep; do "$0" $s; done ;;
  *) echo "usage: $0 identity|sweepd|tunedd|gapsweep|all"; exit 2 ;;
esac
