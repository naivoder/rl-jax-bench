#!/bin/sh
# Small-scale: the paper's smallest setting (E1, E2 at T = 1e4), all arms, 3 seeds x 500 replications on seeds
# 100-102 (disjoint from the Full-scale headline seeds 0-4), then the labelled D-UCB diagnostic (README Small-scale).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
mkdir -p runs/logs
for e in e1 e2; do
  $PY -m src.train --config src/configs/$e.yaml --seeds 100,101,102 --reps 500 --out runs/small/$e
done
$PY -m src.eval --small runs/small --out runs/small/summary.json
for e in e1 e2; do
  $PY -m src.train --config src/configs/diag_ducb_$e.yaml --seeds 100,101,102 --reps 500 --out runs/small_diag/$e
  $PY -m src.eval --summarize runs/small_diag/$e --out runs/small_diag/$e/summary.json
done
$PY scripts/small_verdict.py
