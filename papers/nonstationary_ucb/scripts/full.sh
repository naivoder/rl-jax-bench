#!/bin/sh
# Full-scale (see README). Headline seeds 0-4 x 2000 reps, all seeds batched per policy.
# Usage: scripts/full.sh core | sweep | tuned | x1 <grid> [points] | x4 <Ts> | report
#        post-hoc follow-up (see README): sweepj | sweepjx | tunedj | abl2 | perrun
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
S=0,1,2,3,4
TR="$PY -m src.train --seeds $S --batch-seeds"
mkdir -p runs/logs
case "$1" in
  core)
    for c in e1 e2 padb_e1 padb_e2 e1b e2b padb_e1b padb_e2b abl_e1 abl_e2 e1_text; do
      $TR --config src/configs/$c.yaml --reps 2000 --out runs/full/$c
    done
    $TR --config src/configs/e1.yaml --reps 2000 --tie lowest --policies ucb1,exp3s,ducb,swucb --out runs/full/tie_e1
    $TR --config src/configs/e2.yaml --reps 2000 --tie lowest --out runs/full/tie_e2 ;;
  sweep)
    for c in sweep_e1 sweep_e2; do $TR --config src/configs/$c.yaml --reps 1000 --out runs/full/$c; done ;;
  tuned)
    $PY scripts/tuned.py
    for c in tuned_e1 tuned_e2; do
      $PY -m src.train --seeds 20,21,22,23,24 --batch-seeds --config src/configs/$c.yaml --reps 2000 --out runs/full/$c
    done ;;
  x1)
    $PY -m src.sweep x1 --grid "$2" ${3:+--points $3} --seeds $S --reps 1000 --out runs/x1 --ckpt runs/x1_ckpt ;;
  x4)
    $PY -m src.sweep x4 --Ts "$2" --seeds $S --reps 1000 --out runs/x4 ;;
  sweepj)   # (post-hoc): joint grids, same seeds 0-4 x 1000
    for c in sweepj_e1 sweepj_e2; do $TR --config src/configs/$c.yaml --reps 1000 --out runs/full/$c; done ;;
  sweepjx)  # follow-up: small-xi extension of the joint D-UCB / SW-UCB grids (merged with sweepj)
    for c in sweepjx_e1 sweepjx_e2; do $TR --config src/configs/$c.yaml --reps 1000 --out runs/full/$c; done ;;
  tunedj)   # every arm at its JOINT-grid argmin (sweepj + sweepjx), fresh seeds 20-24 x 2000
    $PY scripts/tuned.py --joint
    for c in tunedj_e1 tunedj_e2; do
      $PY -m src.train --seeds 20,21,22,23,24 --batch-seeds --config src/configs/$c.yaml --reps 2000 --out runs/full/$c
    done ;;
  abl2)     # gamma = 1 with padding B; EXP3.S S = 7 on E2 (seeds 0-4 x 2000, CRN with e1/e2)
    for c in abl2_e1 abl2_e2; do $TR --config src/configs/$c.yaml --reps 2000 --out runs/full/$c; done ;;
  perrun)   # per-run regret at the intermediate read-off times (same keys as the headline)
    $PY scripts/perrun_curves.py ;;
  report)
    $PY scripts/report.py ;;
  *) echo "usage: $0 core|sweep|tuned|x1 <grid> [points]|x4 <Ts>|sweepj|sweepjx|tunedj|abl2|perrun|report"; exit 2 ;;
esac
