#!/bin/sh
# Authors'-module reference runs (Hopper-v4, 1M steps, CPU, single-threaded TF; 26-36 min per run), sequential.
# Needs the authors' checkout and its TF venv in $OFFICIAL_ROOT (README, Checked against). Outputs: runs/official_loop/.
set -e
PAPER=$(cd "$(dirname "$0")/../.." && pwd)
OFFICIAL_ROOT=${OFFICIAL_ROOT:-"$PAPER/external/official"}
OUT="$PAPER/runs/official_loop"; mkdir -p "$OUT"
export OFFICIAL_ROOT CUDA_VISIBLE_DEVICES= TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1
cd "$OFFICIAL_ROOT"
for job in base:0 base:1 rec:0 rec:1 base_pre:0 base_pre:1 base_pre:2 released:0 released:1 released:2 rec_pre:0 rec_pre:1 rec_pre:2 base:2 rec:2; do
  cfg=${job%:*}; seed=${job#*:}
  .venv/bin/python "$PAPER/scripts/official/official_loop.py" Hopper-v4 $cfg $seed "$OUT/hopper_${cfg}_s${seed}.json"
done
# HalfCheetah-v4 400k cause test (REC vs init std 1.0 vs lr 1e-4, seeds 0-1; 11-14 min each)
for cfg in rec_pre rec_pre_std1 rec_pre_lr1em4; do for seed in 0 1; do
  .venv/bin/python "$PAPER/scripts/official/official_loop.py" HalfCheetah-v4 $cfg $seed "$OUT/halfcheetah_${cfg}_s${seed}_400k.json" 400000
done; done
cd "$PAPER" && ${PY:-"uv run python"} scripts/official_loop_summary.py && ${PY:-"uv run python"} scripts/official_cause_test.py
