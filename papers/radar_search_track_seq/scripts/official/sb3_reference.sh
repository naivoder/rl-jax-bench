#!/bin/sh
# Reference run: the authors' SB3 / imitation forks driving our environment, Flat arm, {paper, sb3} settings x seeds 0-2,
# 100,000 steps, 32 evaluation scenarios, CPU. Needs sh scripts/official/setup.sh first. Then:
#   uv run python scripts/official_reference_report.py
set -eu
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
PY="$OFFICIAL_ROOT/.venv-sb3/bin/python"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
mkdir -p runs/official/sb3_ref
for setting in paper sb3; do
  for seed in 0 1 2; do
    "$PY" -I scripts/official/sb3_reference_run.py --setting $setting --seed $seed --steps 100000 --n-eval 32 \
      --out runs/official/sb3_ref > runs/official/sb3_ref/${setting}_seed${seed}.log 2>&1
  done
done
