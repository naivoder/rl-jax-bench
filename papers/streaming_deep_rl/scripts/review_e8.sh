#!/bin/sh
# Review E8 chain on CPU: (1) the full-scale E8 program re-run unchanged (same 60-lane layout => bitwise-identical
# trajectories) to recompute the aligned metric from the full series; (2) the B1 sensitivity arm (classic_td_keepnorm).
cd "$(dirname "$0")/.." || exit 1
export JAX_PLATFORMS=cpu
PY=${PY:-"uv run python"}
mkdir -p runs/review/logs
$PY -m src.train --config src/configs/e8_ett.yaml --tag review > runs/review/logs/e8.log 2>&1
$PY -m src.train --config src/configs/review_e8_sens.yaml --tag review_sens > runs/review/logs/e8_sens.log 2>&1
