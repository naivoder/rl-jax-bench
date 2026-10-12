#!/bin/sh
# Landscape rerun (E4): step phase (re-derives the identical steps plus the per-minibatch PPO surface), then the
# true-reward phase with the whole-pair-budget estimator. CPU. Usage: sh scripts/landscape_rerun.sh <arm> <its...>
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
A=$1; shift
$PY -m src.eval --run runs/full/$A --its "$@" --exps landscape --seeds 0 1 2 3 4 --seed_idx 0 --land_seed_idx 0 \
  --chunk 10000 --land_few 2048 --land_many 100000 --land_step_chunk 2000 --land_grid 7 --land_phase steps --tag fullr_landsteps
$PY -m src.eval --run runs/full/$A --its "$@" --exps landscape --seeds 0 1 2 3 4 --seed_idx 0 --land_seed_idx 0 \
  --land_true_pairs 100000 --land_true_chunk 2000 --land_grid 7 --land_phase true --land_steps_from fullr_landsteps \
  --land_point_batch 2 --actor_batch 25 --agent_batch 1 --tag fullr_land
