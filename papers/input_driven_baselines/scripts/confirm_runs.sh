#!/bin/sh
# Post-review runs (declared before launch):
#   tuned config on fresh seeds 10-19 (state, multi, maml)    -> runs/full_confirm
#   obs_mode=work sensitivity, tuned, state + multi, seeds 0-2 -> runs/sens_obs_work
#   literal sum inner loss at lr 1e-4 (expected NaN abort)     -> runs/maml_sum_literal
#   MAML probes at the training k = 16                         -> scripts/probe_maml_k16.sh
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
set -e
for A in state multi maml; do
  $PY -m src.train --config src/configs/tuned.yaml --arm $A --seeds 10,11,12,13,14,15,16,17,18,19 \
      --out runs/full_confirm --set eval_start=2400
done
for A in state multi; do
  $PY -m src.train --config src/configs/tuned.yaml --arm $A --seeds 0,1,2 --out runs/sens_obs_work \
      --set obs_mode=work --set eval_start=2400
done
# expected and observed: NaN-guard abort at iteration 12
$PY -m src.train --config src/configs/small.yaml --arm maml --seed 0 --out runs/maml_sum_literal \
    --set meta_inner_reduction=sum --set meta_inner_lr=1e-4 --set meta_inner_clip=0 --set iterations=30 || true
$PY scripts/check_overlap.py --seeds 10-19 --iters 2500 --out runs/full_confirm_overlap_check.json
sh scripts/probe_maml_k16.sh
$PY scripts/summarize.py runs/full_confirm
$PY scripts/summarize.py runs/sens_obs_work
echo CONFIRM_RUNS_DONE
