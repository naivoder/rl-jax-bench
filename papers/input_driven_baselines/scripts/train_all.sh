#!/bin/sh
# Full-scale load-balancing runs (2,500 iterations = 2e7 timesteps under the 16 x 500 reading), paper config,
# 10 seeds per arm, plus the two mechanism ablations (5 seeds each). Runs sequentially.
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
S=${SEEDS:-0,1,2,3,4,5,6,7,8,9}
A=${ABL_SEEDS:-0,1,2,3,4}
set -e
for arm in state none multi maml; do
  $PY -m src.train --arm $arm --seeds $S --out runs/full
done
$PY -m src.train --arm multi_shuffled --seeds $A --out runs/full_ablation
$PY -m src.train --arm maml --seeds $A --out runs/full_ablation_noadapt --set meta_inner_steps=0
# entropy-decay diagnostic (scored on the test set)
for arm in state maml; do
  $PY -m src.train --arm $arm --seeds 0,1,2 --out runs/full_diag_ent1000 --set entropy_decay_iters=1000
done
# matched-policy gradient-variance probes
$PY -m src.eval --probe runs/full --out runs/full/probe_main.json
$PY -m src.eval --probe runs/full_ablation --out runs/full_ablation/probe.json
$PY -m src.eval --probe runs/full_ablation_noadapt --out runs/full_ablation_noadapt/probe.json
$PY scripts/check_overlap.py --seeds 0-9,100,101 --iters 2500
echo TRAIN_ALL_DONE
