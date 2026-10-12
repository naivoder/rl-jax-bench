#!/bin/sh
# Full-scale training (E0/E2/E5): full 488 iterations x 2048 steps, 5 seeds (0-4) vmapped per process, CPU,
# checkpoints {0,150,300,450}; E5 no-baseline PPO arms (no checkpoints). 6 processes in parallel.
set -e
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
mkdir -p runs/full/logs
i=0
for cfg in "hopper ppo" "hopper trpo" "walker2d ppo" "walker2d trpo"; do
  set -- $cfg
  $PY -m src.train --env $1 --alg $2 --seeds 0 1 2 3 4 --iters 488 \
      --out runs/full --ckpt_iters 0 150 300 450 --resume_every 50 > runs/full/logs/train_$1_$2.log 2>&1 &
  i=$((i+1))
done
for e in hopper walker2d; do
  $PY -m src.train --env $e --alg ppo --adv_baseline zero --seeds 0 1 2 3 4 \
      --iters 488 --out runs/full --ckpt_iters --resume_every 50 > runs/full/logs/train_${e}_ppo-nobaseline.log 2>&1 &
  i=$((i+1))
done
wait
