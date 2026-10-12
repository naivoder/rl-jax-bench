#!/bin/sh
# Small-scale (10x-smaller proxy of the full protocol): 49 training iterations (the first 49 of the 488-iteration run:
# same keys and LR schedule), checkpoints {0,15,30,45} (= paper's {0,150,300,450} / 10), 3 seeds, all 4 (env, alg)
# arms; instruments with every size and sample divided by 10 (reference 1e5, V* 1e5, landscape true reward 1e4/point,
# many-sample step 1e4) and chunk 2000.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
mkdir -p runs/small/logs
i=0
for cfg in "hopper ppo" "hopper trpo" "walker2d ppo" "walker2d trpo"; do
  set -- $cfg
  $PY -m src.train --env $1 --alg $2 --seeds 0 1 2 --iters 49 --out runs/small \
      --ckpt_iters 0 15 30 45 --resume_every 25 > runs/small/logs/train_$1_$2.log 2>&1 &
  i=$((i+1))
done
wait
for cfg in hopper_ppo hopper_trpo walker2d_ppo walker2d_trpo; do
  # PPO's many-sample step needs n divisible by its 32 minibatches: 16000 (=8x2000) for PPO, 10000 for TRPO
  MANY=10000; case $cfg in *ppo) MANY=16000;; esac
  $PY -m src.eval --run runs/small/$cfg --its 0 15 45 --exps grad value baseline landscape \
      --grad_sizes 100 1000 2000 10000 --ref_pairs 100000 --chunk 2000 --vstar_pairs 100000 \
      --base_sizes 100 1000 2000 4000 6000 8000 10000 --land_few 2048 --land_many $MANY --land_true_pairs 10000 \
      --land_grid 7 --tag small > runs/small/logs/eval_$cfg.log 2>&1 &
done
for e in hopper walker2d; do   # E5 no-baseline arm (training only)
  $PY -m src.train --env $e --alg ppo --adv_baseline zero --seeds 0 1 2 --iters 49 --out runs/small --ckpt_iters \
      > runs/small/logs/train_${e}_ppo-nobaseline.log 2>&1 &
done
wait
$PY scripts/small_scale_report.py
