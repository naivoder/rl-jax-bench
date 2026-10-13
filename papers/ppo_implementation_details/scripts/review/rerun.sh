#!/bin/sh
# Re-run selected full-scale arms without checkpoints to check bit-for-bit reproducibility (compare with rerun_diff.py).
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-"uv run python"}
$PY -m src.train --config src/configs/classic_control.yaml --tag review_rerun --envs CartPole-v1,Acrobot-v1 --arms separate --set 'seeds=[1,2,3,4,5,6,7,8,9,10]' --no-ckpt
$PY -m src.train --config src/configs/pendulum.yaml --tag review_rerun --envs Pendulum-v1 --arms trunc_bootstrap,no_rscale --set 'seeds=[1,2,3,4,5,6,7,8,9,10]' --no-ckpt
$PY -m src.train --config src/configs/classic_control.yaml --tag review_s9 --envs Acrobot-v1 --arms separate --set 'seeds=[9]' --no-ckpt
