#!/bin/sh
# Small-scale runs (1/5 of the paper's smallest horizon), every arm. One process per experiment; seeds vmapped.
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-"uv run python"}
L=runs/small/logs
mkdir -p $L
$PY -m src.train --config src/configs/classic_control.yaml --tag small --envs CartPole-v1,Acrobot-v1,MountainCar-v0 \
  --arms separate,shared,advnorm_batch,advnorm_ddof0,no_vclip,trunc_bootstrap \
  --set total_timesteps=100000 --set seeds=[1,2,3,4,5] > $L/classic.log 2>&1
$PY -m src.train --config src/configs/pendulum.yaml --tag small \
  --arms default,no_rscale,no_vclip,gymnasium029,trunc_bootstrap,advnorm_batch \
  --set total_timesteps=200000 --set seeds=[1,2,3] > $L/pendulum.log 2>&1
$PY -m src.train --config src/configs/lstm_masked_cartpole.yaml --tag small_lstm \
  --arms lstm,lstm_noreset,lstm_zero_train,ff_masked,ff_full \
  --set total_timesteps=100000 --set seeds=[1,2,3,4,5] > $L/lstm.log 2>&1
$PY -m src.train --config src/configs/brax_hopper.yaml --tag small --arms default \
  --set total_timesteps=51200 --set seeds=[1,2] --set chunk_updates=5 > $L/brax_hopper.log 2>&1
