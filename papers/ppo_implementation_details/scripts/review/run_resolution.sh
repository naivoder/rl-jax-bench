#!/bin/sh
# The post's own LSTM recipe on the masked-CartPole proxy: lstm_atari (eps 0.1, no reward scaling) and lstm_eps01.
# (The original run skipped lstm_eps01 if lstm_atari took longer than 108 s of GPU wall time; here both run.)
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-"uv run python"}
$PY -m src.train --config src/configs/lstm_masked_cartpole.yaml --tag resolution_lstm --envs CartPole-v1 --arms lstm_atari --set 'seeds=[1,2,3,4,5,6,7,8,9,10]' --no-ckpt
$PY -m src.train --config src/configs/lstm_masked_cartpole.yaml --tag resolution_lstm --envs CartPole-v1 --arms lstm_eps01 --set 'seeds=[1,2,3,4,5,6,7,8,9,10]' --no-ckpt
