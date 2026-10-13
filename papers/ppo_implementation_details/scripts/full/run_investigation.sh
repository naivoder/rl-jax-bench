#!/bin/sh
# Acrobot investigations after the full-scale runs: 40 more seeds (11-50) of separate/shared, and the success rate of
# a uniform-random policy on our Acrobot env.
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-"uv run python"}
L=runs/full/logs
mkdir -p $L
$PY -m src.train --config src/configs/classic_control.yaml --tag full_manyseeds --envs Acrobot-v1 --arms separate,shared \
  --set "seeds=[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]" \
  > $L/acrobot_manyseeds.log 2>&1
$PY -m scripts.full.random_policy_acrobot 4000 > $L/random_policy_acrobot.log 2>&1
