#!/bin/sh
# 20k-step, 2-seed, all-arm smoke run of the other three games (pipeline + evaluation cost probe, esp. SpaceInvaders' 10,000-step cap).
# Run from anywhere; outputs go to runs/dev/. On a GPU the original runs also set
#   XLA_FLAGS=--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,WHILE
PY=${PY:-"uv run python"}
cd "$(dirname "$0")/.."
mkdir -p runs/dev
for g in Asterix Seaquest SpaceInvaders; do
  $PY -m src.train --game $g --seeds 2 --steps 20000 --pretrain 2000 --out runs/dev/smoke_$g --ckpt-every 0 > runs/dev/smoke_$g.log 2>&1
done
