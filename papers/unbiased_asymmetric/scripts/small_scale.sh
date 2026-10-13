#!/bin/sh
# Small-scale stage: all 5 arms, 5 runs each, on Shopping-5 (2M = paper length) and Heaven-Hell-3 (1M, a 10x-smaller
# proxy of the paper's 10M), then the reduction -> runs/small/summary.json. One process at a time.
# The stored small-scale runs were trained on an RTX 3090 with the old TF32 matmul default; to reproduce that,
# set `matmul_precision: null` under `overrides` in the two configs (see README).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
mkdir -p runs/small/logs
for cfg in small_shopping5 small_heavenhell3; do
  for m in a2c_asym_hs a2c_asym_s a2c a2c_react_2 a2c_react_4; do
    $PY src/train.py --config src/configs/$cfg.yaml --method $m > runs/small/logs/${cfg}_$m.log 2>&1
  done
done
$PY src/eval.py --tag small
