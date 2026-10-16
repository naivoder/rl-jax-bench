#!/bin/sh
# Full-scale throughput probe: the declared E3 and E6 layouts, short chunks, no learning number read.
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/full/logs
PY=${PY:-"uv run python"}
$PY -m src.train --config src/configs/full_e3_breakout.yaml --tag full_probe --no-ckpt --set total_steps=30000 \
  --set chunk_steps=10000 > runs/full/logs/probe_e3.log 2>&1
$PY -m src.train --config src/configs/full_e6_hopper.yaml --tag full_probe --no-ckpt --set total_steps=15000 \
  --set chunk_steps=5000 > runs/full/logs/probe_e6.log 2>&1
