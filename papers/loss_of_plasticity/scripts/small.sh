#!/bin/sh
# small-scale: small-scale runs of every arm (10x fewer SCR seeds; MNIST 15 of 150 tasks, 3 sweep seeds).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
$PY -m src.train --config src/configs/scr_b10.yaml --set n_seeds=10 tag=small_scr_b10
$PY -m src.train --config src/configs/scr_linear.yaml --set n_seeds=10 tag=small_scr_linear
$PY -m src.train --config src/configs/scr_d12.yaml --set n_seeds=10 tag=small_scr_d12
$PY -m src.train --config src/configs/scr_methods.yaml --set n_seeds=10 tag=small_scr_methods
$PY -m src.train --config src/configs/mnist_reduced.yaml --set n_tasks=15 n_seeds=3 seed_offset=0 tag=small_mnist
$PY -m src.train --config src/configs/mnist_bp_steps.yaml --set n_tasks=15 n_seeds=3 seed_offset=0 tag=small_mnist_bp_steps
$PY -m src.train --config src/configs/scr_d12_mt.yaml --set n_seeds=10 tag=small_scr_d12_mt   # D.12 maturity probe
echo SMALL_RUNS_DONE
