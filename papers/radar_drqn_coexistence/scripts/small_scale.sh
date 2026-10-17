#!/bin/sh
# Toy checks and the 10x-smaller proxy of the whole protocol (50 + 10 CPIs, 5 seeds, every experiment and arm).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
$PY src/toys.py --out runs/toys
$PY scripts/toy_diag.py
$PY src/train.py --config src/configs/small.yaml --out runs/small
$PY src/eval.py runs/small
