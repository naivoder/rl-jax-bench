#!/bin/sh
# MAML gradient-variance probes at the training k = 16: full-scale paper config, tuned config, and small scale.
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
set -e
$PY -m src.eval --probe runs/full --arms maml --out runs/full/probe_maml_k16.json
$PY -m src.eval --probe runs/full_tuned --arms maml --out runs/full_tuned/probe_maml_k16.json
$PY -m src.eval --probe runs/small --arms maml --out runs/small/probe_maml_k16.json
echo PROBES_DONE
