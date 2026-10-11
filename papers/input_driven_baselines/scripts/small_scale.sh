#!/bin/sh
# small-scale: 10x-smaller load-balancing proxy (250 iterations = 2e6 timesteps), 3 seeds, all arms.
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
for arm in none state multi maml; do
  $PY -m src.train --config src/configs/small.yaml --arm $arm --seeds 0,1,2 --out runs/small || exit 1
done
$PY -m src.train --config src/configs/maml_literal.yaml --arm maml --seeds 0,1,2 --out runs/small_maml_literal || exit 1
$PY -m src.eval --heuristics --config src/configs/small.yaml --out runs/small/heuristics.json || exit 1
$PY -m src.eval --probe runs/small --out runs/small/probe.json || exit 1
$PY scripts/summarize.py runs/small runs/small_maml_literal:maml_literal || exit 1
echo SMALL_SCALE_DONE
