#!/bin/sh
# Long-budget recheck: resume full-scale paper-config runs from their 2,500-iteration checkpoints to 10,000
# iterations, config otherwise unchanged (state / multi seeds 0-4, maml seeds 0-2). The original runs resumed from
# checkpoints copied out of runs/full; copy runs/full/<arm>/seed_<n> into runs/recheck/long/<arm>/ first.
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
set -e
for arm in state multi; do $PY -m src.train --arm $arm --seeds 0,1,2,3,4 --out runs/recheck/long --set iterations=10000; done
$PY -m src.train --arm maml --seeds 0,1,2 --out runs/recheck/long --set iterations=10000
# authors' multi-critic schedule at the paper config
$PY -m src.train --arm multi --seeds 0,1,2 --out runs/recheck/mv_official --set mv_schedule=official
$PY scripts/heuristics_recheck.py
$PY scripts/recheck_report.py
echo LONG_BUDGET_DONE
