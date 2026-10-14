#!/bin/sh
# Small-scale run of every arm (5 seeds compact, 3 seeds image, short horizons). Image arms want a GPU.
# Per-arm logs in runs/small/logs/.
# Ran before the TRPO fallbacks and the log-likelihood clamp existed; kl_fallback=false loglik_clamp=false reproduces
# that code path with the current trainer (both options change nothing until they fire).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
L=runs/small/logs; mkdir -p $L
run() { echo "[INFO] $(date +%T) start $*"; $PY -m src.train --exp small "$@" --set kl_fallback=false loglik_clamp=false > $L/$(echo "$*" | tr ' /=' '___').log 2>&1; echo "[INFO] $(date +%T) end rc=$? $*"; }
for E in FL TD; do
  run --env $E --arm rl_mdp --seeds 0-4 --iters 200
  run --env $E --arm a2d_compact --seeds 0-4 --iters 200
  run --env $E --arm rl_compact --seeds 0-4 --iters 200
  run --env $E --arm ail_compact --seeds 0-4 --iters 200
done
run --env TD1 --arm a2d_q --seeds 0-4 --iters 120
run --env TD1 --arm a2d_noq --seeds 0-4 --iters 120
for E in FL TD; do
  run --env $E --arm a2d_image --seeds 0-2 --iters 60
  run --env $E --arm ail_image --seeds 0-2 --iters 60
  run --env $E --arm rl_asym --seeds 0-2 --iters 60
  run --env $E --arm rl_image --seeds 0-2 --iters 60
done
echo "[INFO] SMALL_DONE"
