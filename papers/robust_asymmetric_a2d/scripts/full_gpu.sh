#!/bin/sh
# Full-scale image arms (GPU, float32, 3 seeds). The original run had a GPU-time cap; the estimates (s) below are the
# measured RTX 3090 wall times that sized it. One process at a time.
# Ran before the TRPO fallbacks and the log-likelihood clamp existed; kl_fallback=false loglik_clamp=false reproduces
# that code path with the current trainer (both options change nothing until they fire).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
L=runs/full/logs; mkdir -p $L
run() { echo "[INFO] $(date +%T) start $*"; t0=$(date +%s)
  $PY -m src.train --exp full "$@" --set kl_fallback=false loglik_clamp=false > $L/$(echo "$*" | tr ' /=' '___').log 2>&1
  echo "[INFO] $(date +%T) end rc=$? wall=$(( $(date +%s) - t0 ))s $*"; }
run --env FL --arm a2d_image --seeds 0-2 --iters 80     # ~241 s
run --env TD --arm a2d_image --seeds 0-2 --iters 80     # ~203 s
run --env FL --arm rl_asym --seeds 0-2 --iters 200      # ~203 s
run --env TD --arm rl_asym --seeds 0-2 --iters 200      # ~228 s
run --env FL --arm rl_image --seeds 0-2 --iters 200     # ~405 s
run --env TD --arm rl_image --seeds 0-2 --iters 200     # ~416 s
echo "[INFO] $(date +%T) FULL_GPU_DONE"
