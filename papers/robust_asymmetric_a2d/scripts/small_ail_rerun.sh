#!/bin/sh
# Re-run of the small-scale AIL arms after the pre-training fix (AIL pre-training runs 2 projection epochs, as the
# official code does). Reuses the RL(MDP) experts trained by scripts/small.sh.
# Ran before the TRPO fallbacks and the log-likelihood clamp existed; kl_fallback=false loglik_clamp=false reproduces
# that code path with the current trainer (both options change nothing until they fire).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
L=runs/small_ail_rerun/logs; mkdir -p $L
run() { echo "[INFO] $(date +%T) start $*"; $PY -m src.train --exp small_ail_rerun "$@" > $L/$(echo "$*" | tr ' /=' '___').log 2>&1; echo "[INFO] $(date +%T) end rc=$? $*"; }
for E in FL TD; do
  run --env $E --arm ail_compact --seeds 0-4 --iters 200 --set expert_ckpt=$PWD/runs/small/$E/expert_best.npz kl_fallback=false loglik_clamp=false
done
for E in FL TD; do
  run --env $E --arm ail_image --seeds 0-2 --iters 60 --set expert_ckpt=$PWD/runs/small/$E/expert_best.npz kl_fallback=false loglik_clamp=false
done
echo "[INFO] SMALL_AIL_RERUN_DONE"
