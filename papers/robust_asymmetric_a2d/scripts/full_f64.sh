#!/bin/sh
# Full-scale compact arms in float64 (CPU), the headline compact results: E1 on FL and TD (20 seeds), E2 Q vs no-Q
# (TD1 20 seeds, TD2/TD3 10 seeds), and the two informative TD ablations (10 seeds). Sequential; the original spread
# these over four pinned-core lanes. These runs predate the TRPO fallbacks being switched on by default, so
# kl_fallback=false reproduces them (the fallbacks are bitwise inert until a non-finite step).
set -u
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
PY=${PY:-"uv run python"}
L=runs/full/logs_f64; mkdir -p $L
run() { exp=$1; shift; echo "[INFO] $(date +%T) start $exp $*"; t0=$(date +%s)
  $PY -m src.train --exp $exp "$@" dtype=float64 kl_fallback=false loglik_clamp=false \
    > $L/${exp}_$(echo "$*" | tr ' /=' '___').log 2>&1
  echo "[INFO] $(date +%T) end rc=$? wall=$(( $(date +%s) - t0 ))s $exp $*"; }
for E in TD FL; do
  run full_f64 --env $E --arm rl_mdp --seeds 0-19 --set
  run full_f64 --env $E --arm ail_compact --seeds 0-19 --set
  run full_f64 --env $E --arm a2d_compact --seeds 0-19 --set
  run full_f64 --env $E --arm rl_compact --seeds 0-19 --set
done
run full_f64_e2 --env TD1 --arm a2d_noq --seeds 0-19 --set
run full_f64_e2 --env TD1 --arm a2d_q --seeds 0-19 --set
for E2 in TD2 TD3; do
  run full_f64_e2 --env $E2 --arm a2d_noq --seeds 0-9 --set
  run full_f64_e2 --env $E2 --arm a2d_q --seeds 0-9 --set
done
run full_f64_abl --env TD --arm a2d_compact+beta095 --seeds 0-9 --set beta_decay=0.95
run full_f64_abl --env TD --arm a2d_compact+critic_state --seeds 0-9 --set critic=state
echo "[INFO] $(date +%T) FULL_F64_DONE"
