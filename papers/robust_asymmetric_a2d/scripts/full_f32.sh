#!/bin/sh
# Full-scale compact arms in float32 (CPU). Superseded by the float64 runs of scripts/full_f64.sh: in float32 the TRPO
# step of a near-deterministic flat policy is refused forever (see README, "dtype"). Kept for the record behind
# results/full_report_f32.json. Sequential; the original spread these over three pinned-core queues.
# Ran before the TRPO fallbacks and the log-likelihood clamp existed; kl_fallback=false loglik_clamp=false reproduces
# that code path with the current trainer (both options change nothing until they fire).
set -u
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
PY=${PY:-"uv run python"}
L=runs/full/logs_cpu; mkdir -p $L
run() { exp=$1; shift; echo "[INFO] $(date +%T) start $exp $*"; t0=$(date +%s)
  $PY -m src.train --exp $exp "$@" kl_fallback=false loglik_clamp=false > $L/${exp}_$(echo "$*" | tr ' /=' '___').log 2>&1
  echo "[INFO] $(date +%T) end rc=$? wall=$(( $(date +%s) - t0 ))s $exp $*"; }
for E in FL TD; do
  run full --env $E --arm rl_mdp --seeds 0-19 --set
  run full --env $E --arm ail_compact --seeds 0-19 --set
  run full --env $E --arm a2d_compact --seeds 0-19 --set
  run full --env $E --arm rl_compact --seeds 0-19 --set
done
for E2 in TD1 TD2 TD3; do
  run full_e2 --env $E2 --arm a2d_q --seeds 0-19 --set
  run full_e2 --env $E2 --arm a2d_noq --seeds 0-19 --set
done
for LM in 0 0.25 0.5 0.75 1.0; do
  run full_e3 --env TD2 --arm a2d_compact+lam$LM --seeds 0-9 --iters 250 --set lam_a2d=$LM lam_rl=$LM
done
for E in FL TD; do
  run full_abl --env $E --arm a2d_compact+beta0 --seeds 0-9 --set beta_decay=0
  run full_abl --env $E --arm a2d_compact+beta095 --seeds 0-9 --set beta_decay=0.95
  run full_abl --env $E --arm a2d_compact+critic_state --seeds 0-9 --set critic=state
  run full_abl --env $E --arm a2d_compact+critic_mix05 --seeds 0-9 --set critic=mix05
done
echo "[INFO] $(date +%T) FULL_F32_DONE"
