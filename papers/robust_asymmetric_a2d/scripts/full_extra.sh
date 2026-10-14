#!/bin/sh
# Follow-up runs after the full-scale results were reviewed: CPU, float64, TRPO fallbacks on. Sequential; the
# original spread these over four to seven pinned-core lanes, and the runs listed at the end of the original lanes
# that had any TRPO fallback event were re-run once the entropy fallback landed (this script runs the final code once).
#   E3 in the App. B.1 configuration (a2d_noq: beta 0, KL 1e-3, entropy 0.02 on the surrogate, MC values), lambda in
#     {0, .25, .5, .75, 1}, TD2, 20 seeds, 250 iterations (the E3 headline)
#   E3 under the main args (the official lambda launcher), 10 seeds; E1 TD A2D(Compact), 20 seeds
#   ablations on FL and TD, 10 seeds; E2 TD2 / TD3, 20 seeds
# These runs predate the log-likelihood clamp, so loglik_clamp=false reproduces them.
set -u
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
PY=${PY:-"uv run python"}
L=runs/full_extra/logs; mkdir -p $L
run() { exp=$1; shift; echo "[INFO] $(date +%T) start $exp $*"; t0=$(date +%s)
  $PY -m src.train --exp $exp "$@" dtype=float64 kl_fallback=true loglik_clamp=false \
    > $L/${exp}_$(echo "$*" | tr ' /=' '___').log 2>&1
  echo "[INFO] $(date +%T) end rc=$? wall=$(( $(date +%s) - t0 ))s $exp $*"; }
for LM in 0 0.25 0.5 0.75 1.0; do
  run full_extra_e3b1 --env TD2 --arm a2d_noq+lam$LM --seeds 0-19 --iters 250 --set lam_a2d=$LM lam_rl=$LM
done
run full_extra_main --env TD --arm a2d_compact --seeds 0-19 --set
for LM in 0 0.25 0.5 0.75 1.0; do
  run full_extra_e3main --env TD2 --arm a2d_compact+lam$LM --seeds 0-9 --iters 250 --set lam_a2d=$LM lam_rl=$LM
done
for E in TD FL; do
  run full_extra_abl --env $E --arm a2d_compact+beta0 --seeds 0-9 --set beta_decay=0
  run full_extra_abl --env $E --arm a2d_compact+beta095 --seeds 0-9 --set beta_decay=0.95
  run full_extra_abl --env $E --arm a2d_compact+critic_state --seeds 0-9 --set critic=state
  run full_extra_abl --env $E --arm a2d_compact+critic_mix05 --seeds 0-9 --set critic=mix05
  run full_extra_abl --env $E --arm a2d_compact+paperest --seeds 0-9 --set estimator=paper
  run full_extra_abl --env $E --arm a2d_compact+own --seeds 0-9 --set critic_targets=own
  run full_extra_abl --env $E --arm a2d_compact+critic_mix05own --seeds 0-9 --set critic=mix05 critic_targets=own
done
for E2 in TD2 TD3; do
  run full_extra_e2 --env $E2 --arm a2d_noq --seeds 0-19 --set
  run full_extra_e2 --env $E2 --arm a2d_q --seeds 0-19 --set
done
echo "[INFO] $(date +%T) FULL_EXTRA_DONE"
