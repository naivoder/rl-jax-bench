#!/bin/sh
# Re-runs after the log-likelihood clamp fix (official Categorical clamp on the TRPO / IS action log-likelihoods).
# CPU, float64, kl_fallback on, loglik_clamp on, official_buffer_seed off, so each run is paired seed for seed with its
# stored pre-fix counterpart and is bit-identical to it until its first clamped sample. Same seeds and iterations as the
# stored runs. Sequential; the original pulled these jobs from a queue over five 2-core lanes, in this order
# (priority, then longest first). Labelled variants at the end are not paired with stored runs.
set -u
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
PY=${PY:-"uv run python"}
L=runs/recheck/logs; mkdir -p $L
BASE="dtype=float64 kl_fallback=true loglik_clamp=true"
run() { exp=$1 env=$2 arm=$3 seeds=$4 iters=$5; shift 5
  echo "[INFO] $(date +%T) start $exp $env $arm"; t0=$(date +%s)
  $PY -m src.train --exp $exp --env $env --arm $arm --seeds $seeds --iters $iters --set $BASE "$@" \
    > $L/$(echo "${exp}_${env}_${arm}" | tr '/' '_').log 2>&1
  echo "[INFO] $(date +%T) end rc=$? wall=$(( $(date +%s) - t0 ))s $exp $env $arm"; }
OFF="official_buffer_seed=false"
run recheck_e2 TD2 a2d_q 0-19 200 $OFF
run recheck_e2 TD3 a2d_q 0-19 200 $OFF
for LM in 0 0.25 0.5 0.75 1.0; do
  run recheck_e3b1 TD2 a2d_noq+lam$LM 0-19 250 $OFF lam_a2d=$LM lam_rl=$LM
done
run recheck_e2 TD3 a2d_noq 0-19 200 $OFF
run recheck_e2 TD2 a2d_noq 0-19 200 $OFF
run recheck_main FL a2d_compact 0-19 200 $OFF
run recheck_main TD a2d_compact 0-19 200 $OFF
for LM in 0 0.25 0.5 0.75 1.0; do
  run recheck_e3main TD2 a2d_compact+lam$LM 0-9 250 $OFF lam_a2d=$LM lam_rl=$LM
done
run recheck_e2 TD1 a2d_q 0-19 200 $OFF
run recheck_e2 TD1 a2d_noq 0-19 200 $OFF
for E in TD FL; do
  run recheck_abl $E a2d_compact+beta0 0-9 200 $OFF beta_decay=0
  run recheck_abl $E a2d_compact+beta095 0-9 200 $OFF beta_decay=0.95
  run recheck_abl $E a2d_compact+critic_state 0-9 200 $OFF critic=state
  run recheck_abl $E a2d_compact+critic_mix05 0-9 200 $OFF critic=mix05
  run recheck_abl $E a2d_compact+paperest 0-9 200 $OFF estimator=paper
  run recheck_abl $E a2d_compact+own 0-9 200 $OFF critic_targets=own
  run recheck_abl $E a2d_compact+critic_mix05own 0-9 200 $OFF critic=mix05 critic_targets=own
done
for E in TD FL; do
  run recheck_rl $E rl_compact 0-19 200 $OFF
done
# labeled variants: the shipped TD launcher's lambda_a2d 0.9; the official initial buffer; state / 50-50 critics at B.1
run recheck_var TD a2d_compact+lam09 0-19 200 $OFF lam_a2d=0.9
run recheck_var TD a2d_compact+bufseed 0-19 200 official_buffer_seed=true
for LM in 0.25 0.5; do
  run recheck_var TD2 a2d_noq+lam$LM+critic_state 0-9 250 $OFF lam_a2d=$LM lam_rl=$LM critic=state
  run recheck_var TD2 a2d_noq+lam$LM+critic_mix05own 0-9 250 $OFF lam_a2d=$LM lam_rl=$LM critic=mix05 critic_targets=own
done
# run afterwards by hand: the same lambda 0.9 variant without the clamp
run recheck_var TD a2d_compact+lam09_noclamp 0-19 200 $OFF lam_a2d=0.9 loglik_clamp=false
echo "[INFO] $(date +%T) RECHECK_DONE"
