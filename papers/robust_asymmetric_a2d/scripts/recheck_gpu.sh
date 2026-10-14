#!/bin/sh
# GPU re-runs after the log-likelihood clamp fix: the A2D(Image) pilot (same seeds / iterations / float32 as the full
# scale run), and the shipped Q launcher (launch_q.sh -> RunA2DExperiments = main args, default lambda_a2d 0.9) on TD3,
# 10 seeds. The original ran under a 900 s GPU cap: the Q run was cut at iteration 210 and the no-Q twin was skipped.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
L=runs/recheck/logs; mkdir -p $L
run() { exp=$1; shift; echo "[INFO] $(date +%T) start $exp $*"; t0=$(date +%s)
  $PY -m src.train --exp $exp "$@" > $L/gpu_${exp}_$(echo "$*" | tr ' /=' '___' | cut -c1-120).log 2>&1
  echo "[INFO] $(date +%T) end rc=$? wall=$(( $(date +%s) - t0 ))s $exp $*"; }
run recheck_img --env FL --arm a2d_image --seeds 0-2 --iters 80 --set loglik_clamp=true
run recheck_img --env TD --arm a2d_image --seeds 0-2 --iters 80 --set loglik_clamp=true
run recheck_var --env TD3 --arm a2d_compact+launchq_q1 --seeds 0-9 --iters 250 --set use_q=true lam_a2d=0.9 dtype=float64 loglik_clamp=true
run recheck_var --env TD3 --arm a2d_compact+launchq_q0 --seeds 0-9 --iters 250 --set use_q=false lam_a2d=0.9 dtype=float64 loglik_clamp=true
echo "[INFO] $(date +%T) RECHECK_GPU_DONE"
