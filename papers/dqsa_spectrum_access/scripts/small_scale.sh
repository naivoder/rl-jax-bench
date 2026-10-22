#!/bin/sh
# Small-scale stage: DQSA on the small cases (selection seeds 100-104, R = 10 000 iterations, CPU), the diagnostics, the
# 10-seed lr groups and the training-regime checks against the third-party reference. Runs sequentially.
# Arguments of the first runs (runs/small, runs/diag, runs/ext20k, runs/official_ours) were not recorded in a launch file;
# they are reconstructed here from the run-folder names and the notes, so the ramp / continuation flags are a best reading.
set -e
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
PY=${PY:-"uv run python"}
mkdir -p runs/logs
T="$PY src/train.py"
$T --exp E0 --seeds 100 101 102 --lr 3e-4 --out runs/small > runs/logs/small_E0.log 2>&1
$T --exp E6 --seeds 100 101 102 103 104 --lr 3e-4 --out runs/small > runs/logs/small_E6.log 2>&1
$T --exp E7 --seeds 100 101 102 103 104 --lr 3e-4 --out runs/small > runs/logs/small_E7.log 2>&1
$T --exp E7 --seeds 100 101 102 103 104 --lr 1e-3 --out runs/small > runs/logs/small_E7_lr1e-3.log 2>&1
$T --exp E8 --seeds 100 101 102 --lr 3e-4 --out runs/small > runs/logs/small_E8.log 2>&1
$T --exp E1 --seeds 100 --lr 3e-4 --out runs/small > runs/logs/small_E1.log 2>&1
# diagnostics: shorter alpha/beta ramp, E1 with the sum-rate reward, E7 continued to 20k at the final alpha = 0, beta = 20
for e in E0 E6 E7; do $T --exp $e --seeds 100 101 102 --lr 3e-4 --sched-iters 3000 --out runs/diag --tag ${e}_ramp3000 > runs/logs/diag_${e}.log 2>&1; done
$T --exp E1 --reward sum_rate --seeds 100 --lr 3e-4 --out runs/diag > runs/logs/diag_E1_sum_rate.log 2>&1
$T --exp E7 --seeds 100 101 102 103 104 --lr 3e-4 --iters 20000 --sched-iters 10000 --out runs/ext20k > runs/logs/ext20k_E7.log 2>&1
# lr decision: ONE vmapped group of 10 selection seeds per lr
for lr in 1e-4 3e-4 1e-3; do $T --exp E7 --seeds 100 101 102 103 104 105 106 107 108 109 --lr $lr --out runs/small_g10 > runs/logs/g10_E7_lr$lr.log 2>&1; done
# training-regime checks next to the third-party reference: 40k iterations, and the third-party acting rule
$T --exp E0 --seeds 100 101 102 --lr 3e-4 --iters 40000 --sched-iters 10000 --out runs/official_ours --tag E0_40k > runs/logs/E0_40k.log 2>&1
$T --exp E0 --seeds 100 101 102 --lr 3e-4 --act-rule argmax_eps --out runs/official_ours > runs/logs/E0_argmaxeps.log 2>&1
$PY scripts/small_report.py runs/small runs/small_baselines
