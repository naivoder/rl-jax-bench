#!/bin/sh
# One arm of the recheck stage. Usage: sh scripts/recheck_rerun.sh <arm>
# Every arm: R = 10000, paper ramp over 10000, same selection seeds / group as the baseline it is paired with.
set -e
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
PY=${PY:-"uv run python"}
OUT=runs/recheck/rerun
P="$PY src/train.py"
case "$1" in
  A1) $P --exp E0 --seeds 100 101 102 --epochs 5 --out $OUT --tag E0_A1_ep5_carried ;;
  A2) $P --exp E0 --seeds 100 101 102 --epochs 5 --fresh-opt --out $OUT --tag E0_A2_ep5_fresh ;;
  A3) $P --exp E0 --seeds 100 101 102 --epochs 30 --fresh-opt --l2 1e-4 --lr 1e-3 --out $OUT --tag E0_A3_matlab30 ;;
  B1) $P --exp E6 --seeds 100 101 102 103 104 --reward-scale-by-n --out $OUT --tag E6_B1_scale_1overN ;;
  C1) $P --exp E6 --reward sum_rate_episodic --seeds 100 101 102 103 104 --out $OUT --tag E6_C1_episodic ;;
  BASE) $P --exp E6 --seeds 100 101 102 103 104 --out $OUT --tag E6_BASE_shaped ;;
  C2) $P --exp E6 --reward sum_rate_episodic --reward-scale 0.02 --seeds 100 101 102 103 104 --out $OUT --tag E6_C2_episodic_scale0.02 ;;
  A1F) $P --exp E0 --seeds 103 104 105 --epochs 5 --out $OUT --tag E0_A1F_ep5_carried_fresh_seeds ;;
  A2F) $P --exp E0 --seeds 103 104 105 --epochs 5 --fresh-opt --out $OUT --tag E0_A2F_ep5_fresh_fresh_seeds ;;
  E0F) $P --exp E0 --seeds 103 104 105 --out $OUT --tag E0_E0F_single_step_fresh_seeds ;;
  A1G) $P --exp E0 --seeds 106 107 108 --epochs 5 --out $OUT --tag E0_A1G_ep5_carried_fresh_seeds2 ;;
  E0G) $P --exp E0 --seeds 106 107 108 --out $OUT --tag E0_E0G_single_step_fresh_seeds2 ;;
  *) echo "unknown arm $1"; exit 2 ;;
esac
