#!/bin/sh
# Faithful-env comparison on gymnasium HalfCheetah-v4 (1M steps, CPU): the authors' modules vs our learner, same
# simulator. Official runs need $OFFICIAL_ROOT and its TF venv; ours run in the repository environment (7-36 min each).
# The original seeds 0-4 of "ours" ran an earlier copy of scripts/train_gym_v4.py with the same learner.
set -e
PAPER=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-"uv run python"}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-"$PAPER/external/official"}
D="$PAPER/runs/recheck/official-oracle"; mkdir -p "$D/official" "$D/ours" "$D/ours_fix"
export OFFICIAL_ROOT
official() {  # <cfg> <seed>
  (cd "$OFFICIAL_ROOT" && CUDA_VISIBLE_DEVICES= TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 \
    .venv/bin/python "$PAPER/scripts/official/official_loop_v4.py" HalfCheetah-v4 $1 $2 "$D/official/halfcheetah_$1_s$2.json")
}
ours() {      # <arm> <seed> <subdir> [extra args]
  arm=$1; seed=$2; sub=$3; shift 3
  (cd "$PAPER" && JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 $PY scripts/train_gym_v4.py $arm $seed "$D/$sub/${arm}_s${seed}.json" "$@")
}
# 2M lr horizon (the schedule in force before the fix) for every row compared with seeds 0-4
for s in 0 1 2 3 4 5 6 7 8 9; do
  official rec_pre $s; official rec_pre_vnone $s
  ours REC $s ours --lr_horizon_frames 2000000; ours A7_valuenorm_none $s ours --lr_horizon_frames 2000000
done
for s in 0 1 2 3 4; do
  official rec_pre_std1 $s; official rec_pre_lr1em4 $s
  ours A2b_initstd_1 $s ours --lr_horizon_frames 2000000; ours A5a_lr_1em4 $s ours --lr_horizon_frames 2000000
done
# post-fix lr horizon (= training length), seeds 0-2
for s in 0 1 2; do
  ours X5_rec_lr_to0_1M $s ours; ours X6_lr1em4_to0_1M $s ours
  (cd "$PAPER" && JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 $PY scripts/train_gym_v4.py REC $s "$D/ours/REC_overlap_s$s.json" --overlap)
  ours A2b_initstd_1 $s ours_fix; ours A7_valuenorm_none $s ours_fix
done
cd "$PAPER" && $PY scripts/recheck_report.py
