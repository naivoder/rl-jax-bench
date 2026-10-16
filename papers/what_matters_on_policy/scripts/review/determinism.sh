#!/bin/sh
# Rerun determinism on CPU: the same tiny hopper configuration twice, plus a different batch composition; the lanes
# of det_a and det_b must be bitwise equal, and REC must be equal across all three. Outputs in runs/determinism/.
set -e
PY=${PY:-"uv run python"}
OUT=${OUT:-runs/determinism}; mkdir -p "$OUT"
export JAX_PLATFORMS=cpu
COMMON="--env hopper --seeds 0 --total_steps 8192 --num_envs 32 --iteration_size 256 --eval_every 4096 --eval_episodes 4 --det_final 0 --no_ckpt --tag determinism"
$PY -m src.train $COMMON --arms REC,A7_valuenorm_none --out "$OUT/det_a"
$PY -m src.train $COMMON --arms REC,A7_valuenorm_none --out "$OUT/det_b"
$PY -m src.train $COMMON --arms REC,A2b_initstd_1,A3a_shared_bc1 --out "$OUT/det_c"
