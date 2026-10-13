#!/bin/sh
# ORACLE reference curve: the official main_a2c.py (asym-rlpo@1a065b0, CPU, wandb offline) with the paper's App. E
# settings and Table 2 hyper-parameters. Usage: run_official.sh <env-id> <algo> <lr_pi> <lr_v> <lam0> <seed> <steps> <out> [extra]
ENV=$1; ALGO=$2; LRA=$3; LRC=$4; LAM=$5; SEED=$6; STEPS=$7; OUT=$8; shift 8
HERE=$(cd "$(dirname "$0")" && pwd)
OFF=${OFFICIAL_ROOT:-$HERE/../../external/official}
mkdir -p $OFF/runs
cd $OFF/runs && mkdir -p $OUT && cd $OUT
export WANDB_MODE=offline WANDB_DIR=$OFF/runs/$OUT WANDB_SILENT=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=$OFF/asym-rlpo-aamas2022:$HERE exec $OFF/.venv/bin/python -W ignore \
  $HERE/official_main_shim.py $ENV $ALGO --seed $SEED \
  --max-simulation-timesteps $STEPS --max-episode-timesteps 100 --simulation-num-episodes 2 \
  --optim-lr-actor $LRA --optim-lr-critic $LRC --negentropy-value-from $LAM --negentropy-value-to $(python3 -c "print($LAM/10)") \
  --wandb-offline "$@"
