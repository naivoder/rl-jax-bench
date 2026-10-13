#!/bin/sh
# ORACLE reference curve with the official main_a2c.py ARGPARSE DEFAULTS (asym-rlpo@1a065b0): only env, algo, seed and
# the 2M step budget are passed. Defaults differ from the paper (App. E / Table 2): lr 1e-4 both, negentropy 1.0 -> 0.01,
# episode cap 1000, 1 episode per batch. Usage: run_official_defaults.sh <env-id> <algo> <seed> <steps> <out>
ENV=$1; ALGO=$2; SEED=$3; STEPS=$4; OUT=$5
HERE=$(cd "$(dirname "$0")" && pwd)
OFF=${OFFICIAL_ROOT:-$HERE/../../external/official}
mkdir -p $OFF/runs
cd $OFF/runs && mkdir -p $OUT && cd $OUT
export WANDB_MODE=offline WANDB_DIR=$OFF/runs/$OUT WANDB_SILENT=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=$OFF/asym-rlpo-aamas2022:$HERE exec $OFF/.venv/bin/python -W ignore \
  $HERE/official_main_shim.py $ENV $ALGO --seed $SEED \
  --max-simulation-timesteps $STEPS --wandb-offline
