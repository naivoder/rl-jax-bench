#!/bin/sh
# Torch reference-run matrix (DI-engine loss + original MinAtar), Breakout, 3 seeds x 5 configurations, run in the
# venv made by scripts/official/setup.sh. Run from the paper folder; up to 6 single-thread workers.
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
export OFFICIAL_ROOT
PY=${PY:-"$OFFICIAL_ROOT/.venv/bin/python"}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p runs/ref
for s in 0 1 2; do
  echo "DQfD_eps0.01 DQfD 0.01 0.0 $s"
  echo "DQfD_eps0.1 DQfD 0.1 0.0 $s"
  echo "PDD PDD 0.01 0.0 $s"
  echo "NoMargin NoMargin 0.01 0.0 $s"
  echo "DQfD_eps0.01_sticky0.1 DQfD 0.01 0.1 $s"
done | xargs -P 6 -L 1 sh -c $PY' -I scripts/official/ref_torch_dqfd.py --demo data/demos/Breakout_subopt.npz --arm $1 --eps $2 --sticky $3 --seed $4 --steps 200000 --pretrain 75000 --out runs/ref/$0_s$4.json > runs/ref/$0_s$4.log 2>&1'
