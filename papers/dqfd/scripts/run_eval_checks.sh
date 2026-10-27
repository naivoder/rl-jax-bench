#!/bin/sh
# Checkpoint evaluations: SpaceInvaders learners under eps 0.01 and eps 0 (unbiased late-run returns), then the
# demonstrators' own greedy returns. Run from the paper folder after run_main.sh games.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-"uv run python"}
cd scripts
BIN_W=3000 $PY eval_ckpt.py ../runs/full/SpaceInvaders SpaceInvaders subopt "DQfD,NoMargin,ADET,IMIT,PDD,DQfD+eps0=0.1+eps1=0.1,PDD+eps1=0.1" 5 3e5 0.01 0.0 > ../runs/full/eval_ckpt_SpaceInvaders.log 2>&1
$PY demonstrator_greedy.py > ../runs/full/demonstrator_greedy.log 2>&1
