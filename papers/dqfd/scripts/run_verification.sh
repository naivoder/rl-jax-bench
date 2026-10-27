#!/bin/sh
# Verification runs (added after the main grid), sequential. Run from the paper folder.
# usage: sh scripts/run_verification.sh pddconst | target_period | sticky | maxp | long
#   pddconst       PDD with the paper's literal constant eps = 0.01, every game
#   target_period  DQfD with one update per 4 steps and the target period matched in env steps
#   sticky         Breakout on the sticky-action env (p = 0.1): demonstrator regenerated, then the cell
#   maxp           running-max vs current-max priority for new agent transitions
#   long           Breakout with k = 750 k pre-training updates and 2 M interaction steps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-"uv run python"}
RUN="$PY -m src.train --demonstrator subopt"
mkdir -p runs/full
if [ "$1" = "pddconst" ]; then
  $RUN --game Breakout --seeds 5 --seed0 5 --steps 5e5 --pretrain 75000 --bin-width 5000 --arms PDDconst --out runs/full/breakout_pddconst > runs/full/breakout_pddconst.log 2>&1
  for g in SpaceInvaders Asterix Seaquest; do
    EXTRA=""; [ "$g" = "SpaceInvaders" ] && EXTRA="--set run.final_eval_episodes=15"
    $RUN --game $g --seeds 5 --seed0 0 --steps 3e5 --pretrain 75000 --bin-width 3000 $EXTRA --arms PDDconst --out runs/full/${g}_pddconst > runs/full/${g}_pddconst.log 2>&1
  done
fi
if [ "$1" = "target_period" ]; then
  $RUN --game Breakout --seeds 5 --seed0 0 --steps 2e5 --pretrain 75000 --bin-width 5000 --set optim.target_update_period=2500 \
     --arms "DQfD+upd_every=4" --out runs/full/breakout_flags_tau > runs/full/breakout_flags_tau.log 2>&1
fi
if [ "$1" = "sticky" ]; then
  # regenerates data/demos_sticky (the shipped copy holds the json and the subopt set of the original run)
  $PY scripts/make_demos.py Breakout --sticky 0.1 --demo-dir data/demos_sticky > runs/make_demos_Breakout_sticky.log 2>&1
  $RUN --game Breakout --demo-dir data/demos_sticky --seeds 5 --seed0 0 --steps 5e5 --pretrain 75000 --bin-width 5000 \
     --arms "DQfD,IMIT,PDD,PDDconst,DQfD+eps0=0.1+eps1=0.1,PDD+eps1=0.1" --out runs/full/breakout_sticky > runs/full/breakout_sticky.log 2>&1
  $PY scripts/demonstrator_greedy.py --demo-dir data/demos_sticky --games Breakout --out runs/full/demonstrator_greedy_sticky.json
fi
if [ "$1" = "maxp" ]; then
  for v in ctrl curmax; do
    F=""; [ "$v" = "curmax" ] && F="--curmax"
    $PY scripts/run_maxp_variant.py $F -- --game Breakout --demonstrator subopt --seeds 5 --seed0 0 --steps 2e5 --pretrain 75000 --bin-width 5000 \
       --arms DQfD --out runs/full/breakout_maxp_$v > runs/full/breakout_maxp_$v.log 2>&1
  done
fi
if [ "$1" = "long" ]; then
  $RUN --game Breakout --seeds 5 --seed0 0 --steps 2e6 --pretrain 750000 --bin-width 20000 --set run.episode_log_cap=150000 \
     --arms "DQfD,NoMargin,IMIT" --out runs/full/breakout_long > runs/full/breakout_long.log 2>&1
fi
