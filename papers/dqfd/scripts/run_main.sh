#!/bin/sh
# Full-scale runs behind results/analysis.json, sequential. Run from the paper folder.
# usage: sh scripts/run_main.sh breakout | games [NSEEDS] | flags | pre750k
#   breakout  Breakout seeds 5-9 (seeds 0-4 are runs/breakout_first5, see README) + eps 0.1 twins and the lr probe
#   games     SpaceInvaders, Asterix, Seaquest, 5 seeds, 300 k steps
#   flags     attribution flags on Breakout (Huber, update cadence, loss priority, max n-step, lr)
#   pre750k   Asterix with the paper's k = 750 k pre-training updates
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-"uv run python"}
RUN="$PY -m src.train --demonstrator subopt"
E10="+eps0=0.1+eps1=0.1"
mkdir -p runs/full
if [ "$1" = "breakout" ]; then
  $RUN --game Breakout --seeds 5 --seed0 5 --steps 5e5 --pretrain 75000 --bin-width 5000 \
     --arms "DQfD,NoMargin,NoNstep,ADET,IMIT,PDD,DQfD$E10,PDD+eps1=0.1" --out runs/full/breakout_p1 > runs/full/breakout_p1.log 2>&1
  $RUN --game Breakout --seeds 5 --seed0 0 --steps 5e5 --pretrain 75000 --bin-width 5000 \
     --arms "PDD+eps1=0.1,NoMargin$E10,DQfD${E10}+lr=3e-05,DQfD${E10}+lr=0.0003,PDD+lr=3e-05,PDD+lr=0.0003" --out runs/full/breakout_p2 > runs/full/breakout_p2.log 2>&1
fi
if [ "$1" = "games" ]; then
  N=${2:-5}
  for g in SpaceInvaders Asterix Seaquest; do
    EXTRA=""; [ "$g" = "SpaceInvaders" ] && EXTRA="--set run.final_eval_episodes=15"
    $RUN --game $g --seeds $N --seed0 0 --steps 3e5 --pretrain 75000 --bin-width 3000 $EXTRA \
       --arms "DQfD,NoMargin,ADET,IMIT,PDD,DQfD$E10,PDD+eps1=0.1" --out runs/full/$g > runs/full/$g.log 2>&1
  done
fi
if [ "$1" = "flags" ]; then
  $RUN --game Breakout --seeds 5 --seed0 0 --steps 2e5 --pretrain 75000 --bin-width 5000 \
     --arms "DQfD+huber=1,DQfD+upd_every=4,DQfD+prio_total=1,DQfD+nmax=1,DQfD+lr=3e-05,DQfD+lr=0.0003" --out runs/full/breakout_flags > runs/full/breakout_flags.log 2>&1
fi
if [ "$1" = "pre750k" ]; then
  $RUN --game Asterix --seeds 5 --seed0 0 --steps 3e5 --pretrain 750000 --bin-width 3000 \
     --arms "DQfD,IMIT" --out runs/full/asterix_pre750k > runs/full/asterix_pre750k.log 2>&1
fi
