#!/bin/sh
# Every training run behind the README tables, sequentially, from the paper folder. Outputs land in runs/<tag>/.
# Original runs: one RTX 3090 (shared). Times are the measured wall clock of the original runs.
set -e
PY=${PY:-"uv run python"}

# Small-scale run: REC + every primary/secondary/diagnostic arm, 3 envs x seeds 1000-1002 (~30 min for all three).
for e in halfcheetah walker2d hopper; do
  $PY -m src.train --env $e --arms small --seeds 1000-1002 --tag small --eval_episodes 20
done
$PY scripts/small_report.py

# REC re-run under the study-era action semantics (hopper; halfcheetah/walker2d were stopped at the budget line).
$PY -m src.train --env hopper --arms REC --seeds 1000-1002 --tag rec_rerun --eval_episodes 20 --det_final 0 --no_ckpt

# Full-scale run: halfcheetah, REC + 9 primary arms, seeds 0-4 (~15 min incl. compile and evaluation).
$PY -m src.train --env halfcheetah --arms REC,A1_obsnorm_none,A2a_lastlayer_1,A2b_initstd_1,A3a_shared_bc1,A4_lambda_0p95,A5a_lr_1em4,A6_entropy_1em3,A7_valuenorm_none,A8_advnorm_true \
  --seeds 0-4 --tag full --eval_episodes 20 --det_final 0 --lr_horizon_frames 2000000
$PY -m src.eval --tag full
$PY scripts/write_run_config.py full
$PY scripts/report.py

# Recheck runs on halfcheetah: the shared-trunk control and the lr-to-0 arms (~17 min), and the post-fix A2b/A4/A7
# re-run (it was killed at its time limit in the original run; no result).
$PY -m src.train --env halfcheetah --arms review --seeds 0-4 --tag recheck_p1 --eval_episodes 20 --det_final 0 --no_ckpt \
  --out runs/recheck/learner-strength/p1 --lr_horizon_frames 2000000
$PY -m src.train --env halfcheetah --arms A2b_initstd_1,A4_lambda_0p95,A7_valuenorm_none --seeds 0-4 --tag recheck_lrfix \
  --eval_episodes 20 --det_final 0 --no_ckpt

# Second-best reference: Brax's own PPO on the same proxy envs.
for e in hopper walker2d; do $PY scripts/brax_ppo_reference.py $e 0; done
