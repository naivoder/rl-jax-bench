#!/bin/sh
# Every training run and report behind results/, sequentially. Run from anywhere; outputs go to runs/.
# Original cost: preliminary PD grid 4 min (147 runs vmapped on an RTX 3090), full-scale MiniGrid 10 min GPU,
# full-scale 2D-LH 3 min GPU, full-scale PD (draws + alpha sweep, 96 + 25 runs) about 70 min on 6 CPU cores.
# The authors'-TSV comparisons need `sh scripts/official/setup.sh` first (it fetches their hp_runs/ TSVs).
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
P=src/configs/plans

JAX_PLATFORMS=cpu $PY scripts/toys.py                                      # exact toys -> runs/toys/toys.json

# Preliminary grids: PD seed protocol (lr tuned per arm on seeds 100-101), 2D-LH 5 seeds, MiniGrid 3e5-step proxy
$PY scripts/run_grid.py --env pd --tag prelim_pd --arms ppo bc dagger bc_tf1 bc_ppo bc_then_ppo advisor \
    --seeds 0 1 2 3 4 100 101 --lrs 3e-4 1e-3 3e-3
$PY scripts/report_grid.py prelim_pd
$PY scripts/run_grid.py --env lh --tag prelim_lh --arms bc ppo bc_then_ppo advisor --alphas 20 --seeds 0 1 2 3 4 \
    --lrs 0.0242 --cells 1,1 1,15 5,15 --n_ckpt 1
$PY scripts/report_grid.py prelim_lh len lower
$PY scripts/lh_compare.py prelim_lh
$PY scripts/run_grid.py --env mg --tag prelim_mg --arms ppo bc bc_tf1 bc_ppo bc_then_ppo advisor --seeds 0 1 2 \
    --lrs 1e-3 --total 300000 --n_ckpt 3 --n_eval 500
$PY scripts/report_grid.py prelim_mg

# Full scale
$PY scripts/run_grid.py --env mg --tag full_mg --plan $P/full_mg.json --n_ckpt 4 --n_eval 1000 --n_eval_mid 200 --resume
$PY scripts/run_grid.py --env lh --tag full_lh_og --plan $P/full_lh.json --n_ckpt 1 --n_eval 200 --resume
$PY scripts/run_grid.py --env lh --tag full_lh_dagger_og --plan $P/full_lh_dagger.json --n_ckpt 1 --n_eval 200
$PY scripts/run_grid.py --env lh --tag full_lh_ppo_rep --plan $P/full_lh_ppo_rep.json --n_ckpt 1 --n_eval 200
$PY scripts/run_grid.py --env lh --tag full_lh_ppo_alone --plan $P/full_lh_ppo_s0_24_alone.json --n_ckpt 1 --n_eval 200
JAX_PLATFORMS=cpu $PY scripts/run_grid.py --env pd --tag full_pd --plan $P/full_pd.json \
    --n_ckpt 4 --n_eval 1000 --n_eval_mid 200 --resume
JAX_PLATFORMS=cpu $PY scripts/run_grid.py --env pd --tag full_pd_dadv --plan $P/full_pd_dagger_adv.json \
    --n_ckpt 4 --n_eval 1000 --n_eval_mid 200

# Reports -> runs/report/ and runs/<tag>/rr_report.json
$PY scripts/report.py mg
$PY scripts/report.py lh
$PY scripts/report.py pd
$PY scripts/report.py pd_seeds
$PY scripts/rr_report.py full_pd PoisonedDoors
$PY scripts/rr_report.py full_pd_dadv PoisonedDoors
$PY scripts/pd_door_noise.py full_pd
$PY scripts/table_a1.py
$PY scripts/tsv_reference.py
$PY scripts/mg_tiebreak_effect.py 200
JAX_PLATFORMS=cpu $PY scripts/flops.py
$PY scripts/aggregate_seeds.py
