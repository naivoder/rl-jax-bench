#!/bin/sh
# Every learner run behind results/, in order, then its grader. Sequential; GPU recommended (wall times on one
# RTX 3090 in brackets). Each train.py call resumes from its own orbax checkpoint if re-run.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
# full-scale: the paper protocol at the pre-declared 100-CPI training fallback, 10 seeds, E1-E4 x {eq3, lin} [26 min]
$PY src/train.py --config src/configs/full.yaml --out runs/full
$PY scripts/full_report.py runs/full
# variants T-A (sliding-window DRQN) and T-B (hard_sigmoid gates), E1 only [10 min]
$PY src/train.py --config src/configs/full.yaml --out runs/full_variants --arms ddrqn_win ddrqn_hs --experiments mc_p04 --skip-baselines
$PY scripts/variants_report.py runs/full_variants runs/full
# T-C: DQN and DDRQN at the paper's 500 training CPIs, E1 [17 min]
$PY src/train.py --config src/configs/full_500cpi.yaml --out runs/full_500cpi --skip-baselines
$PY scripts/long_train_report.py runs/full_500cpi runs/full
# review runs: baselines and policy iteration at 500 CPIs, no act-time reset, lr sweep on a fresh pool, E2/E3 at 500 CPIs
$PY src/train.py --config src/configs/review_pi500.yaml --out runs/review_pi500 --baselines-only
$PY src/train.py --config src/configs/full.yaml --out runs/review_nr --arms ddrqn_noreset --experiments mc_p04 --skip-baselines
$PY src/train.py --config src/configs/review_lr.yaml --out runs/review_lr
$PY src/train.py --config src/configs/review_e23.yaml --experiments mc_p02 --out runs/review_e2 --skip-baselines
$PY src/train.py --config src/configs/review_e23.yaml --experiments mc_sched --out runs/review_e3 --skip-baselines
$PY scripts/review_report.py
# recheck: plain SGD, Keras loss scale and per-trace cadence arms on a fresh pool [14 min]
$PY src/train.py --config src/configs/recheck.yaml --out runs/recheck
$PY scripts/recheck_report.py
$PY scripts/regrade_figs.py
# compute accounting (CPU)
JAX_PLATFORMS=cpu $PY scripts/flops.py
