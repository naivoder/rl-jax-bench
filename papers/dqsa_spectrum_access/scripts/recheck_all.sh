#!/bin/sh
# Every recheck arm (A3 was stopped at about 500 iterations in the original run and is left out), the T_s sweeps and the reports.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
mkdir -p runs/logs
for arm in A1 A2 B1 C1 BASE C2 A1F A2F E0F A1G E0G; do sh scripts/recheck_rerun.sh $arm > runs/logs/recheck_$arm.log 2>&1; done
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} OMP_NUM_THREADS=1
$PY scripts/ts_sweep.py E2proxy --n-exp 200 --n-tune 100
$PY scripts/ts_sweep.py E3proxy --n-exp 200 --n-tune 100
$PY scripts/ts_sweep.py E2 --n-exp 50 --n-tune 12
$PY scripts/ts_sweep.py E3 --n-exp 50 --n-tune 12
$PY scripts/ts_report.py E2proxy E3proxy E2 E3
$PY scripts/recheck_report.py
$PY scripts/e0_pooled.py
$PY scripts/seed_distribution.py runs/small_g10/E7_competitive_lr0.0003
