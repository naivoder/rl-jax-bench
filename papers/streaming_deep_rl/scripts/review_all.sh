#!/bin/sh
# Review pass and recheck runs, all on CPU, sequential. Needs `sh scripts/official/setup.sh` first (the real-Hopper
# runs step Hopper-v4 inside the authors' venv through scripts/official/hopper_env_server.py).
cd "$(dirname "$0")/.." || exit 1
export JAX_PLATFORMS=cpu
PY=${PY:-"uv run python"}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$PWD/external/official}; export OFFICIAL_ROOT
mkdir -p runs/review/logs runs/recheck
sh scripts/review_e8.sh
$PY -m src.train --config src/configs/review_e3_cpu.yaml --tag review > runs/review/logs/e3_cpu.log 2>&1   # ~5 h
$PY scripts/ours_on_mujoco_vm.py 350000 6000 1 2 3 4 5 > runs/review/logs/mujoco_vm.log 2>&1
sh scripts/official/run_official_ac.sh Hopper-v4 350000 "$OFFICIAL_ROOT/runs/ac_hopper_350k_s45" 4 5
$PY scripts/review_report.py
# Recheck: the no-ObGD arm at App. F.4's Adam step size, and the frozen-policy probe. The argument strings below were
# reconstructed from the run logs and the scripts' usage lines; the exact original argv was not logged.
$PY scripts/real_hopper_lanes.py runs/recheck/no_obgd_real_hopper 200000 9000 'no_obgd@lr=1e-7:1,2,3,4,5,11,12,13|no_obgd@lr=3e-4:1,2,3'
$PY scripts/no_obgd_report.py
mkdir -p runs/recheck/frozen_policy
( cd runs/recheck/frozen_policy && "$OFFICIAL_ROOT/.venv/bin/python" -I ../../../scripts/official/official_frozen.py 25000 1 2 3 4 5 6 7 8 )
$PY scripts/real_hopper_lanes.py runs/recheck/frozen_policy/ours_lr0 100000 1500 'no_obgd@lr=0:1,2,3,4,5,6,7,8|no_obgd@lr=1e-7@lam=0:1,2,3'
$PY scripts/frozen_policy_report.py
