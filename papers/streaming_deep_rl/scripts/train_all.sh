#!/bin/sh
# Full-scale GPU launches, sequential. Each run is killed at its wall-clock cap (orbax checkpoints
# make a killed run resumable; a kill is reported). Logs: runs/full/logs/
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/full/logs
PY=${PY:-"uv run python"}
run_capped() {  # cap_seconds log args...
  cap=$1; log=$2; shift 2
  date +%s > "$log.start"
  $PY -m src.train "$@" > "$log" 2>&1 &
  pid=$!
  ( sleep "$cap"; kill "$pid" 2>/dev/null && echo "[WARN] killed at wall-clock cap ${cap}s" >> "$log" ) &
  wd=$!
  wait "$pid"; echo "exit=$?" >> "$log"
  kill "$wd" 2>/dev/null
  date +%s > "$log.end"
}
run_capped 1080 runs/full/logs/e6_hopper.log --config src/configs/full_e6_hopper.yaml --tag full
run_capped 1440 runs/full/logs/e3_breakout.log --config src/configs/full_e3_breakout.yaml --tag full
# CPU parts of the full-scale run: E8 (30 seeds) and OUR stream AC / no-norm / no-ObGD arms on real Hopper-v4
# (gymnasium 1.1.1 + mujoco 3.15 from the root environment; 5e5 steps, 2700 s cap, seeds 1-3).
JAX_PLATFORMS=cpu $PY -m src.train --config src/configs/e8_ett.yaml --tag full > runs/full/logs/e8.log 2>&1
for arm in stream_ac no_norm "no_obgd@lr=1e-5"; do
  ARM=$arm JAX_PLATFORMS=cpu $PY scripts/ours_on_mujoco.py 500000 2700 1 2 3 > "runs/full/logs/mujoco_$arm.log" 2>&1
done
# The PPO comparison on the Brax proxy (struck) used the code in papers/ppo_implementation_details with
# total_timesteps=500000 and seeds 1-5; it is not repeated here.
$PY scripts/digitize_figs.py src/configs/paper_digitized.json     # needs the paper PDF and `uv sync --extra pdf`
$PY scripts/report.py full
$PY scripts/collect_seed_metrics.py full
