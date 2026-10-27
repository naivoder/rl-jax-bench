#!/bin/sh
# Trains the 4 PPO demonstrators and writes the demo sets to runs/demos (about 3 min per game on a GPU).
# Compare the sha256 values in runs/demos/<game>_demos.json with the shipped data/demos/<game>_demos.json.
# SpaceInvaders was regenerated with --max-steps 10000 (see README); the shipped Asterix set used the default cap.
PY=${PY:-"uv run python"}
mkdir -p runs
for g in Breakout Asterix Seaquest; do
  $PY scripts/make_demos.py $g --demo-dir runs/demos > runs/make_demos_$g.log 2>&1
done
$PY scripts/make_demos.py SpaceInvaders --max-steps 10000 --demo-dir runs/demos > runs/make_demos_SpaceInvaders.log 2>&1
