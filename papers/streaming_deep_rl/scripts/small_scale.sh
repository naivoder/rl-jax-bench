#!/bin/sh
# Small-scale GPU launches, sequential in one shell. Logs: runs/small/logs/
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/small/logs
PY=${PY:-"uv run python"}
$PY -m src.train --config src/configs/e3_minatar.yaml --tag small --set total_steps=500000 \
  --set "arms=[stream_q,classic_q,dqn1]" --set "seeds=[1,2,3,4,5]" --set ckpt_every_chunks=2 > runs/small/logs/e3_breakout.log 2>&1
$PY -m src.train --config src/configs/e6_hopper.yaml --tag small --set total_steps=200000 --set brax_backend=positional \
  --set "arms=[stream_ac,no_obgd@lr=1e-4,no_obgd@lr=3e-5,no_obgd@lr=1e-5,no_obgd@lr=1e-6,no_norm,no_ln,no_sparse,classic_ac,stream_ac0,classic_ac0]" \
  --set "seeds=[1,2,3]" --set ckpt_every_chunks=1 > runs/small/logs/e6_hopper.log 2>&1
# E8 at small scale on CPU. The exact --set overrides of the original small-scale E8 command were not recorded;
# this runs the config as shipped.
JAX_PLATFORMS=cpu $PY -m src.train --config src/configs/e8_ett.yaml --tag small > runs/small/logs/e8_ett.log 2>&1
# MinAtar parity needs an official-side recording made with scripts/official/minatar_rollouts.py (official venv).
$PY scripts/minatar_parity.py runs/parity_ref.npz 2000 5000
$PY scripts/small_report.py small
