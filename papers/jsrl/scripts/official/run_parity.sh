#!/bin/sh
# Env parity under the GUIDE policy against the original numpy MinAtar (needs `sh scripts/official/setup.sh`): CPU, single thread.
# Asterix runs on the spawn-repaired env, with the unrepaired env as a control that must diverge.
PY=${PY:-"uv run python"}
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
mkdir -p runs/parity
$PY scripts/official/policy_parity.py Asterix subopt 135000 1 --repair > runs/parity/asterix_repaired.json 2> runs/parity/asterix_repaired.err
$PY scripts/official/policy_parity.py Asterix subopt 30000 1 --stop > runs/parity/asterix_unrepaired_control.json 2> runs/parity/asterix_unrepaired_control.err
for g in Breakout Seaquest SpaceInvaders; do
  $PY scripts/official/policy_parity.py $g subopt 30000 1 > runs/parity/${g}_guide.json 2> runs/parity/${g}_guide.err
done
