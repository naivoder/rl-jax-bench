#!/bin/sh
# float64 probes (scripts/x64_probe.py): does float32 training cause the TD seeds stuck at a door? Sequential, CPU.
set -u
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
PY=${PY:-"uv run python"}
L=runs/full/logs_cpu; mkdir -p $L
run() { echo "[INFO] $(date +%T) start x64 $*"; t0=$(date +%s)
  $PY scripts/x64_probe.py "$@" > $L/x64_$1_$2.log 2>&1
  echo "[INFO] $(date +%T) end rc=$? wall=$(( $(date +%s) - t0 ))s x64 $*"; }
run TD rl_mdp
run TD2 a2d_compact+lam0.5 10 250 lam_a2d=0.5 lam_rl=0.5
run TD a2d_compact 20 200
echo "[INFO] X64_DONE"
