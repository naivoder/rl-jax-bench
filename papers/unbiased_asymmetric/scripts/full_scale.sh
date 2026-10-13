#!/bin/sh
# Full-scale stage: paper length (Table 2 hyper-parameters), 5 runs per cell, then the MC ablation (A1) and the
# reduction -> runs/full/summary.json + runs/full/tables.md. One process at a time.
# The stored runs did NOT run to completion: the Heaven-Hell cells (GPU, TF32 default) were stopped at the GPU budget
# after 3.3-6.1M steps and finalized from their last checkpoint (scripts/finalize_from_ckpt.py); Shopping-6 (CPU) was
# stopped after a checkpoint as well; Car-Flag and Cleaner were never run. Run lengths below are the paper's.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
mkdir -p runs/full/logs runs/ablation/logs
for cfg in full_heavenhell4 full_heavenhell3; do
  for m in a2c_asym_hs a2c_asym_s a2c; do
    $PY src/train.py --config src/configs/$cfg.yaml --method $m > runs/full/logs/${cfg}_$m.log 2>&1
  done
done
for m in a2c_asym_hs a2c_asym_s a2c a2c_react_2 a2c_react_4; do
  $PY src/train.py --config src/configs/full_shopping6.yaml --method $m > runs/full/logs/full_shopping6_$m.log 2>&1
done
$PY src/train.py --config src/configs/ablation_shopping5_mc.yaml --method a2c_asym_s > runs/ablation/logs/a2c_asym_s.log 2>&1
# not run for the stored results (budget): full_carflag.yaml and full_cleaner.yaml, arms a2c_asym_hs a2c_asym_s a2c
$PY scripts/official_reference.py   # needs the logs of scripts/official/reference_runs.sh
$PY src/eval.py --tag ablation
$PY src/eval.py --tag full
