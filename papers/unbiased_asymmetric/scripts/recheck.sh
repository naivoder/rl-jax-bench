#!/bin/sh
# Recheck stage, all CPU with full-f32 matmuls: fresh-seed and same-key re-runs of the two contested Heaven-Hell cells,
# a final-entropy-weight sensitivity run, plus authors'-code runs used as the comparison
# (scripts/official/run_official.sh, needs $OFFICIAL_ROOT), then the reduction -> runs/recheck/summary.json.
# Fresh-seed cells train 20 runs as 4 slices of 5 (--run-offset); only 10 of the 20 finished for the stored results.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
OFF=scripts/official/run_official.sh
R=${OFFICIAL_ROOT:-external/official}/runs
mkdir -p runs/recheck/logs "$R"
for o in 0 5 10 15; do
  $PY src/train.py --config src/configs/recheck_hh4_hs_fp32_s2.yaml --method a2c_asym_hs --run-offset $o > runs/recheck/logs/hh4_hs_s2_o$o.log 2>&1
  $PY src/train.py --config src/configs/recheck_hh3_s_fp32_s1.yaml --method a2c_asym_s --run-offset $o > runs/recheck/logs/hh3_s_s1_o$o.log 2>&1
done
$PY src/train.py --config src/configs/recheck_hh4_hs_fp32_s0.yaml --method a2c_asym_hs > runs/recheck/logs/hh4_hs_s0.log 2>&1
$PY src/train.py --config src/configs/recheck_hh3_s_fp32_s0.yaml --method a2c_asym_s > runs/recheck/logs/hh3_s_s0.log 2>&1
$PY src/train.py --config src/configs/recheck_hh3_s_lamfin001_s0.yaml --method a2c_asym_s > runs/recheck/logs/hh3_s_lamfin001_s0.log 2>&1
# learner-strength probe (originally on GPU with TF32 matmuls, stopped at 3.4M)
$PY src/train.py --config src/configs/recheck_learner_strength_hh4_td0_s1.yaml --method a2c_asym_hs --max-env-steps 3400000 \
  > runs/recheck/logs/learner_strength_hh4_td0_s1.log 2>&1
# authors' code: HH-4 asym-hs seeds 0-4 (stopped at 3.65-4.06M in the stored results), Shopping-5 react-4 seeds 0-2,
# HH-3 asym-s seeds 0-4 to 4.5M and 5-19 to 2M
for s in 0 1 2 3 4; do
  sh $OFF POMDP-heavenhell_4-episodic-v0 asym-a2c 0.001 0.001 0.1 $s 10000000 recheck_heavenhell4_asym-a2c_s$s > $R/recheck_heavenhell4_asym-a2c_s$s.log 2>&1
  sh $OFF POMDP-heavenhell_3-episodic-v0 asym-a2c-state 0.001 0.001 1.0 $s 4500000 long_heavenhell3_asym-a2c-state_s$s > $R/long_heavenhell3_asym-a2c-state_s$s.log 2>&1
done
for s in 0 1 2; do
  sh $OFF POMDP-shopping_5-episodic-v1 a2c 0.001 0.001 3.0 $s 2000000 recheck_shopping5_react4_s$s \
    --truncated-histories --truncated-histories-n 4 > $R/recheck_shopping5_react4_s$s.log 2>&1
done
for s in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
  sh $OFF POMDP-heavenhell_3-episodic-v0 asym-a2c-state 0.001 0.001 1.0 $s 2000000 stuck_heavenhell3_asym-a2c-state_s$s \
    > $R/stuck_heavenhell3_asym-a2c-state_s$s.log 2>&1
done
$PY src/eval.py --tag recheck
