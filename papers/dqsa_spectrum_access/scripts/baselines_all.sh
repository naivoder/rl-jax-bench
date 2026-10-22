#!/bin/sh
# Non-learning arms (Aloha, TDMA genie, OCA, DP literal / tuned) on the shared eval worlds -> runs/small_baselines and runs/baselines_v2.
# Sequential replacement of the original per-core queue scripts; paper scale (N = 100, K = 50) DP arms take hours on one CPU core.
set -e
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
PY=${PY:-"uv run python"}
B="$PY scripts/baselines_eval.py"
# small-scale comparison arms (replicate id = DQSA seed)
$PY src/toys.py --only pattern
for e in E0 E7 E6 E8; do $B --exp $e --arms aloha tdma_genie --rep-ids 100 101 102 103 104 --out runs/small_baselines; done
$B --exp E1 --arms aloha tdma_genie --rep-ids 100 101 102 --out runs/small_baselines
$B --exp E2proxy --arms aloha tdma_genie oca dp --rep-ids 100 101 102 --out runs/small_baselines
$B --exp E3proxy --arms aloha oca dp --rep-ids 100 101 102 --out runs/small_baselines
$B --exp E2 --arms aloha tdma_genie oca dp --rep-ids 100 101 --n-exp 200 --out runs/small_baselines
$B --exp E3 --arms aloha oca dp --rep-ids 100 101 --n-exp 200 --out runs/small_baselines
# baselines after the review (default --out runs/baselines_v2); E1K2 Aloha / genie
FIXED="aloha tdma_genie oca oca_sameslot dp_lit_tau0.1 dp_lit_tau0.5 dp_lit_tau1.0 dpv1_lit_tau0.1 dpv1_lit_tau0.5 dpv1_lit_tau1.0"
$B --exp E1K2 --arms aloha tdma_genie --reps 2
for e in E2proxy E3proxy; do $B --exp $e --arms $FIXED --reps 2 --n-exp 200; done
$PY scripts/tune_dp.py E2proxy --n-exp 200 --reps 1 --rule gibbs
$PY scripts/tune_dp.py E2proxy --n-exp 200 --reps 1 --rule count
for e in E2proxy E3proxy; do $B --exp $e --arms dp_tuned dpv1_tuned --reps 2; done
# paper scale, reduced for CPU cost: one replicate for the DP arms, 40 / 12 tuning worlds (count / gibbs)
$B --exp E2 --arms aloha tdma_genie oca oca_sameslot --reps 2
$B --exp E3 --arms aloha tdma_genie oca oca_sameslot --reps 2
for e in E2 E3; do $B --exp $e --arms dpv1_lit_tau0.1 dpv1_lit_tau0.5 dpv1_lit_tau1.0 dp_lit_tau0.1 dp_lit_tau0.5 dp_lit_tau1.0 --reps 1; done
$PY scripts/tune_dp.py E2 --n-exp 40 --reps 1 --rule count
$PY scripts/tune_dp.py E2 --n-exp 12 --reps 1 --rule gibbs
for e in E2 E3; do $B --exp $e --arms dpv1_tuned dp_tuned --reps 1; done
$PY scripts/review_tables.py
