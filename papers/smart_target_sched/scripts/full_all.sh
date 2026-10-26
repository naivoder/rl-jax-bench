#!/bin/sh
# Full-scale run (Nmc = 100): Tables I-IV on two seeds (seed 0 = binding set `full`, seed 1 = independent resample
# `full_s1`), all arms, the ablation arms, the tau / rollout1-base-length sweeps, the variants, and Figs 5-8 on the
# authors' K grids with the Lagrangian bound (`full_figs`). Then the rollout1 base-length grid fill for the 4-D table
# (runs/review/r1_s0, r1_s1). Sequential, one CPU process; every job is resumable (--resume skips finished cells).
# Original cost: about 6 core-hours for the main list plus the grid fill.
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_PLATFORMS=cpu
PY=${PY:-"uv run python"}
mkdir -p runs/logs
run() { name=$1; shift; echo "job $name"; $PY src/train.py "$@" --resume > runs/logs/$name.log 2>&1; }
run fig7_K42 --fig fig7 --Ks 42 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig7_K34 --fig fig7 --Ks 34 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig7_K26 --fig fig7 --Ks 26 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig7_K18 --fig fig7 --Ks 18 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig7_K10 --fig fig7 --Ks 10 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig7_K2 --fig fig7 --Ks 2 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig8_K30 --fig fig8 --Ks 30 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig8_K22 --fig fig8 --Ks 22 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig8_K14 --fig fig8 --Ks 14 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig8_K6 --fig fig8 --Ks 6 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig8_K26 --fig fig8 --Ks 26 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig8_K18 --fig fig8 --Ks 18 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig8_K10 --fig fig8 --Ks 10 --nmc 100 --policies whittle myopic tev --tag full_figs
run fig8_K2 --fig fig8 --Ks 2 --nmc 100 --policies whittle myopic tev --tag full_figs
run t4_s0 --tables t4 --K 1 2 3 --nmc 100 --seed 0 --policies whittle myopic tev passive_cont passive_cont_h10 passive_cont_revisit random rollout1 passive_cont_h2 passive_cont_h3 passive_cont_h5 passive_cont_h20 passive_cont_h50 whittle_blind0 whittle_blind1 myopic_blind0 myopic_blind1 --tag full
run t4_s1 --tables t4 --K 1 2 3 --nmc 100 --seed 1 --policies whittle myopic tev passive_cont passive_cont_h10 passive_cont_revisit random rollout1 passive_cont_h2 passive_cont_h3 passive_cont_h5 passive_cont_h20 passive_cont_h50 whittle_blind0 whittle_blind1 myopic_blind0 myopic_blind1 --tag full_s1
run fig5a --fig fig5a --Ks 2 10 18 26 34 42 50 58 66 74 82 90 98 --nmc 100 --policies whittle myopic tev passive_cont_revisit --lb --tag full_figs
run fig5b --fig fig5b --Ks 2 10 18 26 34 42 50 58 66 74 82 90 98 --nmc 100 --policies whittle myopic tev passive_cont_revisit --lb --tag full_figs
run fig5c --fig fig5c --Ks 2 10 18 26 34 42 50 58 66 74 82 90 98 --nmc 100 --policies whittle myopic tev passive_cont_revisit --lb --tag full_figs
run fig6 --fig fig6 --Ks 2 6 10 14 18 22 26 30 34 38 42 46 --nmc 100 --policies whittle myopic tev passive_cont_revisit --lb --tag full_figs
run t123_s0 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 0 --policies whittle myopic tev passive_cont passive_cont_h10 passive_cont_revisit random rollout1 passive_cont_h2 passive_cont_h3 passive_cont_h5 passive_cont_h20 passive_cont_h50 whittle_blind0 whittle_blind1 myopic_blind0 myopic_blind1 --tag full
run t123_s1 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 1 --policies whittle myopic tev passive_cont passive_cont_h10 passive_cont_revisit random rollout1 passive_cont_h2 passive_cont_h3 passive_cont_h5 passive_cont_h20 passive_cont_h50 whittle_blind0 whittle_blind1 myopic_blind0 myopic_blind1 --tag full_s1
run t123_sampled --tables t1 t2 t3 --K 1 2 3 --nmc 100 --sampled --policies whittle myopic tev passive_cont_revisit --tag full
run t123_skipneg --tables t1 t2 t3 --K 1 2 3 --nmc 100 --skip-neg --policies whittle myopic tev --tag full
run t4_sampled --tables t4 --K 1 2 3 --nmc 100 --sampled --policies whittle myopic tev --tag full
run t4_ftstd --tables t4 --K 1 2 3 --nmc 100 --ft standard --policies whittle myopic tev --tag full
run t4_pcli1lit --tables t4 --K 1 2 3 --nmc 100 --pcli1 literal --policies whittle --tag full
run tau2_t123_s0 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 0 --tau 2 --policies whittle --tag full
run tau2_t4_s0 --tables t4 --K 1 2 3 --nmc 100 --seed 0 --tau 2 --policies whittle --tag full
run tau3_t123_s0 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 0 --tau 3 --policies whittle --tag full
run tau3_t4_s0 --tables t4 --K 1 2 3 --nmc 100 --seed 0 --tau 3 --policies whittle --tag full
run tau5_t123_s0 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 0 --tau 5 --policies whittle --tag full
run tau5_t4_s0 --tables t4 --K 1 2 3 --nmc 100 --seed 0 --tau 5 --policies whittle --tag full
run tau10_t123_s0 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 0 --tau 10 --policies whittle --tag full
run tau10_t4_s0 --tables t4 --K 1 2 3 --nmc 100 --seed 0 --tau 10 --policies whittle --tag full
run tau20_t123_s0 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 0 --tau 20 --policies whittle --tag full
run tau20_t4_s0 --tables t4 --K 1 2 3 --nmc 100 --seed 0 --tau 20 --policies whittle --tag full
run tau50_t123_s0 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 0 --tau 50 --policies whittle --tag full
run tau50_t4_s0 --tables t4 --K 1 2 3 --nmc 100 --seed 0 --tau 50 --policies whittle --tag full
run tau200_t123_s0 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 0 --tau 200 --policies whittle --tag full
run tau200_t4_s0 --tables t4 --K 1 2 3 --nmc 100 --seed 0 --tau 200 --policies whittle --tag full
run r1b5_t123_s0 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 0 --r1-base 5 --policies rollout1 --tag full
run r1b10_t123_s0 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 0 --r1-base 10 --policies rollout1 --tag full
run r1b100_t123_s0 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 0 --r1-base 100 --policies rollout1 --tag full
run r1b5_t4_s0 --tables t4 --K 1 2 3 --nmc 100 --seed 0 --r1-base 5 --policies rollout1 --tag full
run r1b10_t4_s0 --tables t4 --K 1 2 3 --nmc 100 --seed 0 --r1-base 10 --policies rollout1 --tag full
run tau2_t123_s1 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 1 --tau 2 --policies whittle --tag full_s1
run tau2_t4_s1 --tables t4 --K 1 2 3 --nmc 100 --seed 1 --tau 2 --policies whittle --tag full_s1
run tau3_t123_s1 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 1 --tau 3 --policies whittle --tag full_s1
run tau3_t4_s1 --tables t4 --K 1 2 3 --nmc 100 --seed 1 --tau 3 --policies whittle --tag full_s1
run tau5_t123_s1 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 1 --tau 5 --policies whittle --tag full_s1
run tau5_t4_s1 --tables t4 --K 1 2 3 --nmc 100 --seed 1 --tau 5 --policies whittle --tag full_s1
run tau10_t123_s1 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 1 --tau 10 --policies whittle --tag full_s1
run tau10_t4_s1 --tables t4 --K 1 2 3 --nmc 100 --seed 1 --tau 10 --policies whittle --tag full_s1
run tau20_t123_s1 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 1 --tau 20 --policies whittle --tag full_s1
run tau20_t4_s1 --tables t4 --K 1 2 3 --nmc 100 --seed 1 --tau 20 --policies whittle --tag full_s1
run tau50_t123_s1 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 1 --tau 50 --policies whittle --tag full_s1
run tau50_t4_s1 --tables t4 --K 1 2 3 --nmc 100 --seed 1 --tau 50 --policies whittle --tag full_s1
run tau200_t123_s1 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 1 --tau 200 --policies whittle --tag full_s1
run tau200_t4_s1 --tables t4 --K 1 2 3 --nmc 100 --seed 1 --tau 200 --policies whittle --tag full_s1
run r1b5_t123_s1 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 1 --r1-base 5 --policies rollout1 --tag full_s1
run r1b10_t123_s1 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 1 --r1-base 10 --policies rollout1 --tag full_s1
run r1b100_t123_s1 --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed 1 --r1-base 100 --policies rollout1 --tag full_s1
run r1b5_t4_s1 --tables t4 --K 1 2 3 --nmc 100 --seed 1 --r1-base 5 --policies rollout1 --tag full_s1
run r1b10_t4_s1 --tables t4 --K 1 2 3 --nmc 100 --seed 1 --r1-base 10 --policies rollout1 --tag full_s1
run r1grid_b1_s0 --tables t4 --K 1 2 3 --nmc 100 --seed 0 --r1-base 1 --policies rollout1 --tag r1_s0 --runs-root runs/review
run r1grid_b1_s1 --tables t4 --K 1 2 3 --nmc 100 --seed 1 --r1-base 1 --policies rollout1 --tag r1_s1 --runs-root runs/review
run r1grid_b5_s1_rk --tables t4 --qsets reckless --K 2 3 --nmc 100 --seed 1 --r1-base 5 --policies rollout1 --tag r1_s1 --runs-root runs/review
run r1grid_b5_s1_cm --tables t4 --qsets cautious mixed --K 1 2 3 --nmc 100 --seed 1 --r1-base 5 --policies rollout1 --tag r1_s1 --runs-root runs/review
run r1grid_b5_s0_cm --tables t4 --qsets cautious mixed --K 1 2 3 --nmc 100 --seed 0 --r1-base 5 --policies rollout1 --tag r1_s0 --runs-root runs/review
run r1grid_b100_K1_s0 --tables t4 --K 1 --nmc 100 --seed 0 --r1-base 100 --policies rollout1 --tag r1_s0 --runs-root runs/review
run r1grid_b100_K1_s1 --tables t4 --K 1 --nmc 100 --seed 1 --r1-base 100 --policies rollout1 --tag r1_s1 --runs-root runs/review
run r1grid_b100_K2_s0 --tables t4 --K 2 --nmc 100 --seed 0 --r1-base 100 --policies rollout1 --tag r1_s0 --runs-root runs/review
run r1grid_b100_K2_s1 --tables t4 --K 2 --nmc 100 --seed 1 --r1-base 100 --policies rollout1 --tag r1_s1 --runs-root runs/review
run r1grid_b100_K3_s0 --tables t4 --K 3 --nmc 100 --seed 0 --r1-base 100 --policies rollout1 --tag r1_s0 --runs-root runs/review
run r1grid_b100_K3_s1 --tables t4 --K 3 --nmc 100 --seed 1 --r1-base 100 --policies rollout1 --tag r1_s1 --runs-root runs/review
# rollout1 base lengths 1, 2, 3 on the scalar tables and 2, 3 on the 4-D table (completes the grid {1,2,3,5,10,30,100})
for s in 0 1; do
  for b in 1 2 3; do run r1grid_t123_b${b}_s$s --tables t1 t2 t3 --K 1 2 3 --nmc 100 --seed $s --r1-base $b --policies rollout1 --tag r1_s$s --runs-root runs/review; done
  for b in 2 3; do run r1grid_t4_b${b}_s$s --tables t4 --K 1 2 3 --nmc 100 --seed $s --r1-base $b --policies rollout1 --tag r1_s$s --runs-root runs/review; done
done
# skip-negative variant rerun after the nonnegative-index fix (idx >= 0 is tracked)
run skipneg_fix --tables t1 t2 t3 --K 1 2 3 --nmc 100 --skip-neg --policies whittle myopic tev --tag skipneg_fix --runs-root runs/review
# extension X3 (sampled modes) on Figs 5-6 with the bound, K = 2, 6, 10
for f in fig5a fig5b fig5c fig6; do
  run x4_$f --fig $f --Ks 2 6 10 --nmc 100 --sampled --policies whittle myopic tev --lb --tag full_x4
done
