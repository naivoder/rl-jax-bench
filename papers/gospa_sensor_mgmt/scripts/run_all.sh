#!/bin/sh
# Every closed-loop experiment behind the README tables, run one after another (CPU, single-threaded; nothing is learned).
# The original runs used 1-4 parallel processes; this is the sequential equivalent. Roughly 25 CPU core-hours in total.
# Outputs go to $GOSPA_RUNS (default runs/). Each train.py call resumes from its last checkpoint if interrupted.
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
ALGS="Myopic-GD MCTS1-GD MCTS2-GD MCTS3-GD MCTS4-GD Myopic-KLD MCTS1-KLD MCTS2-KLD MCTS3-KLD MCTS4-KLD RANDOM HOLD"
R=${GOSPA_RUNS:-runs}

# 10-run pass of both configurations (tag small); also the reference for the determinism check P8
for cfg in CFG-O CFG-P; do for a in $ALGS; do
    $PY -m src.train --cfg $cfg --algs $a --runs 10 --clutter 0.1 1 2 --tag small
done; done
$PY scripts/merge_metrics.py $R/small

# E1 at the paper's scale: 12 algorithms x 3 clutter rates x 50 runs, CFG-O, CFG-P (v1, archived) and CFG-P2
for cfg in CFG-O CFG-P; do for a in $ALGS; do
    $PY -m src.train --cfg $cfg --algs $a --runs 50 --clutter 0.1 1 2 --tag e1
done; done
for a in $ALGS; do $PY -m src.train --cfg CFG-P2 --algs $a --runs 50 --clutter 0.1 1 2 --tag p2; done
$PY scripts/merge_metrics.py $R/e1; $PY scripts/merge_metrics.py $R/p2

# sensitivity flips F-S and hyper-parameter parity F-T on CFG-O (tag e1s)
$PY scripts/arms.py jobs | while read -r cfg tag alg rest; do
    $PY -m src.train --cfg $cfg --algs $alg --runs 50 --clutter 0.1 1 2 --tag $tag $rest
done
# CFG-P divergence arms P9 (tag e1d), flown on CFG-P v1
for a in Myopic-GD MCTS3-GD; do
    $PY -m src.train --cfg CFG-P --algs $a --runs 50 --clutter 0.1 1 2 --tag e1d --set filt_pd=hard
    $PY -m src.train --cfg CFG-P --algs $a --runs 50 --clutter 0.1 1 2 --tag e1d --set lbp_mode=one_sweep lbp_tol=1e-4 lbp_max_iter=1
    $PY -m src.train --cfg CFG-P --algs $a --runs 50 --clutter 0.1 1 2 --tag e1d --set filt_pd=hard truth_pd=hard
done
# S-TRUTH: ten other truths from the authors' generator (P4b), and the H2 power extension to 200 runs (P10)
for a in $ALGS; do $PY -m src.train --cfg CFG-O --algs $a --runs 50 --clutter 0.1 1 2 --tag e1t --truths data/truths_strut.npz; done
for a in MCTS1-GD MCTS2-GD MCTS3-GD MCTS4-GD; do
    $PY -m src.train --cfg CFG-O --algs $a --runs 150 --run-offset 50 --clutter 0.1 1 2 --tag e2h
done
# CFG-P2 follow-up flips (tag p2_sens), 50 runs each. The original pass used 25 runs in a few cells and skipped, for time,
# info_gain at Myopic-KLD / MCTS4-KLD, budget_mode=iter at MCTS4-KLD, plan_trace=pos at MCTS1-GD, est_rule=gamma and merge_each at MCTS3-KLD.
for spec in "kld_mode=info_gain:Myopic-KLD MCTS1-KLD MCTS2-KLD MCTS3-KLD MCTS4-KLD" "budget_mode=iter:MCTS3-GD MCTS4-GD MCTS3-KLD MCTS4-KLD" \
            "plan_trace=pos:Myopic-GD MCTS1-GD MCTS3-GD MCTS4-GD" "est_rule=gamma:Myopic-GD MCTS3-GD MCTS3-KLD" \
            "merge_each=true:Myopic-GD MCTS3-GD MCTS3-KLD" "uct_eps=300:MCTS3-GD MCTS3-KLD"; do
    flip=${spec%%:*}; for a in ${spec#*:}; do
        $PY -m src.train --cfg CFG-P2 --algs $a --runs 50 --clutter 0.1 1 2 --tag p2_sens --set $flip
    done
done
for t in e1s e1d e1t e2h p2_sens; do $PY scripts/merge_metrics.py $R/$t; done

# reports (JSON in runs/report/; the shipped copies are in results/)
mkdir -p $R/report
for part in e1 sens strut div h2; do $PY scripts/report.py --part $part --out $R/report/${part}_report.json; done
$PY scripts/followup/authors_zscores.py; $PY scripts/followup/cfg_p2_table.py; $PY scripts/followup/sensitivity.py
$PY scripts/theory_t1.py --json $R/report/t1.json; $PY scripts/theory_t2_t3.py --json $R/report/t2t3.json
$PY scripts/e3_snapshot.py --png $R/report/e3_frame152.png --json $R/report/e3.json
$PY scripts/e5_timing.py --cfg CFG-O --json $R/report/e5.json
$PY scripts/figs.py $R/report/e1_report.json $R/report
