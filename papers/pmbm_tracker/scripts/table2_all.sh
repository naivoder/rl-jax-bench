#!/bin/sh
# Table II sweep: 15 (p_d, lambda_c) cells x 100 Monte-Carlo runs on the paper's Fig. 3 truth, measurement root PRNGKey(3003).
# Plain sequential version of the original job queue (which ran at most 4 single-threaded processes at a time).
# Arms: PMBM paper path (graded) and with the 1e-4 global-weight floor (sensitivity); GM-PHD / GM-CPHD in three constant
# sets; Williams TOMB / MOMB under GNU Octave (needs scripts/official/setup.sh first). Original cost: a few CPU-days.
# A nonzero overflow counter makes src.train exit nonzero (zero-counter rule), which stops this script.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
W=${TABLE2_WORK:-runs/table2_work}
RUNS=${RUNS:-100}
ROOT=$(pwd)

$PY scripts/export_table2_meas.py "$W/meas" --runs "$RUNS"          # Octave inputs + sha256 of every set
cp "$W/meas/table2_meas_sha256.json" runs/

for p in 0.95 0.9 0.8 0.7 0.6; do for l in 10 15 20; do
  T="--truth fig3 --meas-seed 3003 --pd $p --lam $l --runs $RUNS --workers 1"
  $PY -m src.train $T --algos pmbm --tag table2_pmbm_pd${p}_l${l}                       # graded: paper path, prune_w 0
  $PY -m src.train $T --algos pmbm --prune-w 1e-4 --tag table2_pmbmpw_pd${p}_l${l}      # sensitivity: authors' later floor
  $PY -m src.train $T --algos phd,phd_vm,cphd --phd-merge 4 --cphd-merge 4 --phd-max 100 --cphd-max 100 \
      --tag table2_cited_pd${p}_l${l}                                                    # graded PHD/CPHD constants (merge 4, cap 100)
  $PY -m src.train $T --algos phd,phd_vm,cphd --tag table2_repo_pd${p}_l${l}             # authors' repo defaults (merge 0.1, cap 30)
  $PY -m src.train $T --algos phd,phd_vm,cphd --phd-merge 4 --cphd-merge 4 --tag table2_m4c30_pd${p}_l${l}
done; done

# Williams TOMB / MOMB (authors' Octave code) on the same measurement sets, then OSPA scoring with both extraction rules.
for p in 0.95 0.9 0.8 0.7 0.6; do for l in 10 15 20; do
  d=$ROOT/$W/meas/pd${p}_l${l}; o=$ROOT/$W/wout/pd${p}_l${l}
  for alg in TOMB MOMB; do
    mkdir -p "$o/out_$alg"
    (cd scripts/official && sh run_octave.sh --eval "drive_williams('$d/in_williams.mat','$o/out_$alg','$alg')")
  done
  for rule in thr map; do
    $PY scripts/eval_williams.py "$d/meas.npz" "$o" table2_wil_pd${p}_l${l} --rule $rule
  done
done; done

$PY scripts/table2_report.py --out runs/table2.md --json runs/table2.json
