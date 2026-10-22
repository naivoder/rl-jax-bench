#!/bin/sh
# Reference runs of the authors' UNMODIFIED main.py (via run_main.py) at small n, then the three-way summary table.
# Job list as recorded for the published numbers: block seeds 123546 + 1000 i (i = 0 is the authors' own seed).
# Needs `sh scripts/official/setup.sh` first. One main.py run is minutes (Myopic) to about an hour (all MCTS blocks) on one core.
set -eu
cd "$(dirname "$0")/../.."
ROOT=${OFFICIAL_ROOT:-external/official}
OPY="$ROOT/.venv/bin/python"
PY=${PY:-"uv run python"}
OUT=${GOSPA_RUNS:-runs}/ref_main
mkdir -p "$OUT"
for i in 0 1 2 3; do
    seed=$((123546 + 1000 * i))
    for lam in 2 0.1 1; do
        $OPY scripts/official/run_main.py --lam $lam --runs 1 --block-seed $seed --cost gospa --out "$OUT/def_gospa_lam${lam}_s$i.json"
    done
done
for lam in 2 0.1 1; do
    $OPY scripts/official/run_main.py --lam $lam --runs 1 --block-seed 123546 --cost kld --mcts 1,2,3,4 --out "$OUT/def_kld_lam${lam}_s0.json"
done
for i in 0 1 2; do       # the paper's stated r_b = 0.03 and R = 2 I in the authors' code
    $OPY scripts/official/run_main.py --lam 2 --runs 1 --block-seed $((123546 + 1000 * i)) --cost gospa --paper-params --out "$OUT/pp_gospa_lam2_s$i.json"
done
# our side of the comparison: 10-run CFG-O (tag small, see scripts/run_all.sh) and CFG-O with the same two overrides (20 runs)
$PY -m src.train --cfg CFG-O --runs 20 --clutter 2 --tag pp_ours --set r_b=0.03 R_var=2.0
$PY scripts/merge_metrics.py ${GOSPA_RUNS:-runs}/pp_ours
$PY scripts/official/summarize_official_reference.py --json ${GOSPA_RUNS:-runs}/report/official_reference_summary.json
