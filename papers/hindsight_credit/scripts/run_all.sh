#!/bin/sh
# Every run behind results/, in order, from the paper folder. Sequential; the whole set is well under one hour
# on one GPU (each run is a tabular, vmapped JAX call). Outputs go to runs/ (git-ignored).
#   PY: the JAX environment (default: the repo environment through uv).
#   OFFICIAL_ROOT: authors' clone as repo/ and its numpy-only venv as venv/ (default external/official);
#     only the official-code steps need it.
set -e
PY=${PY:-"uv run python"}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
OPY="$OFFICIAL_ROOT/venv/bin/python -I"

# exact checks and differential test against the authors' snippet
$PY -m scripts.exact_checks
$PY -m scripts.diff_official
# small scale (20 runs, seed 2)
$PY -m scripts.small_scale
# official-code reference runs at paper scale (numpy venv) and the end-to-end lockstep
mkdir -p runs/official/ref
for e in E1 E3 E4 E5 E6 E7 E8; do
  $OPY scripts/official/ref_driver.py independent --exp $e --runs 100 --seed 0 --out runs/official/ref/$e.json
done
$OPY scripts/official/ref_driver.py fig3r --runs 100 --seed 0 --out runs/official/ref/E2.json
$PY -m scripts.official_lockstep
# NOTE: runs/official/ours_E2_official_immediate.json is read by official_reference_table.py; the command that
# wrote it was not recorded in the original run log.
$PY scripts/official_reference_table.py
# full scale: PG lr tuning (seed 999), evaluation seeds 0, 1, 3, ablations, theory, divergence probes
$PY -m scripts.full tune
for s in 0 1 3; do $PY -m scripts.full eval --seed $s; done
for c in ablation theory divergence oracle_hz cost; do $PY -m scripts.full $c; done
$PY scripts/report.py
$PY scripts/report.py --pre-recheck
# sensitivities requested in review
for c in bins eps0 slide f3rv alpha x0_alias vlr; do $PY -m scripts.sensitivity $c; done
$PY scripts/sensitivity_report.py
# recheck: h-learner readings on seeds 0, 1, 3 and fresh seeds 4, 5; paper-matched PG; E1 start
$PY -m scripts.recheck all
$PY scripts/recheck_report.py
