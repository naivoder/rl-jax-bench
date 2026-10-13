#!/bin/sh
# Differential tests: our side (JAX, CPU) -> the authors' code on the same inputs (torch + CleanRL / baselines /
# gymnasium, in their own venvs under $OFFICIAL_ROOT) -> compare -> rebuild the committed test fixtures.
set -e
cd "$(dirname "$0")/../.."
PAPER=$(pwd)
PY=${PY:-"uv run python"}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-"$PAPER/external/official"}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
D=$PAPER/runs/small/difftest
$PY -m scripts.small.difftest_export
$PY -m scripts.small.wrapper_export
$PY -m scripts.small.env_export
$PY -m scripts.official_check.metric_export
cd "$OFFICIAL_ROOT"
OMP_NUM_THREADS=2 .venv/bin/python -I "$PAPER/scripts/official/oracle_difftest.py" "$D"
.venv/bin/python -I "$PAPER/scripts/official/oracle_wrappers.py" "$D"
.venv/bin/python -I "$PAPER/scripts/official/oracle_envs.py" "$D"
# the post's own environment stack (gym 0.21.0, python 3.8 venv)
.venv_post/bin/python -I "$PAPER/scripts/official/oracle_wrappers_gym021.py" "$D"
.venv_post/bin/python -I "$PAPER/scripts/official/oracle_envs_gym021.py" "$D"
.venv/bin/python -I "$PAPER/scripts/official/oracle_metric.py" "$D"
.venv_post/bin/python -I "$PAPER/scripts/official/oracle_metric.py" "$D"
cd "$PAPER"
$PY -m scripts.small.difftest_compare
$PY -m scripts.small.make_parity_fixtures
