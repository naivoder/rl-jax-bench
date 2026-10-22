#!/bin/sh
# Fetch and set up the authors' code (and the Stone Soup cross-check) used as oracles.
#   OFFICIAL_ROOT (default external/official inside this folder) receives:
#     repo/            sggjon5/GOSPA-Driven-Non-Myopic-Multi-Sensor-Management-with-Multi-Bernoulli-Filtering @ b5959f6 (2026-05-04)
#     .venv/           Python 3.12, numpy 1.26.4, scipy 1.11.4, matplotlib 3.8.4: the authors' own era, for the scripts in
#                      scripts/official/ that run their code (run_main.py, trace_main.py, record_official_fixtures.py, gen_truths.py)
#     stonesoup_venv/  Python 3.12 with Stone Soup 1.9.1, for tests/test_gospa_parity.py::test_gospa_stonesoup_crosscheck
# No patches are needed: the authors' code is imported unmodified and our drivers apply their few overrides in-process.
# After this script the live differential tests (tests/test_*_parity.py except test_official_parity.py) run instead of skipping;
# they import the authors' modules into the repo environment, which needs only numpy, scipy and matplotlib.
# Needs git, uv and network access; CPU only. Untested as a whole script (it reproduces a hand-built setup).
set -eu
cd "$(dirname "$0")/../.."
ROOT=${OFFICIAL_ROOT:-external/official}
mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd)
URL=https://github.com/sggjon5/GOSPA-Driven-Non-Myopic-Multi-Sensor-Management-with-Multi-Bernoulli-Filtering.git
COMMIT=b5959f6a696a0e566e0925ec2376be361982e773

if [ ! -d "$ROOT/repo/.git" ]; then
    git init -q "$ROOT/repo"
    git -C "$ROOT/repo" remote add origin "$URL"
fi
git -C "$ROOT/repo" fetch -q --depth 1 origin "$COMMIT"
git -C "$ROOT/repo" checkout -q "$COMMIT"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    uv venv -q --python 3.12 "$ROOT/.venv"
fi
uv pip install -q --python "$ROOT/.venv/bin/python" numpy==1.26.4 scipy==1.11.4 matplotlib==3.8.4

if [ ! -x "$ROOT/stonesoup_venv/bin/python" ]; then
    uv venv -q --python 3.12 "$ROOT/stonesoup_venv"
fi
uv pip install -q --python "$ROOT/stonesoup_venv/bin/python" stonesoup==1.9.1

echo "authors' code ready in $ROOT (export OFFICIAL_ROOT=$ROOT if it is not the default)"
