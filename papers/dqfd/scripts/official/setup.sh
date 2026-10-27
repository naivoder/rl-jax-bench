#!/bin/sh
# Fetches the reference code used by the differential tests and the torch reference run (no official DQfD code exists).
# Run from the paper folder:  sh scripts/official/setup.sh
#   $OFFICIAL_ROOT/MinAtar     kenjyoung/MinAtar at 4241f246 (the original numpy MinAtar; needed by tests/test_recheck.py
#                              and scripts/record_official_env.py / policy_parity.py)
#   $OFFICIAL_ROOT/DI-engine   opendilab/DI-engine at d0b21d06 (dqfd_nstep_td_error; scripts/official/oracle_di_engine.py)
#   $OFFICIAL_ROOT/.venv       Python 3.12, torch 2.5.1+cpu, numpy 2.5.4 (for the two torch scripts in this folder only)
# Nothing is patched. Requires git and uv.
set -e
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
mkdir -p "$OFFICIAL_ROOT"

fetch() {   # fetch URL DIR COMMIT
  if [ ! -d "$2/.git" ]; then git clone --quiet "$1" "$2"; fi
  git -C "$2" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "$2" checkout --quiet "$3"
  echo "$2 at $(git -C "$2" rev-parse HEAD)"
}
fetch https://github.com/kenjyoung/MinAtar.git "$OFFICIAL_ROOT/MinAtar" 4241f24685622cfb52f843cab9bbcf385a53484a
fetch https://github.com/opendilab/DI-engine.git "$OFFICIAL_ROOT/DI-engine" d0b21d065b2d80317cc16b28a95301be8871d950

if [ "${SKIP_VENV:-0}" != "1" ]; then
  uv venv --python 3.12 "$OFFICIAL_ROOT/.venv"
  uv pip install --python "$OFFICIAL_ROOT/.venv/bin/python" --index-url https://download.pytorch.org/whl/cpu \
     --extra-index-url https://pypi.org/simple "torch==2.5.1+cpu" "numpy==2.5.4"
fi
echo "done; the skipped tests in tests/test_recheck.py now run (pytest -q)"
