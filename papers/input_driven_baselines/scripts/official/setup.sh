#!/bin/sh
# Fetches the authors' code and builds its venv, used by tests/test_official_env.py (the clone only) and by the
# scripts in this folder (the venv).
# Run from the paper folder:  sh scripts/official/setup.sh
#   $OFFICIAL_ROOT/input_driven_rl_example   hongzimao/input_driven_rl_example at 6c9cc231 (run unmodified; the
#                                            TF1 code runs through the tf.compat.v1 alias in run_official.py)
#   $OFFICIAL_ROOT/.venv                     Python 3.8, tensorflow-cpu 2.13.1, numpy 1.24.3, matplotlib 3.7.5
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
fetch https://github.com/hongzimao/input_driven_rl_example.git "$OFFICIAL_ROOT/input_driven_rl_example" \
      6c9cc23174651e3890721a3f6420ca98a90eb99f

if [ "${SKIP_VENV:-0}" != "1" ]; then
  uv venv --python 3.8 "$OFFICIAL_ROOT/.venv"
  uv pip install --python "$OFFICIAL_ROOT/.venv/bin/python" "tensorflow-cpu==2.13.1" "numpy==1.24.3" \
     "matplotlib==3.7.5"
fi
echo "done; tests/test_official_env.py now runs (pytest -q)"
