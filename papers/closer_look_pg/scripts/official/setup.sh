#!/bin/sh
# Fetches the authors' companion code and builds the venv used by tests/test_oracle.py and the scripts in this folder.
# Run from the paper folder:  sh scripts/official/setup.sh
#   $OFFICIAL_ROOT/implementation-matters   MadryLab/implementation-matters at 5ee6ecb1 (run unmodified)
#   $OFFICIAL_ROOT/.venv                     Python 3.12, torch 2.5.1+cpu, gymnasium 1.0.0, mujoco 3.15.0, numpy 2.5.4
# Nothing is patched. Requires git and uv. Reference-run outputs (ref/, grad_ref*/, env_ref/) are written next to them.
set -e
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
mkdir -p "$OFFICIAL_ROOT"

fetch() {   # fetch URL DIR COMMIT
  if [ ! -d "$2/.git" ]; then git clone --quiet "$1" "$2"; fi
  git -C "$2" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "$2" checkout --quiet "$3"
  echo "$2 at $(git -C "$2" rev-parse HEAD)"
}
fetch https://github.com/MadryLab/implementation-matters.git "$OFFICIAL_ROOT/implementation-matters" \
      5ee6ecb12545365d9178135e65576adfc0d82f52

if [ "${SKIP_VENV:-0}" != "1" ]; then
  uv venv --python 3.12 "$OFFICIAL_ROOT/.venv"
  uv pip install --python "$OFFICIAL_ROOT/.venv/bin/python" --index-url https://download.pytorch.org/whl/cpu \
     --extra-index-url https://pypi.org/simple "torch==2.5.1+cpu" "numpy==2.5.4" "gymnasium==1.0.0" "mujoco==3.15.0" \
     "scipy==1.18.1" "pandas==3.0.6" "tables==3.11.1" "tensorboardX==2.6.5" "dill==0.4.1" "tqdm==4.70.1" \
     "matplotlib==3.11.2" "scikit-learn==1.9.1"
fi
echo "done; tests/test_oracle.py now runs (pytest -q)"
