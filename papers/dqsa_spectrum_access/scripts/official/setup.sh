#!/bin/sh
# Fetch and build the third-party reference used as an oracle (the authors released no code for this paper):
#   shkrwnd/Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access at the pinned commit, cloned into $OFFICIAL_ROOT/shkrwnd,
#   and a Python 3.11 venv with tensorflow-cpu 2.15.1 (tf.compat.v1, pins in tf_requirements.txt) at $OFFICIAL_ROOT/.venv.
# The repository is a TF1 script; run_shk.py / make_fixtures.py supply a small tf.contrib shim at run time, so no patch file is needed.
# Usage (from the paper folder): sh scripts/official/setup.sh      OFFICIAL_ROOT defaults to external/official.
set -e
cd "$(dirname "$0")/../.."
HERE=$(pwd)
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
COMMIT=2219c28b6fd12c416dfd32763641b0786c32bafc
mkdir -p "$OFFICIAL_ROOT"
cd "$OFFICIAL_ROOT"
if [ ! -d shkrwnd/.git ]; then
  git clone https://github.com/shkrwnd/Deep-Reinforcement-Learning-for-Dynamic-Spectrum-Access.git shkrwnd
fi
git -C shkrwnd checkout -q "$COMMIT"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python -r "$HERE/scripts/official/tf_requirements.txt"
fi
echo "third-party reference ready in $(pwd)"
