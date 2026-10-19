#!/bin/sh
# Build the two oracle environments used by tests/test_oracle.py and the scripts in scripts/official/.
# No code was released for the paper. The oracles are Stone Soup (the tracking library the paper names) and the first
# author's public forks of stable-baselines3 and imitation, pinned to the commits that were used.
#   $OFFICIAL_ROOT/.venv       Python 3.11, Stone Soup 1.9.1 + filterpy 1.4.5   (tests/test_oracle.py, ss_*_ref.py)
#   $OFFICIAL_ROOT/.venv-sb3   Python 3.11, torch 2.3.1, the two forks, jax 0.7.1 (sb3_*.py, mhsa_sb3.py)
# Requires uv and git. Run from the paper folder: sh scripts/official/setup.sh
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
mkdir -p "$OFFICIAL_ROOT"
cd "$OFFICIAL_ROOT"
clone() {  # url dir commit
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" checkout --quiet "$3"
}
clone https://github.com/iwishiwasaneagle/stable-baselines3.git sb3_fork e46a6d5edab8ee467b733385451f86fe06d06179
clone https://github.com/iwishiwasaneagle/imitation.git imitation_fork f5cb8a473dc27bd9a40dc2466251fcadf765801c
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r "$HERE/requirements-stonesoup.lock.txt"
[ -x .venv-sb3/bin/python ] || uv venv --python 3.11 .venv-sb3
uv pip install --python .venv-sb3/bin/python -r "$HERE/requirements-sb3.lock.txt"
uv pip install --python .venv-sb3/bin/python --no-deps -e ./sb3_fork -e ./imitation_fork
echo "oracle environments ready in $OFFICIAL_ROOT"
