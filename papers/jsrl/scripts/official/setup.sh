#!/bin/sh
# Fetch everything the third-party reference checks need that is not shipped in this folder.
# No official JSRL code was released; the reference is the third-party SB3 implementation steventango/jumpstart-rl.
#   1. steventango/jumpstart-rl @ e5f690fe (unmodified, no patches)           -> $OFFICIAL_ROOT/jumpstart-rl
#   2. the original numpy MinAtar, kenjyoung/MinAtar @ 4241f246 (unmodified) -> $OFFICIAL_ROOT/MinAtar
#      (scripts/official/record_official_env.py, policy_parity.py, run_parity.sh, on_our_task/)
#   3. a Python 3.12 venv with the versions the reference runs used          -> $OFFICIAL_ROOT/.venv
#   4. stable-baselines3 2.7.0 (its n-step replay buffer) installed with --target -> $OFFICIAL_ROOT/sb3_nstep
#      (on_our_task/ref_run.py with NSTEP 10, on_our_task/nstep_theirs.py)
# Run from anywhere: sh scripts/official/setup.sh. Needs git, uv and network access. Idempotent.
set -e
HERE=$(cd "$(dirname "$0")/../.." && pwd)
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$HERE/external/official}
mkdir -p "$OFFICIAL_ROOT"

fetch() {  # url dir commit
  if [ ! -d "$2/.git" ]; then git clone "$1" "$2"; fi
  git -C "$2" checkout --quiet "$3"
  echo "$2 at $(git -C "$2" rev-parse HEAD)"
}
fetch https://github.com/steventango/jumpstart-rl "$OFFICIAL_ROOT/jumpstart-rl" e5f690fe0a766558be9455eff605078597285e51
fetch https://github.com/kenjyoung/MinAtar "$OFFICIAL_ROOT/MinAtar" 4241f246

if [ ! -x "$OFFICIAL_ROOT/.venv/bin/python" ]; then
  uv venv --python 3.12 "$OFFICIAL_ROOT/.venv"
  uv pip install --python "$OFFICIAL_ROOT/.venv/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.5.1+cpu" "stable-baselines3==2.4.0" "gymnasium==1.0.0" "gymnasium-robotics==1.3.1" "mujoco==3.1.6" \
    "numpy==1.26.4" "pandas==3.0.6" "matplotlib==3.11.2" "pyyaml==6.0.3"
fi
if [ ! -d "$OFFICIAL_ROOT/sb3_nstep/stable_baselines3" ]; then
  uv pip install --python "$OFFICIAL_ROOT/.venv/bin/python" --no-deps --target "$OFFICIAL_ROOT/sb3_nstep" "stable-baselines3==2.7.0"
fi
echo "done: OFFICIAL_ROOT=$OFFICIAL_ROOT"
echo "run the reference scripts with: PYTHONPATH=$OFFICIAL_ROOT/jumpstart-rl/src $OFFICIAL_ROOT/.venv/bin/python scripts/official/<script>.py ..."
