#!/bin/sh
# Fetches the authors' code and builds the venv used by the scripts in this folder.
# Run from the paper folder:  sh scripts/official/setup.sh
#   $OFFICIAL_ROOT/asym-rlpo-aamas2022   abaisero/asym-rlpo at 1a065b0c (run unmodified; put on sys.path by our scripts)
#   $OFFICIAL_ROOT/gym-pomdps            abaisero/gym-pomdps at b52e5203     (installed into the venv)
#   $OFFICIAL_ROOT/gym-gridverse         abaisero/gym-gridverse at 9375a10e  (installed into the venv)
#   $OFFICIAL_ROOT/rl-parsers            abaisero/rl-parsers at b2197e3f     (installed into the venv)
#   $OFFICIAL_ROOT/one-to-one            abaisero/one-to-one at 0ae94b5c     (installed into the venv)
#   $OFFICIAL_ROOT/.venv                 Python 3.9, torch 1.13.1+cpu, gym 0.21.0, numpy 1.23.5
# Nothing is patched. Requires git and uv.
# Then, for the full-size oracle tests (tests/test_official_{env,carflag,cleaner,losses}.py, tests/test_schedules.py):
#   for s in env_traces carflag cleaner losses schedules; do
#     $OFFICIAL_ROOT/.venv/bin/python scripts/official/export_$s.py; done
#   $OFFICIAL_ROOT/.venv/bin/python scripts/official/export_losses.py --f64
# and for the authors'-code reference curves read by src/eval.py:  sh scripts/official/reference_runs.sh
set -e
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
mkdir -p "$OFFICIAL_ROOT/out"

fetch() {   # fetch URL DIR COMMIT
  if [ ! -d "$2/.git" ]; then git clone --quiet "$1" "$2"; fi
  git -C "$2" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "$2" checkout --quiet "$3"
  echo "$2 at $(git -C "$2" rev-parse HEAD)"
}
fetch https://github.com/abaisero/asym-rlpo.git "$OFFICIAL_ROOT/asym-rlpo-aamas2022" 1a065b0cb48b829d14c22fd11bb8ae48b70010a7
fetch https://github.com/abaisero/gym-pomdps.git "$OFFICIAL_ROOT/gym-pomdps" b52e5203d3e7ac079c032e11c3b721963c44cca9
fetch https://github.com/abaisero/gym-gridverse.git "$OFFICIAL_ROOT/gym-gridverse" 9375a10e3dc97ef91e709785e08368dba48bc2b4
fetch https://github.com/abaisero/rl-parsers.git "$OFFICIAL_ROOT/rl-parsers" b2197e3f7af2190641599e6b38aa4151160e0a64
fetch https://github.com/abaisero/one-to-one.git "$OFFICIAL_ROOT/one-to-one" 0ae94b5c8ce928388cb312eb6d0adfae0629c29b

if [ "${SKIP_VENV:-0}" != "1" ]; then
  uv venv --python 3.9 "$OFFICIAL_ROOT/.venv"
  PY="$OFFICIAL_ROOT/.venv/bin/python"
  uv pip install --python "$PY" "setuptools==57.5.0" "wheel==0.37.1"
  uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu "torch==1.13.1+cpu"
  uv pip install --python "$PY" --no-build-isolation "gym==0.21.0"
  uv pip install --python "$PY" "numpy==1.23.5" "ply==3.11" "more-itertools==10.8.0" "pyyaml==6.0.3" \
     "pytest==8.4.2" "cached-property==2.0.1" "imageio==2.37.2" "imageio-ffmpeg==0.6.0" "schema==0.7.8" \
     "termcolor==3.1.0" "typing-extensions==4.16.0" "pygame==2.6.1" "pyglet==1.5.27" "wandb==0.26.1"
  uv pip install --python "$PY" --no-deps "$OFFICIAL_ROOT/rl-parsers" "$OFFICIAL_ROOT/one-to-one" \
     "$OFFICIAL_ROOT/gym-pomdps" "$OFFICIAL_ROOT/gym-gridverse"
fi
echo "done; now run the export scripts listed at the top of this file"
