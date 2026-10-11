#!/bin/sh
# Fetches the authors' HCA snippet and builds the numpy-only venv used by scripts/official/run_official.py,
# scripts/official/ref_driver.py, scripts/diff_official.py and scripts/official_lockstep.py.
# Run from the paper folder:  sh scripts/official/setup.sh
#   $OFFICIAL_ROOT/repo    hca-neurips2019/hca at 3a564ad6 (hca_classes.py, run unmodified)
#   $OFFICIAL_ROOT/venv    Python 3.12, numpy 2.5.4
# Nothing is patched. Requires git and uv. No test needs this: the official parity tests replay shipped fixtures.
set -e
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
mkdir -p "$OFFICIAL_ROOT"

fetch() {   # fetch URL DIR COMMIT
  if [ ! -d "$2/.git" ]; then git clone --quiet "$1" "$2"; fi
  git -C "$2" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "$2" checkout --quiet "$3"
  echo "$2 at $(git -C "$2" rev-parse HEAD)"
}
fetch https://github.com/hca-neurips2019/hca.git "$OFFICIAL_ROOT/repo" 3a564ad6438df2b1c68ca75a91d713d87d56453a

if [ "${SKIP_VENV:-0}" != "1" ]; then
  uv venv --python 3.12 "$OFFICIAL_ROOT/venv"
  uv pip install --python "$OFFICIAL_ROOT/venv/bin/python" "numpy==2.5.4"
fi
echo "done"
