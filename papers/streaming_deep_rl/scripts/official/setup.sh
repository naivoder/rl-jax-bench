#!/bin/sh
# Fetch everything the reference checks need that is not shipped in this folder:
#   1. the authors' code, github.com/mohmdelsayed/streaming-drl, branch 2024 @ 407dca7 (unmodified, no patches)
#      -> $OFFICIAL_ROOT/repo
#   2. a Python 3.11 venv with the versions the reference runs used -> $OFFICIAL_ROOT/.venv
#   3. the paper PDF (arXiv:2410.14606v2, sha256-checked) -> external/data/2410.14606v2.pdf (tests/test_digitize_calib.py,
#      scripts/digitize_figs.py; these also need the root `pdf` extra: `uv sync --extra pdf`)
#   4. ETTm2.csv (sha256-checked) -> external/data/ETTm2.csv (src/data.py would otherwise download it on first use)
# Run from anywhere: sh scripts/official/setup.sh. Needs git, curl, uv and network access. Idempotent.
set -e
HERE=$(cd "$(dirname "$0")/../.." && pwd)
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$HERE/external/official}
DATA=$HERE/external/data
COMMIT=407dca7a8b584c1c20bc649053557f66e270b1e6
mkdir -p "$OFFICIAL_ROOT" "$DATA"

if [ ! -d "$OFFICIAL_ROOT/repo/.git" ]; then
  git clone --branch 2024 https://github.com/mohmdelsayed/streaming-drl "$OFFICIAL_ROOT/repo"
fi
git -C "$OFFICIAL_ROOT/repo" checkout --quiet "$COMMIT"
echo "authors' code at $(git -C "$OFFICIAL_ROOT/repo" rev-parse HEAD)"

if [ ! -x "$OFFICIAL_ROOT/.venv/bin/python" ]; then
  uv venv --python 3.11 "$OFFICIAL_ROOT/.venv"
  uv pip install --python "$OFFICIAL_ROOT/.venv/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.3.0+cpu" "numpy==1.26.4" "gymnasium==0.29.1" "mujoco==2.3.7" "MinAtar==1.0.15" \
    "imageio==2.34.0" "matplotlib==3.8.4" "pandas==2.2.2" "scipy"
fi

check() {  # file sha256
  echo "$2  $1" | sha256sum -c - >/dev/null 2>&1
}
PDF=$DATA/2410.14606v2.pdf
PDF_SHA=9215cd3ee4310ddb7bc2ccadaaec2055d5e01a79d65ff2b22cefbb5d6ec0aa24
check "$PDF" $PDF_SHA || curl -L -o "$PDF" https://arxiv.org/pdf/2410.14606v2
check "$PDF" $PDF_SHA || { echo "paper PDF sha256 mismatch" >&2; exit 1; }

ETT=$DATA/ETTm2.csv
ETT_SHA=db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd
check "$ETT" $ETT_SHA || curl -L -o "$ETT" https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm2.csv
check "$ETT" $ETT_SHA || { echo "ETTm2.csv sha256 mismatch" >&2; exit 1; }
echo "done: OFFICIAL_ROOT=$OFFICIAL_ROOT, data in $DATA"
