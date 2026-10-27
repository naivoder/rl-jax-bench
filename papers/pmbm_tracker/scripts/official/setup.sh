#!/bin/sh
# Builds everything the differential checks against the authors' code need, under $OFFICIAL_ROOT
# (default: external/official inside the paper folder, git-ignored):
#   MTT/                 the authors' MATLAB code, github.com/Agarciafernandez/MTT at the pinned commit
#   work/                Octave work copy: "PMBM filter", "GOSPA code", Assignment (re-encoded Latin-1 -> UTF-8,
#                        patches/ applied), williams2015_anc/ (arXiv 1203.2995 ancillary files), shims/
#   octave_env/          GNU Octave 10.3.0 from conda-forge (needs conda or mamba on PATH)
#   stonesoup-venv/      optional: python with stonesoup==1.9.1, for a live re-run of tests/test_eval.py's cross-check
# Afterwards: scripts/official/run_octave.sh runs Octave with this work copy; see the README "Checked against".
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$HERE/../../external/official}
COMMIT=cc5c6df0e9343dc1345d91ed6ab95f01db4d1821
mkdir -p "$OFFICIAL_ROOT"; cd "$OFFICIAL_ROOT"

if [ ! -d MTT ]; then git clone https://github.com/Agarciafernandez/MTT.git MTT; fi
git -C MTT checkout -q $COMMIT

rm -rf work; mkdir -p work/williams2015_anc work/shims
cp -r "MTT/PMBM filter" "MTT/GOSPA code" MTT/Assignment work/
find work -name '*.m' | while IFS= read -r f; do iconv -f latin1 -t utf-8 "$f" > "$f.tmp" && mv "$f.tmp" "$f"; done
for p in "$HERE"/patches/*.patch; do patch -s -p1 -d work < "$p"; done
cp "$HERE"/shims/*.m work/shims/
for f in gentruth lbp main momb predict tomb update; do
  curl -fsSL -o "work/williams2015_anc/$f.m" "https://arxiv.org/src/1203.2995/anc/$f.m"
done

if [ ! -x octave_env/bin/octave-cli ]; then
  CONDA=$(command -v mamba || command -v conda || true)
  if [ -n "$CONDA" ]; then
    "$CONDA" create -y -p "$OFFICIAL_ROOT/octave_env" -c conda-forge --override-channels octave=10.3.0
  else
    echo "conda/mamba not found: install GNU Octave 10.3 into $OFFICIAL_ROOT/octave_env yourself" >&2
  fi
fi

if [ "${WITH_STONESOUP:-0}" = 1 ] && [ ! -x stonesoup-venv/bin/python ]; then
  python3 -m venv stonesoup-venv && stonesoup-venv/bin/pip install -q stonesoup==1.9.1
  echo "export STONESOUP_PYTHON=$OFFICIAL_ROOT/stonesoup-venv/bin/python"
fi
echo "official-code work copy ready in $OFFICIAL_ROOT/work"
