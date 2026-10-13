#!/bin/sh
# Fetches the authors' code, their published run logs and the gym-0.21 source, and builds the venv used by the
# scripts in this folder. After it, tests/test_eval.py runs (it needs only the logs).
# Run from the paper folder:  sh scripts/official/setup.sh
#   $OFFICIAL_ROOT/repo                               twni2016/pomdp-baselines at e7c19c32 (run unmodified)
#   $OFFICIAL_ROOT/numeric_results/extracted/logs_txt the authors' "numeric results" archive linked from their README
#                                                     (Google Drive, 5.78 GB zip; only its csv/yml/yaml/json files,
#                                                     about 1.1 GB, are kept; the zip is deleted after extraction)
#   $OFFICIAL_ROOT/gym021/gym-0.21.0                  gym 0.21.0 sdist (Pendulum-v1 source for run_official_v1.py)
#   $OFFICIAL_ROOT/.venv                              Python 3.8, torch 1.13.1+cpu, gym 0.18.0, numpy 1.20.1
# Nothing is patched. Requires git, uv, curl, unzip and network access to Google Drive.
# SKIP_LOGS=1 skips the 5.78 GB download; SKIP_VENV=1 skips the venv.
set -e
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
mkdir -p "$OFFICIAL_ROOT"

fetch() {   # fetch URL DIR COMMIT
  if [ ! -d "$2/.git" ]; then git clone --quiet "$1" "$2"; fi
  git -C "$2" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "$2" checkout --quiet "$3"
  echo "$2 at $(git -C "$2" rev-parse HEAD)"
}
fetch https://github.com/twni2016/pomdp-baselines.git "$OFFICIAL_ROOT/repo" e7c19c32a20033d75414b29fbc466c77c211e968

if [ ! -d "$OFFICIAL_ROOT/gym021/gym-0.21.0" ]; then
  mkdir -p "$OFFICIAL_ROOT/gym021"
  curl -sSL -o "$OFFICIAL_ROOT/gym021/gym-0.21.0.tar.gz" \
       https://files.pythonhosted.org/packages/source/g/gym/gym-0.21.0.tar.gz
  tar -xzf "$OFFICIAL_ROOT/gym021/gym-0.21.0.tar.gz" -C "$OFFICIAL_ROOT/gym021"
fi

LOGS="$OFFICIAL_ROOT/numeric_results/extracted/logs_txt"
if [ "${SKIP_LOGS:-0}" != "1" ] && [ ! -d "$LOGS" ]; then
  N="$OFFICIAL_ROOT/numeric_results"
  mkdir -p "$N/unzip"
  uvx --from "gdown==5.2.0" gdown 1syDNw_suIAgS-DxGp0luzepqy1pYB-5Q -O "$N/results.bin"
  echo "7bde5109ee51906bea2366b0fe4e02bfe4d89c7f54e9c0ee3a032af43977e2cb  $N/results.bin" | sha256sum -c -
  unzip -q "$N/results.bin" '*.csv' '*.yml' '*.yaml' '*.json' -d "$N/unzip"
  rm -f "$N/results.bin"
  # the archive's log root is the directory that holds pomdp/ and meta/
  top=$(dirname "$(find "$N/unzip" -type d -name pomdp | awk '{ print length, $0 }' | sort -n | head -1 | cut -d' ' -f2-)")
  mkdir -p "$N/extracted"
  mv "$top" "$LOGS"
  rm -rf "$N/unzip"
fi

if [ "${SKIP_VENV:-0}" != "1" ]; then
  # gym 0.18.0 has metadata that current pip / uv reject: build it with pip 23.3.2 and old setuptools / wheel
  uv venv --seed --python 3.8 "$OFFICIAL_ROOT/.venv"
  PY="$OFFICIAL_ROOT/.venv/bin/python"
  $PY -m pip install -q "pip==23.3.2" "setuptools==65.5.0" "wheel==0.38.4"
  $PY -m pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==1.13.1+cpu"
  printf 'setuptools==65.5.0\nwheel==0.38.4\n' > "$OFFICIAL_ROOT/build_constraints.txt"
  PIP_CONSTRAINT="$OFFICIAL_ROOT/build_constraints.txt" $PY -m pip install -q "gym==0.18.0"
  $PY -m pip install -q "numpy==1.20.1" "pandas==1.4.4" "scipy==1.7.3" "matplotlib==3.5.3" "seaborn==0.13.2" \
     "scikit-learn==1.0.2" "ruamel.yaml==0.17.40" "PyYAML==6.0.3" "future-fstrings==1.2.0" "psutil==7.2.2" \
     "tensorboard==2.11.2" "pycolab==1.2" "protobuf==3.20.3" "Pillow==9.0.1" "cloudpickle==1.6.0"
fi
echo "done; tests/test_eval.py now runs (pytest -q)"
