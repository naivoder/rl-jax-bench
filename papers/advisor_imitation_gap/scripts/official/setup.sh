#!/bin/sh
# Fetch and set up the authors' code for the scripts in scripts/official/ (and for regenerating tests/fixtures/).
#   OFFICIAL_ROOT (default external/official inside this folder) receives:
#     advisor/    allenai/advisor @ 6849755 (2022-01-01); its hp_runs/ holds the authors' per-run TSVs (the paper's data)
#     allenact/   allenai/allenact @ f00445e4 (2021-12-16), the last AllenAct commit before the advisor commit.
#                 v0.5.0 cannot import the advisor MiniGrid/PD configs (create_minigrid_offpolicy_data_iterator).
#     shim/       makes advisor/ importable as `projects.advisor`, as the configs expect
#     .venv/      Python 3.8, torch 1.13.1 CPU and the pinned versions below (built with uv)
#     env.sh      `. $OFFICIAL_ROOT/env.sh` sets PYTHONPATH for the authors' code
# No patches are needed: the authors' code is imported unmodified; our drivers apply their few overrides in-process.
# Needs git, uv and network access; CPU only. Untested as a whole script (it reproduces a hand-built setup).
set -eu
cd "$(dirname "$0")/../.."
ROOT=${OFFICIAL_ROOT:-external/official}
mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd)

fetch() {  # fetch <url> <dir> <commit>
    if [ ! -d "$2/.git" ]; then
        git init -q "$2"
        git -C "$2" remote add origin "$1"
    fi
    git -C "$2" fetch -q --depth 1 origin "$3"
    git -C "$2" checkout -q "$3"
}
fetch https://github.com/allenai/advisor.git "$ROOT/advisor" 6849755042c6dab1488f64cf21bde2322add3cc1
fetch https://github.com/allenai/allenact.git "$ROOT/allenact" f00445e4ae8724ccc001b3300af5c56a7f882614

mkdir -p "$ROOT/shim/projects"
: > "$ROOT/shim/projects/__init__.py"
ln -sfn "$ROOT/advisor" "$ROOT/shim/projects/advisor"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    uv venv -q --python 3.8 "$ROOT/.venv"
fi
uv pip install -q --python "$ROOT/.venv/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch==1.13.1+cpu"
uv pip install -q --python "$ROOT/.venv/bin/python" \
    gym==0.17.3 gym-minigrid==1.0.3 networkx==2.5 numpy==1.23.5 scipy==1.10.1 pandas==2.0.3 patsy==1.0.3 \
    stable-baselines3==1.1.0 opencv-python-headless==4.8.1.78 pickle5==0.0.11 tensorboardx==2.6.2.2 \
    setproctitle==1.3.8 moviepy==1.0.3 matplotlib==3.7.5 canonicaljson==2.0.0 attrs==25.3.0 filelock==3.16.1 \
    tqdm==4.70.1 blosc==1.11.1 pyglet==1.5.0 protobuf==5.29.6 \
    "babyai @ git+https://github.com/Lucaweihs/babyai.git@0b450eeb3a2dc7116c67900d51391986bdbb84cd"

cat > "$ROOT/env.sh" <<EOF
# source me: PYTHONPATH for the authors' code (advisor mounted as projects.advisor through shim/)
export OFFICIAL_ROOT=$ROOT
export PYTHONPATH=$ROOT/shim:$ROOT/allenact:$ROOT/advisor
EOF
echo "authors' code ready in $ROOT; use: . $ROOT/env.sh && $ROOT/.venv/bin/python scripts/official/run_reference.py ..."
