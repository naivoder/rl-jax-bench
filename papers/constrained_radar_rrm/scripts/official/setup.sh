#!/bin/sh
# Set up the reference code used as oracles (the authors released no code for this paper).
#   OFFICIAL_ROOT (default external/official inside this folder) receives:
#     cleanrl/        vwxyzjn/cleanrl @ fe8d8a0; tests/test_oracle.py reads ddpg_continuous_action_jax.py from it
#     venv/           Python 3.12: filterpy 1.4.5, scipy 1.18.1, numpy 2.5.4 (the EKF / DARE / SLSQP / ncx2 references)
#     .venv_cleanrl/  Python 3.10: CleanRL's own pinned stack, for the Pendulum reference runs
#     fixtures/       filterpy_ekf.npz, written here by scripts/official/filterpy_ekf_harness.py
#     ref_runs/       logs of CleanRL's DDPG on Pendulum-v1 (only written if you run the loop printed at the end)
# No patches are needed. After this script the two skipped tests in tests/test_oracle.py run.
# Needs git, uv and network access; CPU only. Untested as a whole script (it reproduces a hand-built setup).
set -eu
cd "$(dirname "$0")/../.."
ROOT=${OFFICIAL_ROOT:-external/official}
mkdir -p "$ROOT/fixtures" "$ROOT/ref_runs"
ROOT=$(cd "$ROOT" && pwd)

if [ ! -d "$ROOT/cleanrl/.git" ]; then
    git init -q "$ROOT/cleanrl"
    git -C "$ROOT/cleanrl" remote add origin https://github.com/vwxyzjn/cleanrl.git
fi
git -C "$ROOT/cleanrl" fetch -q --depth 1 origin fe8d8a03c41a7ef5b523e2e354bd01c363e786bb
git -C "$ROOT/cleanrl" checkout -q fe8d8a03c41a7ef5b523e2e354bd01c363e786bb

# filterpy / scipy reference venv, then the filterpy EKF fixture
[ -x "$ROOT/venv/bin/python" ] || uv venv -q --python 3.12 "$ROOT/venv"
uv pip install -q --python "$ROOT/venv/bin/python" filterpy==1.4.5 scipy==1.18.1 numpy==2.5.4 matplotlib==3.11.2
OFFICIAL_ROOT=$ROOT "$ROOT/venv/bin/python" -I scripts/official/filterpy_ekf_harness.py

# CleanRL venv for the Pendulum reference runs (scripts/oracle_pendulum_compare.py reads their logs)
[ -x "$ROOT/.venv_cleanrl/bin/python" ] || uv venv -q --python 3.10 "$ROOT/.venv_cleanrl"
uv pip install -q --python "$ROOT/.venv_cleanrl/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1+cpu"
uv pip install -q --python "$ROOT/.venv_cleanrl/bin/python" \
    jax==0.4.8 jaxlib==0.4.7 flax==0.6.8 optax==0.1.4 chex==0.1.5 numpy==1.24.4 scipy==1.10.1 \
    gymnasium==0.29.1 tyro==0.5.10 tensorboard==2.11.2 protobuf==3.20.3 orbax-checkpoint==0.2.2

cat <<EOF
reference code ready in $ROOT
Pendulum reference runs (CPU, a few minutes each), from $ROOT/ref_runs:
  for S in 1 2 3; do
    PYTHONPATH=../cleanrl ../.venv_cleanrl/bin/python ../cleanrl/cleanrl/ddpg_continuous_action_jax.py \\
      --env-id Pendulum-v1 --total-timesteps 60000 --seed \$S --policy-frequency 1 > cleanrl_pend_pf1_s\$S.log 2>&1
  done
  (also: default policy frequency -> cleanrl_pend_pf2_s\$S.log; add --learning-rate 2e-4 --gamma 0.9 --tau 0.01
   --batch-size 128 -> cleanrl_pend_paperhp_s\$S.log)
Environment-parity fixture: uv run python scripts/oracle_dump_env.py --T 10000 (both tags), then
  OFFICIAL_ROOT=$ROOT $ROOT/venv/bin/python -I scripts/official/oracle_env_harness.py, then
  uv run python scripts/build_parity_fixture.py (rewrites tests/fixtures/official_oracle.npz).
EOF
