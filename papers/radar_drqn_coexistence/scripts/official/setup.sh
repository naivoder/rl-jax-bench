#!/bin/sh
# Rebuilds the oracle side of the official-code checks under $OFFICIAL_ROOT (default external/official, git-ignored).
# No author code exists for this paper; the oracles are labelled second-best:
#   cleanrl/          CleanRL @ fe8d8a0 (dqn_jax.py run unmodified)          venv .venv-cleanrl (py3.10, CleanRL's lock pins)
#   pomdp-baselines/  pomdp-baselines @ e7c19c3 (Critic_RNN, SACD loss)      venv .venv-pomdp   (py3.8, torch 1.13.1 CPU)
#   .venv-tf22        TF 2.2 tf.keras for the independent Keras learner in scripts/official/keras/ (py3.8)
# The shipped tests do not need any of this (they replay tests/fixtures/*.npz). Use it to regenerate the fixtures,
# the CleanRL CartPole reference logs and the Keras runs by hand (commands in the README, "Checked against").
# Needs git and uv. Not run end to end in this repository; the pins are the ones the original runs used.
set -e
cd "$(dirname "$0")/../.."
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
mkdir -p "$OFFICIAL_ROOT" && cd "$OFFICIAL_ROOT"

clone() {  # url dir commit
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" checkout -q "$3"
}
clone https://github.com/vwxyzjn/cleanrl cleanrl fe8d8a03c41a7ef5b523e2e354bd01c363e786bb
clone https://github.com/twni2016/pomdp-baselines pomdp-baselines e7c19c32a20033d75414b29fbc466c77c211e968

# CleanRL dqn_jax.py venv (jaxlib 0.4.7 is only on the JAX release index)
[ -x .venv-cleanrl/bin/python ] || uv venv -p 3.10 .venv-cleanrl
uv pip install --python .venv-cleanrl/bin/python -f https://storage.googleapis.com/jax-releases/jax_releases.html \
  jax==0.4.8 jaxlib==0.4.7 flax==0.6.8 optax==0.1.4 chex==0.1.5 orbax==0.1.0 gymnasium==0.29.1 numpy==1.24.4 \
  scipy==1.10.1 tyro==0.5.10 tensorboard==2.11.2 \
  torch==2.4.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# pomdp-baselines venv (float64 CPU differentials)
[ -x .venv-pomdp/bin/python ] || uv venv -p 3.8 .venv-pomdp
uv pip install --python .venv-pomdp/bin/python --extra-index-url https://download.pytorch.org/whl/cpu \
  torch==1.13.1+cpu numpy==1.20.1 gym==0.18.0 ruamel.yaml==0.17.40 PyYAML==6.0.3 scipy==1.7.3

# TF 2.2 Keras venv (the paper's framework)
[ -x .venv-tf22/bin/python ] || uv venv -p 3.8 .venv-tf22
uv pip install --python .venv-tf22/bin/python tensorflow-cpu==2.2.0 numpy==1.18.5 scipy==1.4.1 h5py==2.10.0 protobuf==3.20.3

mkdir -p fixtures ref_runs
echo "official oracles ready in $(pwd)"
