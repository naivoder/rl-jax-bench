#!/bin/sh
# CleanRL's headline experiment with its own code: dqn_jax.py unmodified at its defaults on CartPole-v1, seeds 1-10, CPU.
# The original command line was not recorded; this assumes the defaults plus --seed and --env-id.
# Logs go to $OFFICIAL_ROOT/ref_runs/, read by scripts/official_ref_compare.py. Sequential (about 3.5 min per seed).
set -e
cd "$(dirname "$0")/../.."
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
cd "$OFFICIAL_ROOT"
for s in 1 2 3 4 5 6 7 8 9 10; do
  (cd cleanrl && JAX_PLATFORMS=cpu ../.venv-cleanrl/bin/python cleanrl/dqn_jax.py --seed $s --env-id CartPole-v1) \
      > ref_runs/cleanrl_dqn_jax_cartpole_s$s.log 2>&1
done
