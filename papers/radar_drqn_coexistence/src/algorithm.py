"""The recurrent double-DQN core used by every learner in this folder.

The implementation lives in the shared package as `rl_jax_bench.drqn` (it started as this file). The shared copy is a
superset: it adds default-off options (replay priorities, a Q-filter, LayerNorm, masked actions, n-step targets, an
SMDP discount) that this reproduction never turns on; at the defaults its outputs are those of the original file.
"""
from rl_jax_bench.drqn import *  # noqa: F401,F403
