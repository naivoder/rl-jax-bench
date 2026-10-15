"""Networks. The paper adds no architecture: PPO uses Schulman et al. (2017)'s separate 2x64 tanh actor and critic
with a state-independent log-std (paper §1 'OpenAI Baselines ... hyperparameters reported by Schulman et al.').
These are the vendored PPO-details modules (src/ppo8/model.py, differential-tested against CleanRL / baselines); this
file only adds the PAIRED initialisation used by the time-awareness arms.

Paired init: every arm's network is built with obs_dim + 1 inputs; the last input is the time feature tau. For the
time-unaware arms tau is held at 0, so the first-layer weights of that column receive exactly zero gradient (and,
with Adam, exactly zero updates) and the network is the same function as one without the column. All arms of a seed
therefore start from IDENTICAL parameters and differ only in the mechanism under test (tests/test_smoke.py checks
the equivalence and the zero updates). Tabular learners (E1, E2, E9) have no network.
"""
from __future__ import annotations

import jax.numpy as jnp

from .ppo8 import model as _m

SeparateAC = _m.SeparateAC
log_prob = _m.log_prob
entropy = _m.entropy
sample = _m.sample
mode = _m.mode


def build(n_out: int, continuous: bool, hidden: int = 64):
    """Separate actor/critic, 2 x `hidden` tanh, orthogonal init (gains sqrt2 / 0.01 / 1), log-std init 0."""
    return SeparateAC(n_out, continuous, hidden)


def net_input(obs_normed, tau):
    """Concatenate the (normalized) observation and the time feature (paper §2: 'the time remaining ... as part of
    the agent's input'). tau = 0 for time-unaware arms (see module docstring)."""
    return jnp.concatenate([obs_normed, jnp.asarray(tau, obs_normed.dtype)[..., None]], axis=-1)
