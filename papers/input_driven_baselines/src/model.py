"""Policy and value networks (flax.linen).

Paper App. K: actor and critic are MLPs with hidden sizes 64/32 and ReLU for the discrete
(A2C) tasks; 128/64 ReLU for MuJoCo. Init: Glorot-uniform weights, zero biases (official
nn_ops.glorot). float32 throughout.
"""
from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

_ACT = {"relu": nn.relu, "tanh": nn.tanh, "leaky_relu": lambda x: nn.leaky_relu(x, 0.2)}


class MLP(nn.Module):
    hidden: Sequence[int]
    out: int
    activation: str = "relu"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        act = _ACT[self.activation]
        init = nn.initializers.glorot_uniform()
        for h in self.hidden:
            x = act(nn.Dense(h, kernel_init=init, bias_init=nn.initializers.zeros, use_bias=self.use_bias)(x))
        return nn.Dense(self.out, kernel_init=init, bias_init=nn.initializers.zeros, use_bias=self.use_bias)(x)


class Actor(nn.Module):
    """Categorical policy: returns logits [..., n_actions]."""

    hidden: Sequence[int]
    n_actions: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        return MLP(self.hidden, self.n_actions, self.activation)(obs)


class Critic(nn.Module):
    """State (or input-adapted) value: returns a scalar per input row."""

    hidden: Sequence[int]
    activation: str = "relu"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        return MLP(self.hidden, 1, self.activation, self.use_bias)(x)[..., 0]


def init_actor(key, hidden, n_actions, obs_dim, activation="relu"):
    m = Actor(tuple(hidden), n_actions, activation)
    return m, m.init(key, jnp.zeros((obs_dim,), jnp.float32))


def init_critic(key, hidden, in_dim, activation="relu"):
    m = Critic(tuple(hidden), activation)
    return m, m.init(key, jnp.zeros((in_dim,), jnp.float32))


def init_critic_stack(key, hidden, in_dim, n, activation="relu"):
    """N independent value nets with parameters stacked on a leading axis (multi-value arm)."""
    m = Critic(tuple(hidden), activation)
    params = jax.vmap(lambda k: m.init(k, jnp.zeros((in_dim,), jnp.float32)))(jax.random.split(key, n))
    return m, params


def n_params(params) -> int:
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
