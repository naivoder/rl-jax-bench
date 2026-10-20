"""Networks (float32, flax.linen). The DQL network is rl_jax_bench.drqn's QNetwork (imported in algorithm.py).

DDPG actor: MLP 256-128 ReLU (paper §VII-A1) -> sigmoid x T0 per dimension (A-20).
DDPG critic: MLP 100-100 ReLU on concat(s, a / T0) at the first layer (A-21; the 1/T0 is a fixed input scale, A-23).
Kernel init: flax default lecun_normal, zero bias (as CleanRL ddpg_continuous_action_jax; the paper is silent).
"""
from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class Actor(nn.Module):
    act_dim: int = 5
    hidden: Sequence[int] = (256, 128)
    T0: float = 2.5
    out_bias: float = 0.0     # final-layer bias init; 0 = flax default (declared). logit(Theta_max / K) = -1.516 makes the
                              # initial policy budget-feasible for up to K tracks (pilot debug amendment, A-20a)

    @nn.compact
    def __call__(self, s):
        x = s.astype(jnp.float32)
        for h in self.hidden:
            x = nn.relu(nn.Dense(h, param_dtype=jnp.float32, dtype=jnp.float32)(x))
        x = nn.Dense(self.act_dim, param_dtype=jnp.float32, dtype=jnp.float32,
                     bias_init=nn.initializers.constant(self.out_bias))(x)
        return nn.sigmoid(x) * self.T0


class Critic(nn.Module):
    hidden: Sequence[int] = (100, 100)
    T0: float = 2.5

    @nn.compact
    def __call__(self, s, a):
        x = jnp.concatenate([s.astype(jnp.float32), a.astype(jnp.float32) / self.T0], -1)
        for h in self.hidden:
            x = nn.relu(nn.Dense(h, param_dtype=jnp.float32, dtype=jnp.float32)(x))
        return nn.Dense(1, param_dtype=jnp.float32, dtype=jnp.float32)(x)[..., 0]
