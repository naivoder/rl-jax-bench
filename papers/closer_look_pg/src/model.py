"""Policy and value networks (flax.linen) plus the diagonal-Gaussian policy algebra.

Architecture: tanh MLP [64, 64] for both nets (paper Table 1; tanh from official code models.py).
Policy: mean head + state-independent log-std parameter initialised at 0 (official code CtsPolicy).
Init: PPO orthogonal (gain sqrt2 hidden, 0.01 policy head, 1.0 value head, zero biases);
TRPO xavier-uniform on every weight matrix, zero biases (official code configs, models.initialize_weights).

All instrument code works on *flat* policy parameter vectors (jax.flatten_util.ravel_pytree), so
"the policy gradient" is one vector over every policy parameter including log-std .
"""
from __future__ import annotations

import math
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

LOG2PI = math.log(2.0 * math.pi)


def _kernel_init(init: str, gain: float):
    if init == "orthogonal":
        return nn.initializers.orthogonal(scale=gain)
    if init == "xavier":
        return nn.initializers.xavier_uniform()
    raise ValueError(init)


class MLP(nn.Module):
    hidden: Sequence[int]
    out: int
    init_kind: str
    head_gain: float

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.tanh(nn.Dense(h, kernel_init=_kernel_init(self.init_kind, math.sqrt(2.0)),
                                 bias_init=nn.initializers.zeros)(x))
        return nn.Dense(self.out, kernel_init=_kernel_init(self.init_kind, self.head_gain),
                        bias_init=nn.initializers.zeros)(x)


class Policy(nn.Module):
    act_dim: int
    hidden: Sequence[int] = (64, 64)
    init_kind: str = "orthogonal"
    log_std_init: float = 0.0

    @nn.compact
    def __call__(self, obs):
        mean = MLP(self.hidden, self.act_dim, self.init_kind, 0.01, name="mean")(obs)
        log_std = self.param("log_std", nn.initializers.constant(self.log_std_init), (self.act_dim,))
        return mean, log_std


class Value(nn.Module):
    hidden: Sequence[int] = (64, 64)
    init_kind: str = "orthogonal"

    @nn.compact
    def __call__(self, obs):
        return MLP(self.hidden, 1, self.init_kind, 1.0, name="v")(obs)[..., 0]


def gauss_logp(mean, log_std, act):
    """log N(act; mean, diag exp(log_std)^2), summed over action dims (official get_loglikelihood)."""
    z = (act - mean) * jnp.exp(-log_std)
    return -0.5 * jnp.sum(z * z, -1) - 0.5 * LOG2PI * act.shape[-1] - jnp.sum(log_std, -1)


def gauss_kl(mean_p, log_std_p, mean_q, log_std_q):
    """KL(p || q) per state for diagonal Gaussians (official CtsPolicy.calc_kl)."""
    var_p, var_q = jnp.exp(2 * log_std_p), jnp.exp(2 * log_std_q)
    d = mean_p.shape[-1]
    return 0.5 * (jnp.sum(2 * log_std_q - 2 * log_std_p, -1) - d + jnp.sum(var_p / var_q, -1)
                  + jnp.sum((mean_q - mean_p) ** 2 / var_q, -1))


class Nets:
    """Bundles the two modules with flat-vector helpers for one (env, alg) configuration."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.pi = Policy(act_dim, tuple(cfg["policy_hidden"]), cfg["init"], cfg["log_std_init"])
        self.vf = Value(tuple(cfg["value_hidden"]), cfg["init"])
        dummy = jnp.zeros((1, obs_dim))
        p0 = self.pi.init(jax.random.PRNGKey(0), dummy)
        v0 = self.vf.init(jax.random.PRNGKey(0), dummy)
        self.pi_flat0, self.pi_unravel = ravel_pytree(p0)
        self.v_flat0, self.v_unravel = ravel_pytree(v0)
        self.n_pi, self.n_v = self.pi_flat0.size, self.v_flat0.size

    def init(self, key):
        kp, kv = jax.random.split(key)
        dummy = jnp.zeros((1, self.obs_dim))
        return (ravel_pytree(self.pi.init(kp, dummy))[0], ravel_pytree(self.vf.init(kv, dummy))[0])

    def pi_apply(self, flat, obs):
        return self.pi.apply(self.pi_unravel(flat), obs)

    def v_apply(self, flat, obs):
        return self.vf.apply(self.v_unravel(flat), obs)

    def logp(self, flat, obs, act):
        mean, log_std = self.pi_apply(flat, obs)
        return gauss_logp(mean, log_std, act)
