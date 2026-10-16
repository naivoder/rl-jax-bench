"""Actor-critic networks and action distributions (paper C2, C13, K1-K3, L1-L5).

Parameter layouts follow the official PyTorch code so that weights can be copied 1:1 into the CleanRL oracle for the
Gate-2 differential tests: Dense kernels are (in, out) (= torch weight.T); the LSTM keeps torch's stacked (4H, in)
matrices, two bias vectors and gate order (i, f, g, o) the README.

* ``SeparateAC``  -- ppo.py / ppo_continuous_action.py: two 2x64 tanh MLPs (paper C13, K4 value_network='copy').
* ``SharedAC``    -- ppo_shared.py: one 2x64 tanh torso, policy head + value head.
* ``LSTMAgent``   -- ppo_atari_lstm.py with the Nature CNN replaced by Linear(64)+tanh.
Init (paper C2): orthogonal, gain sqrt(2) hidden / 0.01 policy head / 1.0 value head, zero biases; LSTM weights
orthogonal(1.0), biases 0 (paper L1). Continuous: state-independent log-std parameter initialised to 0 (paper K2).
"""
from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

SQRT2 = math.sqrt(2.0)  # paper C2 (np.sqrt(2))
LOG_2PI = math.log(2.0 * math.pi)


def dense(features: int, scale: float, name: str | None = None) -> nn.Dense:
    """Linear layer with orthogonal(scale) weights and zero bias (paper C2)."""
    return nn.Dense(features, kernel_init=nn.initializers.orthogonal(scale), bias_init=nn.initializers.zeros,
                    name=name)


class MLP(nn.Module):
    hidden: int = 64  # paper C13 (2 x 64 tanh)

    @nn.compact
    def __call__(self, x):
        x = jnp.tanh(dense(self.hidden, SQRT2, "l0")(x))
        return jnp.tanh(dense(self.hidden, SQRT2, "l1")(x))


class SeparateAC(nn.Module):
    """Separate actor and critic MLPs. Returns (dist_params, value)."""
    n_out: int
    continuous: bool = False
    hidden: int = 64

    @nn.compact
    def __call__(self, x):
        out = dense(self.n_out, 0.01, "actor_head")(MLP(self.hidden, name="actor")(x))
        v = dense(1, 1.0, "critic_head")(MLP(self.hidden, name="critic")(x))[..., 0]
        if self.continuous:
            log_std = self.param("log_std", nn.initializers.zeros, (self.n_out,))  # paper K2
            return (out, jnp.broadcast_to(log_std, out.shape)), v
        return out, v


class SharedAC(nn.Module):
    """One torso feeding a policy head and a value head (ppo_shared.py)."""
    n_out: int
    continuous: bool = False
    hidden: int = 64

    @nn.compact
    def __call__(self, x):
        h = MLP(self.hidden, name="torso")(x)
        out = dense(self.n_out, 0.01, "actor_head")(h)
        v = dense(1, 1.0, "critic_head")(h)[..., 0]
        if self.continuous:
            log_std = self.param("log_std", nn.initializers.zeros, (self.n_out,))
            return (out, jnp.broadcast_to(log_std, out.shape)), v
        return out, v


class TorchLSTMCell(nn.Module):
    """torch.nn.LSTM single-layer cell: gates = W_ih x + b_ih + W_hh h + b_hh, order (i, f, g, o)."""
    hidden: int = 128

    @nn.compact
    def __call__(self, carry, x):
        h, c = carry
        ortho = nn.initializers.orthogonal(1.0)  # paper L1; official code orthogonal_(param, 1.0)
        w_ih = self.param("weight_ih", ortho, (4 * self.hidden, x.shape[-1]))
        w_hh = self.param("weight_hh", ortho, (4 * self.hidden, self.hidden))
        b_ih = self.param("bias_ih", nn.initializers.zeros, (4 * self.hidden,))
        b_hh = self.param("bias_hh", nn.initializers.zeros, (4 * self.hidden,))
        gates = x @ w_ih.T + b_ih + h @ w_hh.T + b_hh
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return (h, c), h


class ResetLSTM(nn.Module):
    """Scan body: (h, c) <- (1 - done_t) * (h, c) BEFORE step t (paper L3; CleanRL get_states), then the cell."""
    hidden: int = 128
    reset: bool = True

    @nn.compact
    def __call__(self, carry, inp):
        x, d = inp
        if self.reset:
            m = (1.0 - d)[..., None]
            carry = (carry[0] * m, carry[1] * m)
        return TorchLSTMCell(self.hidden, name="cell")(carry, x)


class LSTMAgent(nn.Module):
    """obs -> Linear(64)+tanh -> LSTM(128) -> shared policy head (0.01) + value head (1.0).

    Call with x (T, B, obs_dim), dones (T, B), carry ((B, H), (B, H)); returns (logits (T,B,A), value (T,B), carry).
    """
    n_out: int
    hidden: int = 64
    lstm_hidden: int = 128
    reset: bool = True

    @nn.compact
    def __call__(self, x, dones, carry):
        z = jnp.tanh(dense(self.hidden, SQRT2, "encoder")(x))
        scan = nn.scan(ResetLSTM, variable_broadcast="params", split_rngs={"params": False}, in_axes=0, out_axes=0)
        carry, hs = scan(self.lstm_hidden, self.reset, name="lstm")(carry, (z, dones))
        logits = dense(self.n_out, 0.01, "actor_head")(hs)
        v = dense(1, 1.0, "critic_head")(hs)[..., 0]
        return logits, v, carry

    def initial_carry(self, batch: int):
        z = jnp.zeros((batch, self.lstm_hidden), jnp.float32)  # paper L2: zero initial state
        return (z, z)


def build_network(cfg: dict, n_out: int, continuous: bool) -> nn.Module:
    arch = cfg["architecture"]
    if arch == "separate":
        return SeparateAC(n_out, continuous, cfg["hidden"])
    if arch == "shared":
        return SharedAC(n_out, continuous, cfg["hidden"])
    if arch == "lstm":
        assert not continuous, "LSTM agent is discrete only (ppo_atari_lstm.py)"
        return LSTMAgent(n_out, cfg["hidden"], cfg["lstm_hidden"], cfg["lstm_reset"])
    raise ValueError(arch)


# ------------------------------------------------------------------ distributions
def log_prob(dist: Any, action, continuous: bool):
    if continuous:  # paper K3: independent components, log-prob summed over action dims
        mean, log_std = dist
        lp = -((action - mean) ** 2) / (2.0 * jnp.exp(2.0 * log_std)) - log_std - 0.5 * LOG_2PI
        return lp.sum(-1)
    logp_all = jax.nn.log_softmax(dist, axis=-1)
    return jnp.take_along_axis(logp_all, action[..., None].astype(jnp.int32), axis=-1)[..., 0]


def entropy(dist: Any, continuous: bool):
    if continuous:
        _, log_std = dist
        return (0.5 + 0.5 * LOG_2PI + log_std).sum(-1)
    logp_all = jax.nn.log_softmax(dist, axis=-1)
    return -(jnp.exp(logp_all) * logp_all).sum(-1)


def sample(key, dist: Any, continuous: bool):
    if continuous:
        mean, log_std = dist
        return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape, mean.dtype)
    return jax.random.categorical(key, dist, axis=-1)


def mode(dist: Any, continuous: bool):
    """Deterministic action (diagnostic eval only)."""
    return dist[0] if continuous else jnp.argmax(dist, axis=-1)


# ------------------------------------------------------------------ MultiDiscrete (M1) and invalid-action masking (X4)
# No experiment here uses these (E5/E6 need Gym-muRTS: UNREPRODUCIBLE). They exist so that the two
# closed-form properties can be unit-tested (tests/test_dist_closed_form.py).
def multidiscrete_log_prob_entropy(logits, nvec, action):
    """Paper M1: the logits (..., sum(nvec)) are split into one Categorical per action component, and
    prob(a) = prod_i prob(a^i), so the log-prob is the SUM over components. The post states only the product rule;
    summing the entropy over components is from the code (ppo_multidiscrete.py:138-140)."""
    bounds = [0]
    for n in nvec:
        bounds.append(bounds[-1] + int(n))
    lp, ent = 0.0, 0.0
    for i in range(len(nvec)):
        part = logits[..., bounds[i]:bounds[i + 1]]
        lp = lp + log_prob(part, action[..., i], False)
        ent = ent + entropy(part, False)
    return lp, ent


def mask_logits(logits, mask, fill: float = -1e8):
    """Paper X4: the logits of invalid actions are replaced before the softmax. The post says negative infinity; the
    code uses -1e8 (ppo_multidiscrete_mask.py:118, CategoricalMasked), which we follow (it keeps gradients finite)."""
    return jnp.where(mask, logits, jnp.asarray(fill, logits.dtype))


def masked_entropy(logits, mask, fill: float = -1e8):
    """Entropy of the masked Categorical, with the masked terms set to 0 (ppo_multidiscrete_mask.py:121-126)."""
    ml = mask_logits(logits, mask, fill)
    logp = jax.nn.log_softmax(ml, axis=-1)
    return -jnp.where(mask, jnp.exp(logp) * logp, 0.0).sum(-1)
