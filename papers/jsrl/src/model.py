# PROVENANCE: copied from our earlier DQfD reproduction (src/model.py; our own prior code, not the authors'); JSRL changes are marked 'JSRL:'.
"""Networks. Dueling Q network (paper §Experimental Setup: 'dueling state-advantage convolutional network',
Wang et al. 2016) scaled to MinAtar, and the PPO actor-critic used only as the DEMONSTRATOR.

Parameters are plain flax pytrees; all arms use the identical architecture (arms differ only in the mechanism under test).
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class Conv3x3(nn.Module):
    """3x3, stride 1, SAME conv written as nine shifted slices + ONE matmul (im2col).  Numerically the same function as
    nn.Conv(features, (3,3), padding='SAME') with the same parameter layout (kernel (3,3,C,F), bias (F,)), but under
    jax.vmap over independent per-run parameters it lowers to a batched matmul instead of a (slow) grouped convolution."""

    features: int

    @nn.compact
    def __call__(self, x):  # (B, H, W, C)
        C = x.shape[-1]
        k = self.param("kernel", nn.initializers.lecun_normal(), (3, 3, C, self.features))
        b = self.param("bias", nn.initializers.zeros, (self.features,))
        H, W = x.shape[1], x.shape[2]
        xp = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        cols = jnp.concatenate([xp[:, i:i + H, j:j + W, :] for i in range(3) for j in range(3)], axis=-1)  # (B,H,W,9C)
        return cols @ k.reshape(9 * C, self.features) + b


class DuelingQNet(nn.Module):
    """Conv(16, 3x3, s1, SAME)+ReLU -> flatten -> two streams Dense(h)+ReLU -> V (1), A (|A|);
    Q = V + A - mean_a A   (Wang et al. 2016 mean aggregator).  [ASSUMED; h = 64 per stream so the two
    streams together hold MinAtar's usual 128 hidden units]."""

    n_actions: int
    channels: int = 16
    stream_hidden: int = 64

    @nn.compact
    def __call__(self, x):  # x: (B, 10, 10, C) float32
        x = nn.relu(Conv3x3(self.channels, name="conv")(x))
        x = x.reshape((x.shape[0], -1))
        v = nn.relu(nn.Dense(self.stream_hidden, name="v_hidden")(x))
        v = nn.Dense(1, name="v_out")(v)
        a = nn.relu(nn.Dense(self.stream_hidden, name="a_hidden")(x))
        a = nn.Dense(self.n_actions, name="a_out")(a)
        return v + a - a.mean(axis=-1, keepdims=True)


class ActorCritic(nn.Module):
    """PPO demonstrator network (purejaxrl-style MinAtar actor-critic with a conv torso)."""

    n_actions: int
    channels: int = 16
    hidden: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(self.channels, (3, 3), padding="SAME")(x))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(self.hidden)(x))
        logits = nn.Dense(self.n_actions, kernel_init=nn.initializers.orthogonal(0.01))(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return logits, value[:, 0]


def l2_half_sumsq(params):
    """0.5 * sum(theta^2) over ALL weights and biases."""
    import jax
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
