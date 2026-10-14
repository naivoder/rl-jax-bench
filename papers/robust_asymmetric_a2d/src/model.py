"""Networks (flax.linen). Architecture and init follow the official code, which produced the figures:

* flat (compact) inputs: MLP 2 x 64 tanh, orthogonal init gain sqrt(2), zero biases, output layer orthogonal with
  gain `out_gain` (0.01 for flat policy/value nets, 1.0 for image heads, sqrt(2) for Q nets) — official
  `initialize_weights_scalable` (the `--initialization fixed` flag of the Q-args is never read by the code).
* image inputs: PixelEncoder = 4 x conv3x3 (32 filters, strides 2,1,1,1, VALID, ReLU) -> Linear 64 -> LayerNorm ->
  tanh, PyTorch-default init (kaiming_uniform(a=sqrt5) == U(+-1/sqrt(fan_in)) for weights and biases), then the
  MLP head (paper App. D.1 says 2 conv / 50 features; the official code differs). Policy and value nets have
  separate encoders (App. D.1).
Policy nets return normalized log-probabilities over the 4 actions; value nets return a scalar per input.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import flax.linen as nn

SQRT2 = math.sqrt(2.0)


def torch_uniform(fan_in: int):
    bound = 1.0 / math.sqrt(fan_in)

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init


class MLP(nn.Module):
    out_dim: int
    out_gain: float
    hidden: int = 64
    depth: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(SQRT2),
                                  bias_init=nn.initializers.zeros)(x))
        return nn.Dense(self.out_dim, kernel_init=nn.initializers.orthogonal(self.out_gain),
                        bias_init=nn.initializers.zeros)(x)


class PixelEncoder(nn.Module):
    feature_dim: int = 64
    filters: int = 32

    @nn.compact
    def __call__(self, x):                       # x: (..., 42, 42, 3) in [0, 1]
        cin = x.shape[-1]
        for i, s in enumerate([2, 1, 1, 1]):
            fan = 9 * (cin if i == 0 else self.filters)
            x = nn.relu(nn.Conv(self.filters, (3, 3), strides=(s, s), padding='VALID',
                                kernel_init=torch_uniform(fan), bias_init=torch_uniform(fan))(x))
        x = x.reshape(x.shape[:-3] + (-1,))
        fan = x.shape[-1]
        x = nn.Dense(self.feature_dim, kernel_init=torch_uniform(fan), bias_init=torch_uniform(fan))(x)
        x = nn.LayerNorm(epsilon=1e-5)(x)
        return jnp.tanh(x)


class PolicyNet(nn.Module):
    kind: str = 'flat'          # 'flat' | 'image'
    out_gain: float = 0.01
    n_act: int = 4

    def setup(self):
        if self.kind == 'image':
            self.encoder = PixelEncoder()
        self.head = MLP(self.n_act, self.out_gain)

    def __call__(self, x):
        h = self.encoder(x) if self.kind == 'image' else x
        return jax.nn.log_softmax(self.head(h), axis=-1)


class ValueNet(nn.Module):
    kind: str = 'flat'
    out_gain: float = 0.01

    def setup(self):
        if self.kind == 'image':
            self.encoder = PixelEncoder()
        self.head = MLP(1, self.out_gain)

    def __call__(self, x):
        h = self.encoder(x) if self.kind == 'image' else x
        return self.head(h)[..., 0]


def make_policy(kind: str) -> PolicyNet:
    """Official gains: flat policy 0.01, image policy head 1.0."""
    return PolicyNet(kind=kind, out_gain=0.01 if kind == 'flat' else 1.0)


def make_value(kind: str) -> ValueNet:
    return ValueNet(kind=kind, out_gain=0.01 if kind == 'flat' else 1.0)


def make_q() -> ValueNet:
    """Q(x, a) on [x | one_hot(a)], slim_init=False -> output gain sqrt(2)."""
    return ValueNet(kind='flat', out_gain=SQRT2)
