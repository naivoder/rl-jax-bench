"""Networks for the loss-of-plasticity reproduction (flax.linen).

* ``MLP`` -- fully connected net, hidden activation chosen by an integer id (so activation arms can be vmapped);
  returns (output, [hidden activations per layer]).  Used for the Slowly-Changing-Regression learner
  (20 -> 5 -> 1, paper Table B.3) and the Online-Permuted-MNIST net (784 -> w -> w -> w -> 10, paper §3).
* ``init_mlp`` -- Kaiming-uniform init, U(-b, b) with b = gain * sqrt(3 / fan_in), gain(act) for hidden layers and
  gain 1 for the output layer, all biases 0 (paper App. B; official code nets/ffnn.py, nets/deep_ffnn.py).
* the Slowly-Changing-Regression *target* network (LTU net) lives in data.py because it is part of the data stream.

Parameters are the flax dict ``{"params": {"layer_0": {"kernel": (in, out), "bias": (out,)}, ...}}``; the CBP
surgery in algorithm.py indexes kernels as (fan_in, units) -- the transpose of PyTorch's (out, in).
"""
from __future__ import annotations

import math
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

# Activation ids (vmappable integer).  Order is fixed; configs refer to names.
ACT_NAMES = ("relu", "tanh", "sigmoid", "elu", "leaky_relu", "swish", "identity")
ACT_ID = {n: i for i, n in enumerate(ACT_NAMES)}
LEAKY_SLOPE = 0.01  # torch nn.LeakyReLU default (official code) -- 

# Init gain used by the official nets (torch kaiming_uniform_ with a=0): leaky_relu -> sqrt(2/(1+0^2)) = sqrt(2);
# swish/elu use the relu gain (nets/ffnn.py).  paper App. B lists the same gains.
INIT_GAIN = {"relu": math.sqrt(2.0), "tanh": 5.0 / 3.0, "sigmoid": 1.0, "elu": math.sqrt(2.0),
             "leaky_relu": math.sqrt(2.0), "swish": math.sqrt(2.0), "identity": 1.0}
# Re-initialisation bound gain used by CBP (official gnt.py compute_bounds: torch calculate_gain(act) with the
# default leaky slope 0.01 -> sqrt(2/(1+0.01^2))).  Differs from INIT_GAIN only for leaky_relu (4th decimal).
REINIT_GAIN = dict(INIT_GAIN, leaky_relu=math.sqrt(2.0 / (1.0 + LEAKY_SLOPE ** 2)))


def _acts():
    return (jax.nn.relu, jnp.tanh, jax.nn.sigmoid, jax.nn.elu,
            lambda z: jnp.where(z >= 0, z, LEAKY_SLOPE * z), jax.nn.silu, lambda z: z)


def activation(act_id, z):
    """Apply activation ``act_id`` (python int -> direct call; traced int -> lax.switch, vmappable)."""
    fns = _acts()
    if isinstance(act_id, int):
        return fns[act_id](z)
    return jax.lax.switch(act_id, fns, z)


class MLP(nn.Module):
    """x (B, n_in) -> (out (B, n_out), [h_l (B, n_l) for each hidden layer])."""
    widths: Sequence[int]
    n_out: int

    @nn.compact
    def __call__(self, x, act_id):
        hs = []
        for i, w in enumerate(self.widths):
            x = nn.Dense(w, name=f"layer_{i}", kernel_init=_unit_uniform, bias_init=nn.initializers.zeros)(x)
            x = activation(act_id, x)
            hs.append(x)
        out = nn.Dense(self.n_out, name=f"layer_{len(self.widths)}", kernel_init=_unit_uniform,
                       bias_init=nn.initializers.zeros)(x)
        return out, hs


def _unit_uniform(key, shape, dtype=jnp.float32):
    """Kaiming-uniform with gain 1: U(-sqrt(3/fan_in), sqrt(3/fan_in)); fan_in = shape[0] (flax kernel (in, out))."""
    b = math.sqrt(3.0 / shape[0])
    return jax.random.uniform(key, shape, dtype, -b, b)


def init_mlp(key, n_in: int, widths: Sequence[int], n_out: int, init_gain):
    """Kaiming-uniform params; hidden kernels scaled by ``init_gain`` (may be a traced scalar for vmapped arms)."""
    model = MLP(tuple(widths), n_out)
    params = model.init(key, jnp.zeros((1, n_in), jnp.float32), 0)
    p = dict(params["params"])
    for i in range(len(widths)):
        p[f"layer_{i}"] = {"kernel": p[f"layer_{i}"]["kernel"] * init_gain, "bias": p[f"layer_{i}"]["bias"]}
    return {"params": p}


def make_model(widths: Sequence[int], n_out: int) -> MLP:
    return MLP(tuple(widths), n_out)


def layer_names(n_hidden: int):
    return [f"layer_{i}" for i in range(n_hidden + 1)]


def reinit_bounds(n_in: int, widths: Sequence[int], reinit_gain):
    """Bound of U(-b, b) for re-initialised incoming weights of each hidden layer (gnt.py compute_bounds)."""
    fans = [n_in] + list(widths[:-1])
    return [reinit_gain * math.sqrt(3.0 / f) for f in fans]
