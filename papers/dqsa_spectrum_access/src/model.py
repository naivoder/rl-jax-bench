"""DQSA network (paper §IV-A, Fig. 2): LSTM(100) -> dueling heads (advantage 10 units, value 10 units), mean-subtracted
aggregation (Eq. 10).  The LSTM cell is rl_jax_bench.drqn's LSTMCell (Keras convention: glorot input,
orthogonal recurrent, unit forget bias == the Matlab-toolbox defaults of); drqn.py's QNetwork has no dueling head, so the
dueling network lives here and plugs into drqn.sequence_loss / drqn.update through their `unroll_fn` hook (drqn.py is NOT edited:
it is the shared, tested core; this keeps its parity tests green by construction).

Eq. 11 action selection and the (alpha, beta) schedules live here too (they are part of the paper's network/policy definition).
"""
from __future__ import annotations

import sys

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from rl_jax_bench.drqn import LSTMCell  # noqa: E402  (shared recurrent core)


class DQSANet(nn.Module):
    n_actions: int                 # K'+1 (index 0 = silent)
    lstm: int = 100                # paper §VI
    hid: int = 10                  # paper §VI (advantage layer 10 units, value layer 10 units)
    dueling: bool = True           # False = plain linear Q head (ablation only)

    def setup(self):
        ki = nn.initializers.glorot_uniform()                    # ASSUMED
        self.cell = LSTMCell(self.lstm, "glorot_uniform", "sigmoid", name="lstm")
        if self.dueling:
            self.a_hid = nn.Dense(self.hid, kernel_init=ki, name="adv_hidden")
            self.a_out = nn.Dense(self.n_actions, kernel_init=ki, name="adv_out")
            self.v_hid = nn.Dense(self.hid, kernel_init=ki, name="val_hidden")
            self.v_out = nn.Dense(1, kernel_init=ki, name="val_out")
        else:
            self.q_out = nn.Dense(self.n_actions, kernel_init=ki, name="q")

    def proj(self, x):
        return self.cell.input_proj(x)

    def recur(self, carry, xp):
        return self.cell.recur(carry, xp)

    def value(self, h):
        """V(h) of the dueling head (= mean_a Q(h,a) by Eq. 10; used by the identity test)."""
        return self.v_out(nn.relu(self.v_hid(h)))[..., 0]

    def heads(self, h):
        if not self.dueling:
            return self.q_out(h)
        A = self.a_out(nn.relu(self.a_hid(h)))                    # ASSUMED ReLU on the 10-unit layers
        V = self.v_out(nn.relu(self.v_hid(h)))
        return V + A - A.mean(-1, keepdims=True)                  # Eq. 10 (average variant),

    def __call__(self, x, carry):
        carry, h = self.cell(carry, x)
        return self.heads(h), carry


def make_net(n_actions: int, lstm: int = 100, hid: int = 10, dueling: bool = True) -> DQSANet:
    return DQSANet(n_actions, lstm, hid, dueling)


def zero_carry(net: DQSANet, batch_shape: tuple, dtype=jnp.float32):
    z = jnp.zeros(batch_shape + (net.lstm,), dtype)
    return (z, z)


def init_params(net: DQSANet, key, obs_dim: int, dtype=jnp.float32):
    return net.init(key, jnp.zeros((1, obs_dim), dtype), zero_carry(net, (1,), dtype))


def unroll(net: DQSANet, params, inputs, carry0, resets=None):
    """Q over sequences: inputs [B,T,D] -> (q [B,T,A], carry_T).  Signature = drqn `unroll_fn` hook."""
    xp = jnp.swapaxes(net.apply(params, inputs, method=DQSANet.proj), 0, 1)          # [T,B,4H]

    def body(carry, xpt):
        carry, h = net.apply(params, carry, xpt, method=DQSANet.recur)
        return carry, h

    carry_T, hs = jax.lax.scan(body, carry0, xp, unroll=4)
    return net.apply(params, jnp.swapaxes(hs, 0, 1), method=DQSANet.heads), carry_T


def make_unroll_fn(net: DQSANet):
    """Closure with the (params, inputs, carry0, resets) signature expected by drqn.sequence_loss; zero carry0 if None."""
    def fn(params, inputs, carry0, resets=None):
        if carry0 is None or carry0 == ():
            carry0 = zero_carry(net, inputs.shape[:1], inputs.dtype)
        return unroll(net, params, inputs, carry0, resets)
    return fn


def act_step(net: DQSANet, params, x, carry):
    """One acting step for any leading batch shape: (q [...,A], carry')."""
    return net.apply(params, x, carry)


# ---------------------------------------------------------------- Eq. 11 policy and schedules
def eq11_probs(q, alpha, beta):
    """pi(a|h) = (1-alpha) softmax(beta Q)_a + alpha/(K'+1)  (paper Eq. 11, README)."""
    return (1.0 - alpha) * jax.nn.softmax(beta * q, axis=-1) + alpha / q.shape[-1]


def sample_eq11(key, q, alpha, beta):
    """Sample a ~ Eq. 11 (independent draw per leading index)."""
    p = eq11_probs(q, alpha, beta)
    return jax.random.categorical(key, jnp.log(jnp.maximum(p, 1e-30)), axis=-1).astype(jnp.int32)


def greedy_action(q):
    """argmax, ties to the lowest index (the argmax ablation; identical users then act identically)."""
    return jnp.argmax(q, axis=-1).astype(jnp.int32)


def schedule(i, n_iters, a0=0.05, a1=0.0, b0=1.0, b1=20.0):
    """alpha: 0.05 -> 0 and beta: 1 -> 20, linear over the R iterations (paper §VI numbers; shape ASSUMED)."""
    f = jnp.clip(i / jnp.maximum(n_iters - 1, 1), 0.0, 1.0)
    return a0 + (a1 - a0) * f, b0 + (b1 - b0) * f
