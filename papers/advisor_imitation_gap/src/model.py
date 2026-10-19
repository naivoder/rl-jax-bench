"""Actor-critic networks with an auxiliary imitation head (paper Fig. 3, App. A.8), in flax.linen.

Every arm uses the SAME network (main actor, aux actor, critic as one Linear(hidden -> 2|A| + 1) head, official
LinearAdvisorActorCritic with ensure_same_init_aux_weights=False); arms that do not use the aux head simply give it
no gradient. This keeps arms identical except for the loss mechanism under test.

* PDNet  (Poisoned Doors, App. A.8 + official RNNActorCriticWithEmbed): Embedding(4 -> 128, N(0,1) as torch default)
  -> LSTM(128) (orthogonal weights, zero biases, official RNNStateEncoder.layer_init) with a TRAINABLE initial state
  0.1 * N(0,1) shared by h and c (official trainable_masked_hidden_state=True) -> Linear(128 -> 2|A|+1) orthogonal, 0 bias.
* LHNet  (2D-Lighthouse, App. A.8 + official LinearAdvisorActorCritic): a single Linear(6400 -> 2|A|+1) on the official
  factorial-design features, evaluated as a sum of the <= 13 active weight rows (exactly X @ W for the 0/1 row X).
* MGNet  (MiniGrid, App. A.8 + official MiniGridSimpleConvRNN): see envs/crossing.py; defined below.
"""
from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

orth = nn.initializers.orthogonal()
zeros = nn.initializers.zeros


class Head(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(2 * self.n_actions + 1, kernel_init=orth, bias_init=zeros, name="head")(x)
        a = self.n_actions
        return out[..., :a], out[..., a:2 * a], out[..., -1]


class LSTMCore(nn.Module):
    """torch.nn.LSTM single layer (gate order i, f, g, o) with orthogonal weights and zero biases.

    TWO bias vectors (b_ih, b_hh) as torch.nn.LSTM has: they receive the same gradient, so under Adam the effective
    bias step is twice that of a single bias and the pair counts twice in the global grad norm. Found by the
    full-update differential test against the official model (tests/test_official_parity.py)."""
    hidden: int

    @nn.compact
    def __call__(self, x, carry):
        h, c = carry
        H = self.hidden
        wi = self.param("w_ih", orth, (x.shape[-1], 4 * H))
        wh = self.param("w_hh", orth, (H, 4 * H))
        b_ih = self.param("b_ih", zeros, (4 * H,))
        b_hh = self.param("b_hh", zeros, (4 * H,))
        z = x @ wi + h @ wh + b_ih + b_hh
        i, f, g, o = jnp.split(z, 4, axis=-1)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return h, (h, c)


def _torch_orth_rows(key, shape, dtype=jnp.float32):
    """Orthogonal init of a flax (in, out) kernel as torch does it on the (out, in) weight."""
    return orth(key, shape[::-1], dtype).T


class PDNet(nn.Module):
    n_actions: int = 7
    n_obs: int = 4
    hidden: int = 128

    def setup(self):
        self.embed = nn.Embed(self.n_obs, self.hidden, embedding_init=nn.initializers.normal(1.0))
        self.core = LSTMCore(self.hidden)
        self.init_state = self.param("init_state", nn.initializers.normal(0.1), (self.hidden,))
        self.head = Head(self.n_actions)

    def initial_carry(self, batch: int):
        z = jnp.zeros((batch, self.hidden))
        return (z, z)

    def __call__(self, obs, carry, start):
        """obs [B] int, carry ((B,H),(B,H)), start [B] bool (episode starts: use the trainable initial state)."""
        h, c = carry
        s = start[:, None]
        h = jnp.where(s, self.init_state[None], h)
        c = jnp.where(s, self.init_state[None], c)
        x = self.embed(obs)
        y, carry = self.core(x, (h, c))
        logits, aux, v = self.head(y)
        return logits, aux, v, carry


class LHNet(nn.Module):
    """Linear(6400 -> 2|A|+1) on the factorial-design row given as its active-column list [B, 13] (-1 = none)."""
    n_actions: int = 4
    feat_dim: int = 6400

    @nn.compact
    def __call__(self, cols, carry, start):
        out_dim = 2 * self.n_actions + 1
        W = self.param("kernel", _torch_orth_rows, (self.feat_dim, out_dim))
        b = self.param("bias", zeros, (out_dim,))
        g = W[jnp.clip(cols, 0, self.feat_dim - 1)] * (cols >= 0)[..., None]
        out = g.sum(-2) + b
        a = self.n_actions
        return out[..., :a], out[..., a:2 * a], out[..., -1], carry

    def initial_carry(self, batch: int):
        return jnp.zeros((batch, 1))


class MGNet(nn.Module):
    """official MiniGridSimpleConvRNN: 3 embedding tables (object/colour/state, dim 8) -> 7x7x24 flattened ->
    LSTM(128) (trainable initial state) -> Linear(128 -> 2|A|+1). (The official class has no conv despite its name.)"""
    n_actions: int = 4
    num_objects: int = 11
    num_colors: int = 6
    num_states: int = 3
    embed: int = 8
    hidden: int = 128

    def setup(self):
        ini = nn.initializers.normal(1.0)
        self.eo = nn.Embed(self.num_objects, self.embed, embedding_init=ini)
        self.ec = nn.Embed(self.num_colors, self.embed, embedding_init=ini)
        self.es = nn.Embed(self.num_states, self.embed, embedding_init=ini)
        self.core = LSTMCore(self.hidden)
        self.init_state = self.param("init_state", nn.initializers.normal(0.1), (self.hidden,))
        self.head = Head(self.n_actions)

    def initial_carry(self, batch: int):
        z = jnp.zeros((batch, self.hidden))
        return (z, z)

    def __call__(self, img, carry, start):
        """img [B, 7, 7, 3] int (object, colour, state)."""
        h, c = carry
        s = start[:, None]
        h = jnp.where(s, self.init_state[None], h)
        c = jnp.where(s, self.init_state[None], c)
        img = img.astype(jnp.int32)
        # one-hot @ table instead of gather: identical values/gradients, avoids a slow scatter-add backward on GPU
        emb = lambda e, n, idx: jax.nn.one_hot(idx, n, dtype=e.embedding.dtype) @ e.embedding
        x = jnp.concatenate([emb(self.eo, self.num_objects, img[..., 0]), emb(self.ec, self.num_colors, img[..., 1]),
                             emb(self.es, self.num_states, img[..., 2])], -1)
        x = x.reshape(x.shape[0], -1)
        y, carry = self.core(x, (h, c))
        logits, aux, v = self.head(y)
        return logits, aux, v, carry
