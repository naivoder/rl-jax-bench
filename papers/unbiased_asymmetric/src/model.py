"""Networks of Baisero & Amato (2022), App. D, as flax.linen modules (JAX re-implementation of asym-rlpo@1a065b0
``models_flat.py`` / ``models_extra_carflag.py`` semantics; no code copied).

* History encoder: GRU (PyTorch gate equations and init) over inputs ``[phi(a_{t-1}), phi(o_t)]`` with
  ``phi(a_{-1}) = 0``. Categorical envs: 64-d N(0,1) embeddings for actions/observations; the extra reset observation
  (index ``n_obs``) embeds to a fixed zero vector (torch ``padding_idx=-1``). Vector envs: one-hot actions, identity
  observations.
* React-k: the same GRU, run from a zero hidden state over the last k inputs, zero-padded on the left (official
  ``compute_truncated_history_features``), for actor AND critic.
* Heads: MLP 512 -> 256 -> out, ReLU, Xavier-normal (gain sqrt 2 hidden / 1 output), zero bias. Policy ends in
  log-softmax.
* Critic modes: ``h`` (A2C: own encoder), ``s`` (A2C-asym-s: state features only), ``hs`` (A2C-asym-hs: own encoder
  features concatenated with state features). Actor and critic never share parameters (App. D).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class NetSpec:
    kind: str            # "categorical" | "vector"
    n_actions: int
    n_obs: int           # categorical: number of real observations (reset obs = n_obs) ; vector: obs dim
    n_states: int        # categorical: number of states ; vector: state dim
    emb: int = 64        # App. D (categorical embeddings)
    hidden: int = 128    # App. D GRU size (gridverse would be 64)
    head: tuple = (512, 256)  # App. D


def xavier(gain: float):
    # torch.nn.init.xavier_normal_: std = gain * sqrt(2 / (fan_in + fan_out)), untruncated normal
    return nn.initializers.variance_scaling(gain * gain, "fan_avg", "normal")


class MLP(nn.Module):
    hidden: tuple
    out: int
    log_softmax: bool = False

    @nn.compact
    def __call__(self, x):
        for i, h in enumerate(self.hidden):
            x = nn.relu(nn.Dense(h, kernel_init=xavier(math.sqrt(2.0)), bias_init=nn.initializers.zeros,
                                 name=f"l{i}")(x))
        x = nn.Dense(self.out, kernel_init=xavier(1.0), bias_init=nn.initializers.zeros, name="out")(x)
        return jax.nn.log_softmax(x, -1) if self.log_softmax else x


def _gru_gates(gi, gh, h, H):
    r = jax.nn.sigmoid(gi[..., :H] + gh[..., :H])
    z = jax.nn.sigmoid(gi[..., H:2 * H] + gh[..., H:2 * H])
    n = jnp.tanh(gi[..., 2 * H:] + r * gh[..., 2 * H:])
    return r, z, n, (1.0 - z) * n + z * h


@jax.custom_vjp
def gru_scan(gi, w_hh, b_hh, h0):
    """Sequence GRU given the hoisted input projections gi = x W_ih + b_ih [T, 3H]. Returns hs [T, H].

    Custom VJP (tested against plain autodiff in tests/test_grads.py): the backward pass emits per-step gate
    cotangents and forms dW_hh = sum_t h_{t-1}^T dgh_t with ONE matmul, instead of accumulating a [H, 3H]
    weight cotangent inside the reverse scan (memory-bound when weights are per-run)."""
    return _gru_fwd(gi, w_hh, b_hh, h0)[0]


def _gru_fwd(gi, w_hh, b_hh, h0):
    H = h0.shape[-1]

    def body(h, g):
        gh = h @ w_hh + b_hh
        r, z, n, hn = _gru_gates(g, gh, h, H)
        return hn, (h, r, z, n, gh[..., 2 * H:])

    _, (hp, r, z, n, ghn) = jax.lax.scan(body, h0, gi)
    hs = (1.0 - z) * n + z * hp
    return hs, (hp, r, z, n, ghn, w_hh)


def _gru_bwd(res, dhs):
    hp, r, z, n, ghn, w_hh = res

    def body(dh_next, x):
        dh_out, hp_t, r_t, z_t, n_t, ghn_t = x
        dh = dh_out + dh_next
        dn = dh * (1.0 - z_t)
        dz = dh * (hp_t - n_t) * z_t * (1.0 - z_t)
        dn_pre = dn * (1.0 - n_t * n_t)
        dr = dn_pre * ghn_t * r_t * (1.0 - r_t)
        dgh = jnp.concatenate([dr, dz, dn_pre * r_t], -1)
        dgi = jnp.concatenate([dr, dz, dn_pre], -1)
        dh_prev = dh * z_t + dgh @ w_hh.T
        return dh_prev, (dgi, dgh)

    dh0, (dgi, dgh) = jax.lax.scan(body, jnp.zeros_like(hp[0]), (dhs, hp, r, z, n, ghn), reverse=True)
    dw_hh = jnp.einsum("ti,tj->ij", hp, dgh)
    return dgi, dw_hh, dgh.sum(0), dh0


gru_scan.defvjp(_gru_fwd, _gru_bwd)


class GRUCell(nn.Module):
    """torch.nn.GRU cell (gate order r, z, n; n = tanh(W_in x + b_in + r * (W_hn h + b_hn))), init U(+-1/sqrt(H))."""
    hidden: int
    in_dim: int

    def setup(self):
        H = self.hidden
        b = 1.0 / math.sqrt(H)
        u = lambda k, s: jax.random.uniform(k, s, jnp.float32, -b, b)  # noqa: E731
        self.w_ih = self.param("w_ih", u, (self.in_dim, 3 * H))
        self.w_hh = self.param("w_hh", u, (H, 3 * H))
        self.b_ih = self.param("b_ih", u, (3 * H,))
        self.b_hh = self.param("b_hh", u, (3 * H,))

    def __call__(self, h, x):
        hn = _gru_gates(x @ self.w_ih + self.b_ih, h @ self.w_hh + self.b_hh, h, self.hidden)[3]
        return hn, hn

    def seq(self, x, h0):
        return gru_scan(x @ self.w_ih + self.b_ih, self.w_hh, self.b_hh, h0)


class InputFeatures(nn.Module):
    """phi(a_{t-1}) ++ phi(o_t). prev_action = -1 -> zero action features."""
    spec: NetSpec

    @nn.compact
    def __call__(self, prev_action, obs):
        sp = self.spec
        if sp.kind == "categorical":
            a_tab = self.param("a_emb", nn.initializers.normal(1.0), (sp.n_actions, sp.emb))
            o_tab = self.param("o_emb", nn.initializers.normal(1.0), (sp.n_obs, sp.emb))
            fa = jnp.take(a_tab, jnp.clip(prev_action, 0, sp.n_actions - 1), axis=0)
            fo = jnp.take(o_tab, jnp.clip(obs, 0, sp.n_obs - 1), axis=0)
            fo = jnp.where((obs >= sp.n_obs)[..., None], 0.0, fo)   # reset observation -> fixed zero vector
        else:
            fa = jax.nn.one_hot(prev_action, sp.n_actions)
            fo = obs.astype(jnp.float32)
        fa = jnp.where((prev_action < 0)[..., None], 0.0, fa)
        return jnp.concatenate([fa, fo], -1)


class StateFeatures(nn.Module):
    spec: NetSpec

    @nn.compact
    def __call__(self, state):
        sp = self.spec
        if sp.kind == "categorical":
            tab = self.param("s_emb", nn.initializers.normal(1.0), (sp.n_states, sp.emb))
            return jnp.take(tab, jnp.clip(state, 0, sp.n_states - 1), axis=0)
        return state.astype(jnp.float32)


class HistoryEncoder(nn.Module):
    """Sequence features [T, H] for full histories (react_k = 0) or react-k windows; one-step update for rollouts."""
    spec: NetSpec
    react_k: int = 0

    def setup(self):
        self.inp = InputFeatures(self.spec)
        self.cell = GRUCell(self.spec.hidden, self.in_dim())

    def in_dim(self):
        sp = self.spec
        return (sp.n_actions if sp.kind == "vector" else sp.emb) + (sp.n_obs if sp.kind == "vector" else sp.emb)

    def sequence(self, actions, obs):
        """actions [T] (a_t), obs [T] (o_t, o_0 = reset obs) -> features [T, H] of h_t (h_t includes o_t)."""
        prev = jnp.concatenate([jnp.full((1,), -1, actions.dtype), actions[:-1]])
        x = self.inp(prev, obs)
        H = self.spec.hidden
        if self.react_k == 0:
            return self.cell.seq(x, jnp.zeros((H,), x.dtype))
        k = self.react_k
        T = x.shape[0]
        xp = jnp.concatenate([jnp.zeros((k - 1, x.shape[-1]), x.dtype), x], 0)
        h = jnp.zeros((T, H), x.dtype)
        for i in range(k):  # window element i of every t: xp[t + i]
            h, _ = self.cell(h, jax.lax.dynamic_slice_in_dim(xp, i, T, 0))
        return h

    def init_carry(self):
        H = self.spec.hidden
        if self.react_k == 0:
            return jnp.zeros((H,))
        return jnp.zeros((self.react_k, self.in_dim()))

    def step(self, carry, prev_action, obs):
        """Incremental update: returns (new_carry, features of the history ending at obs)."""
        x = self.inp(prev_action, obs)
        if self.react_k == 0:
            h, _ = self.cell(carry, x)
            return h, h
        buf = jnp.concatenate([carry[1:], x[None]], 0)
        h = jnp.zeros((self.spec.hidden,), x.dtype)
        for i in range(self.react_k):
            h, _ = self.cell(h, buf[i])
        return buf, h


class Actor(nn.Module):
    spec: NetSpec
    react_k: int = 0

    def setup(self):
        self.enc = HistoryEncoder(self.spec, self.react_k)
        self.pi = MLP(self.spec.head, self.spec.n_actions, log_softmax=True)

    def __call__(self, actions, obs):  # sequence log-probs [T, A]
        return self.pi(self.enc.sequence(actions, obs))

    def init_carry(self):
        return self.enc.init_carry()

    def step(self, carry, prev_action, obs):
        carry, f = self.enc.step(carry, prev_action, obs)
        return carry, self.pi(f)


class Critic(nn.Module):
    """mode: 'h' (V(h)), 's' (V(s)), 'hs' (V(h,s))."""
    spec: NetSpec
    mode: str
    react_k: int = 0

    def setup(self):
        if "h" in self.mode:
            self.enc = HistoryEncoder(self.spec, self.react_k)
        if "s" in self.mode:
            self.sf = StateFeatures(self.spec)
        self.v = MLP(self.spec.head, 1)

    def __call__(self, actions, obs, states):  # values [T]
        feats = []
        if "h" in self.mode:
            feats.append(self.enc.sequence(actions, obs))
        if "s" in self.mode:
            feats.append(self.sf(states))
        return self.v(jnp.concatenate(feats, -1))[..., 0]


METHODS = {
    # name: (critic mode, react_k)          paper §5 / §3.4 / §3.3 / §6; official AsymA2C / AsymA2C_State / A2C
    "a2c_asym_hs": ("hs", 0),
    "a2c_asym_s": ("s", 0),
    "a2c": ("h", 0),
    "a2c_react_2": ("h", 2),
    "a2c_react_4": ("h", 4),
}


def build(spec: NetSpec, method: str):
    mode, k = METHODS[method]
    return Actor(spec, k), Critic(spec, mode, k)


def init_params(spec: NetSpec, method: str, key):
    actor, critic = build(spec, method)
    ka, kc = jax.random.split(key)
    a0 = jnp.zeros((3,), jnp.int32)
    o0 = jnp.zeros((3,), jnp.int32) if spec.kind == "categorical" else jnp.zeros((3, spec.n_obs))
    s0 = jnp.zeros((3,), jnp.int32) if spec.kind == "categorical" else jnp.zeros((3, spec.n_states))
    pa = actor.init(ka, a0, o0)["params"]
    pc = critic.init(kc, a0, o0, s0)["params"]
    return pa, pc
