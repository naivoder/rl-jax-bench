"""Networks (flax.linen leaf modules in PyTorch weight layout, composed functionally).

Parameter layout follows torch exactly (Linear weight (out, in); LSTM/GRU w_ih (G*H, in), w_hh (G*H, H),
b_ih, b_hh) so the differential tests can load the official PyTorch weights one-to-one.

Architecture (paper Fig. 7 / App. B; official recurrent_actor.py / recurrent_critic.py / policy_rnn_shared.py):

  encoder:  x_t = [relu(W_a a_{t-1}), relu(W_r r_t), relu(W_o o_t)]  (absent input => embedder removed)
            h_t = RNN(h_{t-1}, x_t), h_{-1} = 0                     (torch LSTM/GRU equations)
  actor:    z_t = [h_t, relu(W_s o_t)]  ->  MLP [128,128] ReLU  ->  td3: tanh(.) | sac: (mean, log_std) | sacd: logits
  critic:   z_t = [h_t, relu(W_sa [o_t, a_t])] (sacd: relu(W_s o_t) of width obs+act+rew embedding -- the official
            code's observation-only shortcut; App. B says the embedder is REMOVED for SAC-D, a deviation used only by
            the toy stage toy) -> twin MLP heads -> Q1, Q2

Initialisation (official): RNN weights orthogonal (torch ``orthogonal_`` on the stacked gate matrix), RNN biases 0;
embedders torch-default U(+-1/sqrt(in)); MLP hidden layers U(+-1/sqrt(OUT)) (rlkit ``fanin_init`` reads size[0]
of the torch weight, i.e. the output dim) with bias 0.1; MLP last layer U(+-init_w) (critic 3e-3, actor 1e-3,
also the SAC log-std layer).
"""
from __future__ import annotations

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32
LOG_SIG_MIN, LOG_SIG_MAX = -20.0, 2.0  # official actor.py


# ------------------------------------------------------------------------------------------------ initialisers
def _u(bound):
    def init(key, shape, dtype=F32):
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init


def torch_default_w(key, shape, dtype=F32):  # nn.Linear default: kaiming_uniform(a=sqrt 5) == U(+-1/sqrt(in))
    return _u(1.0 / np.sqrt(shape[1]))(key, shape, dtype)


def rlkit_fanin_w(key, shape, dtype=F32):  # torchkit.pytorch_utils.fanin_init: fan_in = size[0] = OUT dim
    return _u(1.0 / np.sqrt(shape[0]))(key, shape, dtype)


def torch_orthogonal(key, shape, dtype=F32):
    """torch.nn.init.orthogonal_ (gain 1) on a (rows, cols) matrix."""
    rows, cols = shape
    a = jax.random.normal(key, (rows, cols), dtype)
    if rows < cols:
        a = a.T
    q, r = jnp.linalg.qr(a)
    q = q * jnp.sign(jnp.diag(r))[None, :]
    if rows < cols:
        q = q.T
    return q.astype(dtype)


# ------------------------------------------------------------------------------------------------ leaf modules
class TorchLinear(nn.Module):
    features: int
    kind: str = "default"  # default | fanin | last
    init_w: float = 3e-3
    b_value: float = 0.1

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        if self.kind == "default":
            w = self.param("weight", torch_default_w, (self.features, fan_in))
            b = self.param("bias", _u(1.0 / np.sqrt(fan_in)), (self.features,))
        elif self.kind == "fanin":
            w = self.param("weight", rlkit_fanin_w, (self.features, fan_in))
            b = self.param("bias", nn.initializers.constant(self.b_value), (self.features,))
        elif self.kind == "last":
            w = self.param("weight", _u(self.init_w), (self.features, fan_in))
            b = self.param("bias", _u(self.init_w), (self.features,))
        else:
            raise ValueError(self.kind)
        return x @ w.T + b


class MLP(nn.Module):
    """rlkit/torchkit Mlp: hidden fanin-quirk layers + ReLU, last layer U(+-init_w); n_out heads share the trunk
    (n_out=2 is the TanhGaussianPolicy: last_fc and last_fc_log_std)."""
    hidden: tuple
    out: int
    init_w: float = 3e-3
    n_out: int = 1

    @nn.compact
    def __call__(self, x):
        for i, hsz in enumerate(self.hidden):
            x = nn.relu(TorchLinear(hsz, "fanin", name=f"fc{i}")(x))
        outs = [TorchLinear(self.out, "last", self.init_w, name="last_fc")(x)]
        if self.n_out == 2:
            outs.append(TorchLinear(self.out, "last", self.init_w, name="last_fc_log_std")(x))
        return outs[0] if self.n_out == 1 else tuple(outs)


class TorchRNN(nn.Module):
    """Single-layer torch LSTM (gates i,f,g,o) or GRU (gates r,z,n) over a time-major sequence (T, B, in).
    The input projection is computed for all steps at once; only h @ w_hh runs inside the scan."""
    cell: str
    hidden: int
    unroll: int = 1
    fast_bptt: bool = True

    @nn.compact
    def __call__(self, x, carry=None):
        G = 4 if self.cell == "lstm" else 3
        H = self.hidden
        w_ih = self.param("w_ih", torch_orthogonal, (G * H, x.shape[-1]))
        w_hh = self.param("w_hh", torch_orthogonal, (G * H, H))
        b_ih = self.param("b_ih", nn.initializers.zeros, (G * H,))
        b_hh = self.param("b_hh", nn.initializers.zeros, (G * H,))
        if carry is None and self.fast_bptt:
            return rnn_seq(self.cell, w_ih, w_hh, b_ih, b_hh, x, self.unroll), None
        return rnn_scan(self.cell, w_ih, w_hh, b_ih, b_hh, x, carry, self.unroll)


def zero_carry(cell, batch_shape, H):
    z = jnp.zeros(tuple(batch_shape) + (H,), F32)
    return (z, z) if cell == "lstm" else z


def rnn_cell(cell, w_hh, b_hh, carry, xi):
    """one step given the precomputed input projection xi = x @ w_ih.T + b_ih."""
    if cell == "lstm":
        h, c = carry
        g = xi + h @ w_hh.T + b_hh
        i, f, gg, o = jnp.split(g, 4, axis=-1)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(gg)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return (h, c), h
    h = carry
    hh = h @ w_hh.T + b_hh
    xr, xz, xn = jnp.split(xi, 3, axis=-1)
    hr, hz, hn = jnp.split(hh, 3, axis=-1)
    r = jax.nn.sigmoid(xr + hr)
    z = jax.nn.sigmoid(xz + hz)
    n = jnp.tanh(xn + r * hn)
    h = (1.0 - z) * n + z * h
    return h, h


# ------------------------------------------------------------------------------------------------ fast BPTT
# Training unrolls start from the zero state and only need the hidden sequence.  ``rnn_seq`` is the same
# function as ``rnn_scan`` with a hand-written VJP: the reverse loop only propagates (dh, dc) through time and emits
# the per-step gate gradients; the weight gradients are then ONE einsum over (T, B) after the loop (instead of
# accumulating a (G*H, H) weight cotangent in the scan carry at every step, which is what autodiff of lax.scan
# does).  tests/test_grads.py checks it against autodiff of ``rnn_scan`` and finite differences.
def _fwd_steps(cell, w_hh, b_hh, xi, unroll):
    B, H = xi.shape[1], w_hh.shape[1]
    carry0 = zero_carry(cell, (B,), H)

    def step(carry, x_t):
        if cell == "lstm":
            h, c = carry
            g = x_t + h @ w_hh.T + b_hh
            i, f, gg, o = jnp.split(g, 4, axis=-1)
            i, f, gg, o = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jnp.tanh(gg), jax.nn.sigmoid(o)
            c2 = f * c + i * gg
            h2 = o * jnp.tanh(c2)
            return (h2, c2), (h2, (i, f, gg, o, c2))
        h = carry
        hh = h @ w_hh.T + b_hh
        xr, xz, xn = jnp.split(x_t, 3, axis=-1)
        hr, hz, hn = jnp.split(hh, 3, axis=-1)
        r = jax.nn.sigmoid(xr + hr)
        z = jax.nn.sigmoid(xz + hz)
        n = jnp.tanh(xn + r * hn)
        h2 = (1.0 - z) * n + z * h
        return h2, (h2, (r, z, n, hn))
    _, (hs, saved) = jax.lax.scan(step, carry0, xi, unroll=unroll)
    return hs, saved


@partial(jax.custom_vjp, nondiff_argnums=(0, 6))
def rnn_seq(cell, w_ih, w_hh, b_ih, b_hh, x, unroll=1):
    xi = x @ w_ih.T + b_ih
    return _fwd_steps(cell, w_hh, b_hh, xi, unroll)[0]


def _rnn_seq_fwd(cell, w_ih, w_hh, b_ih, b_hh, x, unroll):
    xi = x @ w_ih.T + b_ih
    hs, saved = _fwd_steps(cell, w_hh, b_hh, xi, unroll)
    return hs, (w_ih, w_hh, x, hs, saved)


def _rnn_seq_bwd(cell, unroll, res, dhs):
    w_ih, w_hh, x, hs, saved = res
    T, B, H = hs.shape
    h_prev = jnp.concatenate([jnp.zeros((1, B, H), hs.dtype), hs[:-1]], 0)
    if cell == "lstm":
        i, f, gg, o, c = saved
        c_prev = jnp.concatenate([jnp.zeros((1, B, H), c.dtype), c[:-1]], 0)

        def back(carry, inp):
            dh_n, dc_n = carry
            dh_t, i_t, f_t, g_t, o_t, c_t, cp_t = inp
            dh = dh_t + dh_n
            tc = jnp.tanh(c_t)
            do = dh * tc
            dc = dc_n + dh * o_t * (1.0 - tc * tc)
            da = jnp.concatenate([dc * g_t * i_t * (1.0 - i_t), dc * cp_t * f_t * (1.0 - f_t),
                                  dc * i_t * (1.0 - g_t * g_t), do * o_t * (1.0 - o_t)], -1)
            return (da @ w_hh, dc * f_t), da
        z0 = jnp.zeros((B, H), hs.dtype)
        _, da = jax.lax.scan(back, (z0, z0), (dhs, i, f, gg, o, c, c_prev), reverse=True, unroll=unroll)
        dxi, dhh = da, da
    else:
        r, z, n, hn = saved

        def back(dh_n, inp):
            dh_t, r_t, z_t, n_t, hn_t, hp_t = inp
            dh = dh_t + dh_n
            dn = dh * (1.0 - z_t)
            dz = dh * (hp_t - n_t)
            dan = dn * (1.0 - n_t * n_t)
            dar = dan * hn_t * r_t * (1.0 - r_t)
            daz = dz * z_t * (1.0 - z_t)
            dxi_t = jnp.concatenate([dar, daz, dan], -1)
            dhh_t = jnp.concatenate([dar, daz, dan * r_t], -1)
            return dh * z_t + dhh_t @ w_hh, (dxi_t, dhh_t)
        _, (dxi, dhh) = jax.lax.scan(back, jnp.zeros((B, H), hs.dtype), (dhs, r, z, n, hn, h_prev), reverse=True,
                                     unroll=unroll)
    d_w_hh = jnp.einsum("tbg,tbh->gh", dhh, h_prev)
    d_b_hh = jnp.sum(dhh, (0, 1))
    d_w_ih = jnp.einsum("tbg,tbi->gi", dxi, x)
    d_b_ih = jnp.sum(dxi, (0, 1))
    dx = dxi @ w_ih
    return d_w_ih, d_w_hh, d_b_ih, d_b_hh, dx


rnn_seq.defvjp(_rnn_seq_fwd, _rnn_seq_bwd)


def rnn_scan(cell, w_ih, w_hh, b_ih, b_hh, x, carry=None, unroll=1):
    H = w_hh.shape[1]
    if carry is None:
        carry = zero_carry(cell, x.shape[1:-1], H)
    xi = x @ w_ih.T + b_ih
    carry, hs = jax.lax.scan(lambda c, xt: rnn_cell(cell, w_hh, b_hh, c, xt), carry, xi, unroll=unroll)
    return hs, carry


class Encoder(nn.Module):
    """embedders (a_prev, r, o) -> RNN.  widths of 0 remove that input entirely (official FeatureExtractor(0))."""
    obs_emb: int
    act_emb: int
    rew_emb: int
    cell: str
    hidden: int
    unroll: int = 1
    fast_bptt: bool = True

    @nn.compact
    def __call__(self, prev_a, r, o, carry=None):
        parts = []  # official concat order (a, r, o)
        if self.act_emb:
            parts.append(nn.relu(TorchLinear(self.act_emb, name="action_embedder")(prev_a)))
        if self.rew_emb:
            parts.append(nn.relu(TorchLinear(self.rew_emb, name="reward_embedder")(r)))
        parts.append(nn.relu(TorchLinear(self.obs_emb, name="observ_embedder")(o)))
        x = jnp.concatenate(parts, -1)
        return TorchRNN(self.cell, self.hidden, self.unroll, self.fast_bptt, name="rnn")(x, carry)


class Shortcut(nn.Module):
    """relu(Linear(.)) branch on the current observation (actor) or observation-action (critic)."""
    features: int

    @nn.compact
    def __call__(self, x):
        return nn.relu(TorchLinear(self.features, name="fc")(x))


# ------------------------------------------------------------------------------------------------ factory
class Nets:
    """Holds the module definitions for one config and functional apply helpers on param sub-trees.

    Separate arch params:  actor = {enc, short, pi}, critic = {enc, short, q1, q2}
    Shared arch params:    {enc, short_o, short_oa, q1, q2, pi}   (targets: {q1, q2, pi})
    Markov arch params:    actor = {pi}, critic = {q1, q2}  (MLPs directly on the observation)
    """

    def __init__(self, cfg, obs_dim, act_dim, discrete):
        self.cfg = cfg
        self.obs_dim, self.act_dim, self.discrete = obs_dim, act_dim, discrete
        self.algo = cfg["algo"]
        inp = cfg["inputs"]
        self.obs_emb = int(cfg["obs_embed"])
        self.act_emb = int(cfg["action_embed"]) if "a" in inp[1:] else 0
        self.rew_emb = int(cfg["reward_embed"]) if "r" in inp[1:] else 0
        self.rnn_in = self.obs_emb + self.act_emb + self.rew_emb
        self.H = int(cfg["rnn_hidden"])
        self.cell = cfg["cell"]
        self.enc = Encoder(self.obs_emb, self.act_emb, self.rew_emb, self.cell, self.H, int(cfg.get("scan_unroll", 1)),
                           bool(cfg.get("fast_bptt", True)))
        self.short_o = Shortcut(self.obs_emb)
        self.short_oa = Shortcut(self.rnn_in)
        hid_pi = tuple(cfg["policy_layers"])
        hid_q = tuple(cfg["dqn_layers"])
        self.pi = MLP(hid_pi, act_dim, 1e-3, n_out=2 if self.algo == "sac" else 1)
        self.q = MLP(hid_q, act_dim if discrete else 1, 3e-3)

    # ---- init
    def init_separate(self, key):
        k = jax.random.split(key, 8)
        T, B, O, A = 2, 1, self.obs_dim, self.act_dim
        a0, r0, o0 = jnp.zeros((T, B, A)), jnp.zeros((T, B, 1)), jnp.zeros((T, B, O))
        zdim = self.H + self.obs_emb
        actor = {"enc": self.enc.init(k[0], a0, r0, o0)["params"],
                 "short": self.short_o.init(k[1], o0)["params"],
                 "pi": self.pi.init(k[2], jnp.zeros((1, zdim)))["params"]}
        sc_in = jnp.zeros((1, O)) if self.discrete else jnp.zeros((1, O + A))
        qdim = self.H + self.rnn_in
        critic = {"enc": self.enc.init(k[3], a0, r0, o0)["params"],
                  "short": self.short_oa.init(k[4], sc_in)["params"],
                  "q1": self.q.init(k[5], jnp.zeros((1, qdim)))["params"],
                  "q2": self.q.init(k[6], jnp.zeros((1, qdim)))["params"]}
        return actor, critic

    def init_shared(self, key):
        if self.discrete:
            raise NotImplementedError("the official shared architecture is continuous-action only")
        k = jax.random.split(key, 8)
        T, B, O, A = 2, 1, self.obs_dim, self.act_dim
        a0, r0, o0 = jnp.zeros((T, B, A)), jnp.zeros((T, B, 1)), jnp.zeros((T, B, O))
        qdim, zdim = self.H + self.rnn_in, self.H + self.obs_emb
        return {"enc": self.enc.init(k[0], a0, r0, o0)["params"],
                "short_oa": self.short_oa.init(k[1], jnp.zeros((1, O + A)))["params"],
                "q1": self.q.init(k[2], jnp.zeros((1, qdim)))["params"],
                "q2": self.q.init(k[3], jnp.zeros((1, qdim)))["params"],
                "short_o": self.short_o.init(k[4], o0)["params"],
                "pi": self.pi.init(k[5], jnp.zeros((1, zdim)))["params"]}

    def init_markov(self, key):
        k = jax.random.split(key, 4)
        O, A = self.obs_dim, self.act_dim
        actor = {"pi": self.pi.init(k[0], jnp.zeros((1, O)))["params"]}
        qin = jnp.zeros((1, O)) if self.discrete else jnp.zeros((1, O + A))
        critic = {"q1": self.q.init(k[1], qin)["params"], "q2": self.q.init(k[2], qin)["params"]}
        return actor, critic

    # ---- apply helpers
    def encode(self, enc_p, prev_a, r, o, carry=None):
        return self.enc.apply({"params": enc_p}, prev_a, r, o, carry)

    def encode_stack(self, enc_list, prev_a, r, o):
        """run several same-shaped encoders in ONE scan (vmapped over a stacked parameter axis)."""
        stacked = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *enc_list)
        hs, _ = jax.vmap(lambda p: self.enc.apply({"params": p}, prev_a, r, o))(stacked)
        return [hs[i] for i in range(len(enc_list))]

    def pi_out(self, pi_p, z):
        return self.pi.apply({"params": pi_p}, z)

    def q_out(self, q_p, z):
        return self.q.apply({"params": q_p}, z)

    def short_o_out(self, p, o):
        return self.short_o.apply({"params": p}, o)

    def short_oa_out(self, p, o, a):
        x = o if self.discrete else jnp.concatenate([o, a], -1)
        return self.short_oa.apply({"params": p}, x)


def rnn_params(enc_params):
    """the torch ``self.rnn`` module's parameters (official grad-norm metric scope)."""
    return enc_params["rnn"]


def tanh_gauss(mean, log_std, eps):
    """official TanhNormal.rsample + log_prob (epsilon 1e-6), log_std clamped to [-20, 2]."""
    log_std = jnp.clip(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
    std = jnp.exp(log_std)
    z = mean + std * eps
    a = jnp.tanh(z)
    lp = -((z - mean) ** 2) / (2 * std ** 2) - jnp.log(std) - np.log(np.sqrt(2 * np.pi))
    lp = lp - jnp.log(1 - a * a + 1e-6)
    return a, jnp.sum(lp, -1, keepdims=True)


def categorical(logits):
    """official CategoricalPolicy: softmax probs, log(clamp(p, 1e-8))."""
    p = jax.nn.softmax(logits, -1)
    return p, jnp.log(jnp.clip(p, 1e-8))


def n_params(tree) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(tree)))
