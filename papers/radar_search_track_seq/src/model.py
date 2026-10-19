"""Networks (paper §2.6, §3): track-list encoders {Flat, BiGRU, BiGRU+MHSA}, NatureCNN on the scan-history raster,
separate 2x128 tanh policy / value MLPs (SB3 ActorCriticPolicy semantics), a factorised (az, el) categorical head, and
the auto-encoder decoder used for track-extractor pre-training (§2.6.2).

All networks run in float32. Orthogonal init with SB3 gains (sqrt 2 for Dense/Conv in the CNN and MLPs, 0.01 for the
action head, 1.0 for the value head).

init_style (NetConfig; official-code oracle phase, README 'Official-code verification', fixture P8):
  "torch" (default from the oracle phase on): what the authors' toolchain does to a torch track extractor. SB3's
          ActorCriticPolicy re-initialises EVERY nn.Linear / nn.Conv2d inside the features extractor (orthogonal,
          gain sqrt 2, zero bias) -- this includes the embed Dense, the BiGRU output Dense and the MHA out_proj (a
          Linear subclass). It leaves nn.GRU (U(+-1/sqrt H) on all weights and both biases) and the MHA in_proj
          (xavier_uniform on the stacked [3d, d] matrix, zero bias) at torch defaults. LayerNorm eps = 1e-5 (torch).
          The AE decoder is not part of the policy: torch Linear default U(+-1/sqrt fan_in) on weight and bias.
  "flax"  the small-scale setting (flax defaults: lecun_normal Dense / MHA kernels, orthogonal GRU recurrent kernel,
          zero biases, LayerNorm eps 1e-6). Kept selectable so the small-scale runs stay reproducible.
"""
from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.configs.config import EnvConfig, NetConfig
from src.data import Obs, render_raster

SQ2 = math.sqrt(2.0)


def ortho(g):
    return nn.initializers.orthogonal(g)


def torch_uniform(bound):
    """U(-bound, bound), torch's default for nn.GRU parameters (bound 1/sqrt H) and nn.Linear (1/sqrt fan_in)."""
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init


def torch_gru_bi(H):
    """Our input bias bi = b_ih + [b_hr, b_hz, 0] (torch keeps two biases): the r/u parts are a sum of two
    U(+-1/sqrt H) draws, the n part one draw (the n part of b_hh is our bhn)."""
    k = 1.0 / math.sqrt(H)

    def init(key, shape, dtype=jnp.float32):
        k1, k2 = jax.random.split(key)
        a = jax.random.uniform(k1, shape, dtype, -k, k)
        b = jax.random.uniform(k2, shape, dtype, -k, k)
        return a + jnp.where(jnp.arange(shape[0]) < 2 * H, b, 0.0).astype(dtype)
    return init


def ortho_reshaped(g, n_in):
    """Orthogonal(g) on the 2-D [n_in, n_out] matrix, reshaped to the flax DenseGeneral kernel shape."""
    base = nn.initializers.orthogonal(g)

    def init(key, shape, dtype=jnp.float32):
        n_out = int(math.prod(shape)) // n_in
        return base(key, (n_in, n_out), dtype).reshape(shape)
    return init


def dense_inits(style):
    """(kernel_init, bias_init) for a Dense inside the track extractor (SB3 re-initialises it under "torch")."""
    if style == "torch":
        return ortho(SQ2), nn.initializers.zeros
    return nn.initializers.lecun_normal(), nn.initializers.zeros


class BiGRU(nn.Module):
    """Bidirectional GRU over the rows of x [..., K, D] with masked carry-through (README padding lemma).

    Cell (flax GRUCell convention): r = s(x W_ir + b_ir + h W_hr), u = s(x W_iu + b_iu + h W_hu),
    n = tanh(x W_in + b_in + r * (h W_hn + b_hn)), h' = (1 - u) n + u h; masked step: h' = h.
    Input projections are computed for all rows at once; both directions advance in one lax.scan
    (forward over rows 0..K-1, backward over K-1..0). Returns [h_fwd_final; h_bwd_final] (2H).
    Init as flax GRUCell: lecun_normal input kernels, orthogonal recurrent kernels, zero biases."""
    hidden: int
    unroll: int = 17
    init_style: str = "torch"

    @nn.compact
    def __call__(self, x, mask):
        H = self.hidden
        D = x.shape[-1]
        if self.init_style == "torch":
            k = 1.0 / math.sqrt(H)
            i_wi, i_bi, i_wh, i_bhn = torch_uniform(k), torch_gru_bi(H), torch_uniform(k), torch_uniform(k)
        else:
            i_wi, i_bi = nn.initializers.lecun_normal(), nn.initializers.zeros
            i_wh, i_bhn = nn.initializers.orthogonal(), nn.initializers.zeros
        p = {}
        for d in ("fwd", "bwd"):
            p[d] = dict(
                wi=self.param(f"{d}_wi", i_wi, (D, 3 * H), jnp.float32),
                bi=self.param(f"{d}_bi", i_bi, (3 * H,), jnp.float32),
                wh=self.param(f"{d}_wh", i_wh, (H, 3 * H), jnp.float32),
                bhn=self.param(f"{d}_bhn", i_bhn, (H,), jnp.float32))
        K = x.shape[-2]
        xi = {d: x @ p[d]["wi"] + p[d]["bi"] for d in p}                     # [..., K, 3H]
        dt = jnp.result_type(x.dtype, p["fwd"]["wi"].dtype)
        hf = jnp.zeros(x.shape[:-2] + (H,), dt)
        hb = jnp.zeros(x.shape[:-2] + (H,), dt)

        def cell(h, xt, q):
            hh = h @ q["wh"]
            r = jax.nn.sigmoid(xt[..., :H] + hh[..., :H])
            u = jax.nn.sigmoid(xt[..., H:2 * H] + hh[..., H:2 * H])
            n = jnp.tanh(xt[..., 2 * H:] + r * (hh[..., 2 * H:] + q["bhn"]))
            return (1.0 - u) * n + u * h

        xs = (jnp.moveaxis(xi["fwd"], -2, 0), jnp.moveaxis(xi["bwd"], -2, 0)[::-1],
              jnp.moveaxis(mask, -1, 0), jnp.moveaxis(mask, -1, 0)[::-1])

        def body(c, inp):
            hf, hb = c
            xf, xb, mf, mb = inp
            hf = jnp.where(mf[..., None], cell(hf, xf, p["fwd"]), hf)
            hb = jnp.where(mb[..., None], cell(hb, xb, p["bwd"]), hb)
            return (hf, hb), None

        (hf, hb), _ = jax.lax.scan(body, (hf, hb), xs, unroll=self.unroll)
        return jnp.concatenate([hf, hb], -1)


class MHSA(nn.Module):
    """One multi-head self-attention block over the track rows: masked keys, residual + LayerNorm, no FFN ."""
    d: int
    heads: int
    ln_eps: float = 1e-5          # torch nn.LayerNorm default (P5); the small-scale runs used flax's 1e-6
    init_style: str = "torch"

    @nn.compact
    def __call__(self, e, mask):
        att_mask = mask[..., None, None, :]                              # [..., 1(heads), 1(q), K]
        if self.init_style == "torch":
            kw = dict(kernel_init=torch_uniform(math.sqrt(6.0 / (self.d + 3 * self.d))),   # xavier on [3d, d]
                      out_kernel_init=ortho_reshaped(SQ2, self.d))                        # SB3 re-init of out_proj
        else:
            kw = {}
        a = nn.MultiHeadDotProductAttention(num_heads=self.heads, qkv_features=self.d, out_features=self.d,
                                            name="mha", **kw)(e, e, mask=att_mask)
        return nn.LayerNorm(epsilon=self.ln_eps, name="ln")(e + a)


class TrackEncoder(nn.Module):
    """z_TL from the track list (rows [..., K, 7], mask [..., K])."""
    cfg: NetConfig

    @nn.compact
    def __call__(self, rows, mask):
        c = self.cfg
        mask = mask if c.masked else jnp.ones_like(mask)
        rows = rows * mask[..., None]
        if c.encoder == "flat":
            return rows.reshape(rows.shape[:-2] + (-1,))                  # [.., 119]
        ki, bi = dense_inits(c.init_style)
        if c.encoder == "bigru":
            h = BiGRU(c.bigru_hidden, init_style=c.init_style, name="bigru")(rows, mask)
            return nn.Dense(c.bigru_out, kernel_init=ki, bias_init=bi, name="out")(h)   # [.., 64] 
        if c.encoder == "mhsa":
            e = nn.Dense(c.mhsa_d, kernel_init=ki, bias_init=bi, name="embed")(rows)
            ln_eps = 1e-5 if c.init_style == "torch" else 1e-6
            e = MHSA(c.mhsa_d, c.mhsa_heads, ln_eps=ln_eps, init_style=c.init_style, name="mhsa")(e, mask)
            return BiGRU(c.mhsa_gru_hidden, init_style=c.init_style, name="bigru")(e, mask)   # [.., 64] 
        raise ValueError(c.encoder)


class NatureCNN(nn.Module):
    """NatureCNN (Mnih et al. 2015 / SB3): conv 32@8x8/4, 64@4x4/2, 64@3x3/1 (ReLU) -> flatten -> Dense(out) ReLU."""
    out: int = 128

    @nn.compact
    def __call__(self, img):                                              # img [..., 48, 48]
        x = img[..., None]
        x = nn.relu(nn.Conv(32, (8, 8), (4, 4), padding="VALID", kernel_init=ortho(SQ2))(x))
        x = nn.relu(nn.Conv(64, (4, 4), (2, 2), padding="VALID", kernel_init=ortho(SQ2))(x))
        x = nn.relu(nn.Conv(64, (3, 3), (1, 1), padding="VALID", kernel_init=ortho(SQ2))(x))
        x = x.reshape(x.shape[:-3] + (-1,))
        return nn.relu(nn.Dense(self.out, kernel_init=ortho(SQ2))(x))


def raster_of(obs: Obs, ecfg: EnvConfig):
    """Render the scan-history raster for any leading batch dims."""
    f = lambda a, e, g, v: render_raster(a, e, g, v, ecfg)
    lead = obs.sv_az.shape[:-1]
    flat = [x.reshape((-1,) + x.shape[len(lead):]) for x in (obs.sv_az, obs.sv_el, obs.sv_age, obs.sv_valid)]
    img = jax.vmap(f)(*flat)
    return img.reshape(lead + img.shape[1:])


class ActorCritic(nn.Module):
    """features = [z_TL; NatureCNN(raster)] shared by pi and V (SB3 share_features_extractor); separate 2x128 tanh
    MLPs; pi -> 2 x n_a logits (factorised az, el), V -> scalar."""
    cfg: NetConfig
    ecfg: EnvConfig

    def setup(self):
        self.track_enc = TrackEncoder(self.cfg, name="track_enc")
        self.cnn = NatureCNN(self.cfg.cnn_out, name="cnn")
        self.pi_layers = [nn.Dense(h, kernel_init=ortho(SQ2), name=f"pi_{i}") for i, h in enumerate(self.cfg.mlp)]
        self.v_layers = [nn.Dense(h, kernel_init=ortho(SQ2), name=f"v_{i}") for i, h in enumerate(self.cfg.mlp)]
        self.action_net = nn.Dense(2 * self.ecfg.n_a, kernel_init=ortho(0.01), name="action_net")
        self.value_net = nn.Dense(1, kernel_init=ortho(1.0), name="value_net")

    def features(self, obs: Obs):
        z_tl = self.track_enc(obs.tl.astype(jnp.float32), obs.tl_mask)
        z_sh = self.cnn(raster_of(obs, self.ecfg).astype(jnp.float32))
        return jnp.concatenate([z_tl, z_sh], -1)

    def heads(self, f):
        hp, hv = f, f
        for l in self.pi_layers:
            hp = jnp.tanh(l(hp))
        for l in self.v_layers:
            hv = jnp.tanh(l(hv))
        logits = self.action_net(hp)
        n = self.ecfg.n_a
        return logits[..., :n], logits[..., n:], self.value_net(hv)[..., 0]

    def __call__(self, obs: Obs):
        return self.heads(self.features(obs))


class AEDecoder(nn.Module):
    """Decoder for track-extractor pre-training: Dense(64 -> 256) ReLU -> Dense(256 -> K*7) ."""
    hidden: int
    out: int
    init_style: str = "torch"

    @nn.compact
    def __call__(self, z):
        if self.init_style == "torch":                                  # torch nn.Linear default
            l1 = nn.Dense(self.hidden, kernel_init=torch_uniform(1 / math.sqrt(z.shape[-1])),
                          bias_init=torch_uniform(1 / math.sqrt(z.shape[-1])))
            l2 = nn.Dense(self.out, kernel_init=torch_uniform(1 / math.sqrt(self.hidden)),
                          bias_init=torch_uniform(1 / math.sqrt(self.hidden)))
        else:
            l1, l2 = nn.Dense(self.hidden), nn.Dense(self.out)
        return l2(nn.relu(l1(z)))


def dummy_obs(ecfg: EnvConfig, lead=()) -> Obs:
    K, N = ecfg.k_max, ecfg.n_sv
    f = lambda *s: jnp.zeros(lead + s, jnp.float32)
    return Obs(f(K, 7), jnp.zeros(lead + (K,), bool), f(N), f(N), jnp.ones(lead + (N,), jnp.float32),
               jnp.zeros(lead + (N,), bool))
