"""Networks and the action distribution of the configurable on-policy agent (paper App. B.7, B.8; Table 2).

Pure-function JAX (no flax module state): parameters are plain dicts so that lanes (arm x seed) can be stacked
and vmapped. One *superset* parameter tree serves every arm:

* ``pi``   policy MLP (width W_pi, depth 2, tanh) + linear head to the action mean x_mu   (C49, C52, C55)
* ``std``  global, trainable std pre-activation x_rho, one per action dim, init 0          (C59 = True)
* ``v``    separate value MLP, zero-padded to the largest width among the lanes            (C50, C53)
* ``vs``   value head on the POLICY trunk, used only when ``shared`` is on                  (C47 = shared)

Zero padding is exact: a hidden unit whose incoming and outgoing weights and bias are all zero outputs tanh(0) = 0
and receives an exactly-zero gradient forever, so a width-64 value net padded to 256 trains bit-for-bit like a
width-64 net (Adam's update on an all-zero gradient history is exactly 0; the joint grad norm is unchanged).
Likewise the unused branch (``vs`` for separate lanes, ``v`` for shared lanes) gets exactly-zero gradients.

Initialisers follow the official gin (seed_rl@0e1e0ac mujoco base config; README): hidden layers
Orthogonal(gain sqrt 2), last policy layer Orthogonal(gain g_pi) (0.01 base, 1.0 in arm A2a), last value layer
Orthogonal(1.0), zero biases (keras Dense default).

Action distribution (B.8, official ``parametric_distribution``):
  x ~ N(x_mu, sigma),  sigma = T_rho(x_rho + c_rho) + min_std,  c_rho = T_rho^{-1}(init_std - min_std)
  T_rho = softplus or safe_exp (exp of the input clipped to [-15, 15] with straight-through gradient).
  STUDY-ERA semantics (``presquash`` = True, the default since official-code check; seed_rl@68c15aa NormalSquashedDistribution,
  the "postprocessing" design in force when the study ran, README): the stored action is the PRE-squash sample
  x, log p(x) = log N(x; mu, sigma) - log|d post(x)/dx| (tanh: the tanh log-det; clip: 0), the env gets tanh(x) or
  clip(x, -1, 1).
  RELEASED-HEAD semantics (``presquash`` = False; seed_rl@0e1e0ac, after the 2020-12-01 refactor; diagnostic arms):
  ``tanh``: the stored action is u = tanh(x); log p(u) uses atanh(clip(u, +-0.999)) and the tail mass beyond the
  threshold (TanhTransformedDistribution). ``clip``: the stored action is clip(x, -1, 1) and log p is the
  Gaussian density AT the clipped action (ClippedIdentity in a TransformedDistribution).
"""
from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm

TANH_THRESHOLD = 0.999          # official TanhTransformedDistribution(threshold=.999)
_ATANH_T = math.atanh(TANH_THRESHOLD)
_LOG_EPS_T = math.log(1.0 - TANH_THRESHOLD)
LOG2 = math.log(2.0)


# ------------------------------------------------------------------ initialisation
def orthogonal(key, shape, gain):
    """Haar-orthogonal matrix of ``shape`` times ``gain`` (same distribution as keras Orthogonal(gain))."""
    return jax.nn.initializers.orthogonal(scale=gain)(key, shape, jnp.float32)


def _dense(key, n_in, n_out, gain, pad_in=None, pad_out=None):
    w = orthogonal(key, (n_in, n_out), gain)
    pad_in, pad_out = pad_in or n_in, pad_out or n_out
    W = jnp.zeros((pad_in, pad_out), jnp.float32).at[:n_in, :n_out].set(w)
    return {"kernel": W, "bias": jnp.zeros((pad_out,), jnp.float32)}


def init_params(key, obs_dim, act_dim, *, pi_width=64, pi_depth=2, v_width=64, v_depth=2, v_width_max=None,
                pi_last_gain=0.01, hidden_gain=math.sqrt(2.0), v_last_gain=1.0):
    """One lane's parameters. Every layer has its own fold_in(key, layer_id) stream, so two lanes with the same key
    get identical policy trunks whatever their value width (pairing; README)."""
    vmax = v_width_max or v_width
    k = lambda i: jax.random.fold_in(key, i)
    pi, n = {}, obs_dim
    for d in range(pi_depth):
        pi[f"h{d}"] = _dense(k(d), n, pi_width, hidden_gain)
        n = pi_width
    pi["out"] = _dense(k(10), pi_width, act_dim, pi_last_gain)
    v, n, pn = {}, obs_dim, obs_dim
    for d in range(v_depth):
        v[f"h{d}"] = _dense(k(20 + d), n, v_width, hidden_gain, pad_in=pn, pad_out=vmax)
        n, pn = v_width, vmax
    v["out"] = _dense(k(30), v_width, 1, v_last_gain, pad_in=vmax)
    vs = _dense(k(40), pi_width, 1, v_last_gain)            # shared-trunk value head (: single linear layer)
    return {"pi": pi, "std": jnp.zeros((act_dim,), jnp.float32), "v": v, "vs": vs}


# ------------------------------------------------------------------ forward (flax.linen)
class _Trunk(nn.Module):
    """tanh MLP (C55) + linear output layer, parameter names h0..h{depth-1}, out (flax Dense kernel/bias)."""
    widths: tuple
    n_out: int

    @nn.compact
    def __call__(self, x):
        for i, w in enumerate(self.widths):
            x = jnp.tanh(nn.Dense(w, name=f"h{i}")(x))
        return x, nn.Dense(self.n_out, name="out")(x)


class Agent(nn.Module):
    """Policy MLP -> x_mu, global std vector x_rho, separate value MLP, and the shared-trunk value head ``vs``.
    Parameters are created by ``init_params`` (own key tree + zero padding), never by ``Agent.init``."""
    act_dim: int
    pi_widths: tuple
    v_widths: tuple

    @nn.compact
    def __call__(self, obs, shared):
        h, mu = _Trunk(self.pi_widths, self.act_dim, name="pi")(obs)
        v_sh = nn.Dense(1, name="vs")(h)[..., 0]
        _, v_sep = _Trunk(self.v_widths, 1, name="v")(obs)
        x_rho = jnp.broadcast_to(self.param("std", nn.initializers.zeros, (self.act_dim,)), mu.shape)
        return mu, x_rho, jnp.where(shared, v_sh, v_sep[..., 0])


def forward(params, obs, shared, pi_depth=2, v_depth=2):
    """obs: (..., obs_dim) ALREADY normalised/clipped. Returns (x_mu, x_rho, value). ``shared`` is a traced bool."""
    agent = Agent(act_dim=params["std"].shape[0],
                  pi_widths=tuple(params["pi"][f"h{d}"]["kernel"].shape[1] for d in range(pi_depth)),
                  v_widths=tuple(params["v"][f"h{d}"]["kernel"].shape[1] for d in range(v_depth)))
    return agent.apply({"params": params}, obs, shared)


# ------------------------------------------------------------------ std transforms (C60, C61, C62)
@jax.custom_jvp
def safe_exp(x):
    """exp(clip(x, -15, 15)) with derivative exp(clip(x, -15, 15)) everywhere (paper footnote 25; official)."""
    return jnp.exp(jnp.clip(x, -15.0, 15.0))


@safe_exp.defjvp
def _safe_exp_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    e = jnp.exp(jnp.clip(x, -15.0, 15.0))
    return e, e * t


def softplus_inv(y):
    return y + jnp.log(-jnp.expm1(-y))


def std_shift(init_std, min_std, softplus):
    """c_rho so that sigma(x_rho = 0) = init_std exactly (official safe_exp_std_fn / softplus_std_fn)."""
    y = init_std - min_std
    return jnp.where(softplus, softplus_inv(y), jnp.log(y))


def std_of(x_rho, shift, min_std, softplus):
    z = x_rho + shift
    return jnp.where(softplus, jax.nn.softplus(z), safe_exp(z)) + min_std


# ------------------------------------------------------------------ distribution (C63)
def tanh_fldj(x):
    """log |d tanh(x)/dx| = 2 (log 2 - x - softplus(-2x)) (numerically stable; TFP Tanh bijector)."""
    return 2.0 * (LOG2 - x - jax.nn.softplus(-2.0 * x))


def sample(key, mu, std, tanh):
    """Stochastic action sent to the env and stored in the batch: tanh(x) (tanh lanes) or clip(x, -1, 1) (clip)."""
    return sample_x(key, mu, std, tanh)[0]


def sample_x(key, mu, std, tanh):
    """(env action, pre-squash sample x). Study-era lanes (presquash) store x, released-HEAD lanes the env action."""
    x = mu + std * jax.random.normal(key, mu.shape, mu.dtype)
    return jnp.where(tanh, jnp.tanh(x), jnp.clip(x, -1.0, 1.0)), x


def mode(mu, tanh):
    return jnp.where(tanh, jnp.tanh(mu), jnp.clip(mu, -1.0, 1.0))


def _gauss_logpdf(x, mu, std):
    return -0.5 * jnp.square((x - mu) / std) - jnp.log(std) - 0.5 * math.log(2.0 * math.pi)


def log_prob(action, mu, std, tanh, presquash):
    """Summed over action dims. presquash (study era, 68c15aa): ``action`` is the raw x, log N(x) - [tanh] log-det.
    Otherwise (released HEAD, 0e1e0ac): tanh = threshold/tail-mass semantics on u; clip = Gaussian at the stored
    (clipped) action."""
    u = jnp.clip(action, -TANH_THRESHOLD, TANH_THRESHOLD)
    x = jnp.arctanh(u)
    inner = _gauss_logpdf(x, mu, std) - tanh_fldj(x)
    left = jnorm.logcdf((-_ATANH_T - mu) / std) - _LOG_EPS_T
    right = jnorm.logcdf((mu - _ATANH_T) / std) - _LOG_EPS_T      # log survival at +atanh(t)
    lp_tanh = jnp.where(u <= -TANH_THRESHOLD, left, jnp.where(u >= TANH_THRESHOLD, right, inner))
    lp_clip = _gauss_logpdf(action, mu, std)
    lp_head = jnp.where(tanh, lp_tanh, lp_clip)
    lp_pre = _gauss_logpdf(action, mu, std) - jnp.where(tanh, tanh_fldj(action), 0.0)
    return jnp.sum(jnp.where(presquash, lp_pre, lp_head), axis=-1)


def entropy(key, mu, std, tanh):
    """Gaussian entropy, plus for tanh the log-det at ONE reparametrised sample (official)."""
    h = 0.5 * math.log(2.0 * math.pi * math.e) + jnp.log(std)
    x = mu + std * jax.random.normal(key, mu.shape, mu.dtype)
    return jnp.sum(h + jnp.where(tanh, tanh_fldj(x), 0.0), axis=-1)


def kl_gauss(mu1, s1, mu2, s2):
    """KL(N(mu1,s1) || N(mu2,s2)) summed over dims (B.6 closed form; used by the toy-test decoupled-KL toy only:
    KL regularisers are out of scope, UNREPRODUCIBLE(budget))."""
    return jnp.sum(jnp.log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2.0 * s2 ** 2) - 0.5, axis=-1)
