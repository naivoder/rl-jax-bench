"""Stream-x estimators and optimizers as pure, jit/vmap-safe functions (paper §3, official branch 2024).

* Welford running mean/variance (Alg. 4, official SampleMeanStd) with Kahan-compensated float32 accumulators
  (: official float64; float32 + compensation is tested against float64 over 5e6 samples).
* NormalizeObservation (Alg. 6) and ScaleReward (Alg. 5, centered variance of the discounted-return trace, ).
* ObGD (Alg. 3): accumulating trace z <- gamma*lambda*z + grad, global L1 norm over ALL params of the network,
  alpha_eff = alpha / max(alpha*kappa*max(|delta|,1)*||z||_1, 1), w <- w + alpha_eff*delta*z.
* Classic-baseline optimizer: the same trace, then Adam on the pseudo-gradient -delta*z (ASSUMED ), torch Adam
  semantics (bias-corrected, eps outside the sqrt), eps = 1e-4 (App. F).
* Adaptive ObGD (App. B, Alg. 11; official AdaptiveObGD with bias correction) -- unit-tested only, never in a figure.

Traces store +grad of the output (the official optimizer stores the grad of -output and subtracts; identical).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ------------------------------------------------------------------ Welford (Kahan-compensated float32)
class Welford(NamedTuple):
    mean: jax.Array
    mean_c: jax.Array   # Kahan compensation of mean
    p: jax.Array        # sum of squared deviations
    p_c: jax.Array
    count: jax.Array    # int32


def welford_init(shape=()) -> Welford:
    z = jnp.zeros(shape, jnp.float32)
    return Welford(z, z, z, z, jnp.int32(0))


def _kadd(s, c, x):
    y = x - c
    t = s + y
    c = (t - s) - y
    return t, c


def welford_update(w: Welford, x) -> Welford:
    """Official SampleMeanStd.update: first sample sets mean = x, p = 0; then the usual recursion."""
    x = jnp.asarray(x, jnp.float32)
    first = w.count == 0
    mean = jnp.where(first, x, w.mean)
    mean_c = jnp.where(first, 0.0, w.mean_c)
    p = jnp.where(first, 0.0, w.p)
    p_c = jnp.where(first, 0.0, w.p_c)
    n = w.count + 1
    d = _dev(x, mean, mean_c)
    new_mean, new_mean_c = _kadd(mean, mean_c, d / n.astype(jnp.float32))
    new_p, new_p_c = _kadd(p, p_c, d * _dev(x, new_mean, new_mean_c))
    return Welford(new_mean, new_mean_c, new_p, new_p_c, n)


def _dev(x, mean, mean_c):
    """x - (compensated mean). The Kahan sum represents mean - mean_c; using the compensation in the deviation keeps
    (x - mean) accurate when |x| >> |x - mean| (e.g. Hopper's torso height 1.25 with sigma ~1e-4 over the first steps:
    plain float32 x - mean lost up to 4e-4 relative vs the official float64 -- closed-loop oracle test, the official-code check)."""
    return (x - mean) + mean_c


def welford_var(w: Welford):
    """var = 1 while count < 2, else p / (count - 1) (unbiased)."""
    return jnp.where(w.count < 2, 1.0, w.p / jnp.maximum(w.count - 1, 1).astype(jnp.float32))


def welford_select(cond, a: Welford, b: Welford) -> Welford:
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


OBS_EPS = 1e-8   # official normalization_wrappers.py
REW_EPS = 1e-8


def normalize_obs(w: Welford, obs):
    """Alg. 6: update the stats with obs FIRST, then (obs - mean) / sqrt(var + eps). Returns (new_w, normalized)."""
    w = welford_update(w, obs)
    return w, _dev(obs, w.mean, w.mean_c) / jnp.sqrt(welford_var(w) + OBS_EPS)


def scale_reward(w: Welford, u, r, done, gamma):
    """Alg. 5 (official ScaleReward): u <- gamma*(1-done)*u + r; update stats with u; return r / sqrt(var_u + eps).
    done = terminated OR truncated of the CURRENT transition."""
    u = u * gamma * (1.0 - done) + r
    w = welford_update(w, u)
    return w, u, r / jnp.sqrt(welford_var(w) + REW_EPS)


# ------------------------------------------------------------------ tree helpers
def tree_l1(t) -> jax.Array:
    return sum(jnp.sum(jnp.abs(x)) for x in jax.tree.leaves(t))


def tree_axpy(a, x, y):
    """y + a * x."""
    return jax.tree.map(lambda xi, yi: yi + a * xi, x, y)


def tree_scale(a, x):
    return jax.tree.map(lambda xi: a * xi, x)


def tree_where(c, a, b):
    return jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)


def tree_zeros(t):
    return jax.tree.map(jnp.zeros_like, t)


# ------------------------------------------------------------------ ObGD
class ObGDInfo(NamedTuple):
    M: jax.Array          # alpha*kappa*delta_bar*||z||_1 (the bound's surrogate effective step size)
    step: jax.Array       # alpha_eff
    z_l1: jax.Array
    dw_l1: jax.Array      # ||Delta w||_1 = alpha_eff*|delta|*||z||_1


def trace_accumulate(trace, grad, gl):
    """z <- gamma*lambda*z + grad."""
    return jax.tree.map(lambda z, g: gl * z + g, trace, grad)


def obgd_apply(params, trace, delta, lr, kappa):
    """Alg. 3 given the already-accumulated trace. Returns (params, info)."""
    z_l1 = tree_l1(trace)
    delta_bar = jnp.maximum(jnp.abs(delta), 1.0)
    M = delta_bar * z_l1 * lr * kappa
    step = jnp.where(M > 1.0, lr / M, lr)
    new = tree_axpy(step * delta, trace, params)
    return new, ObGDInfo(M, step, z_l1, step * jnp.abs(delta) * z_l1)


def obgd_step(params, trace, grad, delta, lr, kappa, gl):
    """Full ObGD step incl. the trace update (official ObGD.step, before any reset). Returns (params, trace, info)."""
    trace = trace_accumulate(trace, grad, gl)
    params, info = obgd_apply(params, trace, delta, lr, kappa)
    return params, trace, info


# ------------------------------------------------------------------ Adam on -delta*z (classic baselines)
class AdamState(NamedTuple):
    m: dict
    v: dict
    t: jax.Array   # int32 step count


ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-4   # App. F ("default betas, eps 1e-4")


def adam_init(params) -> AdamState:
    return AdamState(tree_zeros(params), tree_zeros(params), jnp.int32(0))


def adam_apply(params, st: AdamState, grad_loss, lr, b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS):
    """torch.optim.Adam step on a loss gradient: p -= lr * mhat / (sqrt(vhat) + eps)."""
    t = st.t + 1
    tf = t.astype(jnp.float32)
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, st.m, grad_loss)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, st.v, grad_loss)
    bc1 = 1 - b1 ** tf
    bc2 = 1 - b2 ** tf
    upd = jax.tree.map(lambda m_, v_: (m_ / bc1) / (jnp.sqrt(v_ / bc2) + eps), m, v)
    new = jax.tree.map(lambda p, u: p - lr * u, params, upd)
    return new, AdamState(m, v, t), tree_l1(jax.tree.map(lambda u: lr * u, upd))


# ------------------------------------------------------------------ Adaptive ObGD (App. B, Alg. 11; unit-test only)
class AdaptiveState(NamedTuple):
    v: dict
    t: jax.Array


def adaptive_obgd_step(params, trace, st: AdaptiveState, grad, delta, lr, kappa, gl, beta2=0.999, eps=1e-8):
    """Official AdaptiveObGD.step: v <- b2 v + (1-b2)(delta z)^2, vhat = v/(1-b2^t), zhat = z/sqrt(vhat+eps)."""
    t = st.t + 1
    trace = trace_accumulate(trace, grad, gl)
    v = jax.tree.map(lambda v_, z: beta2 * v_ + (1 - beta2) * (delta * z) ** 2, st.v, trace)
    bc = 1 - beta2 ** t.astype(jnp.float32)
    zhat = jax.tree.map(lambda z, v_: z / jnp.sqrt(v_ / bc + eps), trace, v)
    z_l1 = tree_l1(zhat)
    M = jnp.maximum(jnp.abs(delta), 1.0) * z_l1 * lr * kappa
    step = jnp.where(M > 1.0, lr / M, lr)
    return tree_axpy(step * delta, zhat, params), trace, AdaptiveState(v, t), step


# ------------------------------------------------------------------ exploration
def linear_schedule(start, end, duration, t):
    """Official linear_schedule: max(slope*t + start, end), slope = (end-start)/duration."""
    slope = (end - start) / duration
    return jnp.maximum(slope * t + start, end)


# ------------------------------------------------------------------ Gaussian policy terms (stream AC, App. E)
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def gaussian_logp(a, mu, std):
    """torch Normal.log_prob summed over action dims (of the UNclipped sample, )."""
    return jnp.sum(-((a - mu) ** 2) / (2 * std ** 2) - jnp.log(std) - LOG_SQRT_2PI)


def gaussian_entropy(std):
    return jnp.sum(0.5 + LOG_SQRT_2PI + jnp.log(std))
