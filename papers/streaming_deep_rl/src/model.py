"""Networks for the stream-x agents (paper §3.3-3.4, App. F).

Every network acts on ONE unbatched sample (streaming: batch size 1); replications are added with jax.vmap.
flax.linen modules; the stream-x components are switched per lane by DYNAMIC flags so that all arms of one
experiment run in ONE vmapped program and differ only in the mechanism under test:

* ``use_ln`` (0/1 float): parameter-free LayerNorm over ALL axes of the layer output (torch
  ``F.layer_norm(x, x.size())``, eps 1e-5), applied before every hidden LeakyReLU, never on outputs.
* initialization (``init_params``): SparseInit (Alg. 1; official sparse_init.py: per output unit an independent
  random set of ceil(0.9 fan_in) incoming weights is zeroed, the rest U(+-1/sqrt(fan_in)), biases 0), dense
  U(+-1/sqrt(fan_in)) with zero bias (the no-SparseInit ablation arm, ASSUMED), or the torch default
  (kaiming_uniform(a=sqrt 5) => W, b ~ U(+-1/sqrt(fan_in)); the classic baselines, ASSUMED B1).

Layouts: flax Dense kernel = (fan_in, fan_out) = torch weight.T; Conv kernel = (kh, kw, C_in, C_out) and the conv
output is flattened in (H, W, C) order (torch flattens (C, H, W)) -- a fixed permutation of FC1's input rows, so the
differential tests permute the torch weights (tests/test_official_parity.py).
"""
from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

LN_EPS = 1e-5          # official code (torch F.layer_norm default); paper §3.3 "small"
LEAKY_SLOPE = 0.01     # torch nn.LeakyReLU default; ASSUMED (paper silent)
SPARSITY = 0.9         # Appendix F.1-F.4


def layer_norm(x: jax.Array, eps: float = LN_EPS) -> jax.Array:
    """(x - mean) / sqrt(var + eps) over every axis of one sample, biased variance (paper Eq. 4)."""
    mu = jnp.mean(x)
    var = jnp.mean(jnp.square(x - mu))
    return (x - mu) * jax.lax.rsqrt(var + eps)


def maybe_ln(x: jax.Array, use_ln) -> jax.Array:
    return jnp.where(use_ln > 0.5, layer_norm(x), x)


def leaky_relu(x: jax.Array) -> jax.Array:
    """torch.nn.functional.leaky_relu, INCLUDING its derivative at exactly 0 (= slope: torch tests x > 0).
    The kink matters: the first observation of a run normalizes to exactly 0 (Welford's first sample), so every
    pre-activation is exactly 0 on the first update; and without LN, zero inputs x zero bias give exact zeros."""
    return jnp.where(x > 0, x, LEAKY_SLOPE * x)


def softplus_torch(x: jax.Array) -> jax.Array:
    """torch.nn.functional.softplus (beta 1, threshold 20: linear above 20) -- App. F.4 std head."""
    return jnp.where(x > 20.0, x, jnp.log1p(jnp.exp(jnp.minimum(x, 20.0))))


# ------------------------------------------------------------------ networks
class QNetMinAtar(nn.Module):
    """Conv16@3x3 s1 VALID -> LN(8x8x16) -> LReLU -> flatten 1024 -> FC128 -> LN -> LReLU -> FC|A|
    (App. F.2 + official stream_q_minatar.py; no extra FC1024)."""
    n_actions: int
    hidden: int = 128

    @nn.compact
    def __call__(self, x, use_ln):
        x = nn.Conv(16, (3, 3), strides=1, padding="VALID", name="conv")(x)
        x = leaky_relu(maybe_ln(x, use_ln))
        x = x.reshape(-1)
        x = nn.Dense(self.hidden, name="fc1")(x)
        x = leaky_relu(maybe_ln(x, use_ln))
        return nn.Dense(self.n_actions, name="out")(x)


class MLPTrunk(nn.Module):
    hidden: int = 128

    @nn.compact
    def __call__(self, x, use_ln):
        x = leaky_relu(maybe_ln(nn.Dense(self.hidden, name="fc1")(x), use_ln))
        x = leaky_relu(maybe_ln(nn.Dense(self.hidden, name="fc2")(x), use_ln))
        return x


class ValueNet(nn.Module):
    """Critic / stream-TD value net: MLP 128-128 LN, scalar output (App. F.1, F.4)."""
    hidden: int = 128

    @nn.compact
    def __call__(self, x, use_ln):
        h = MLPTrunk(self.hidden, name="trunk")(x, use_ln)
        return nn.Dense(1, name="out")(h)[0]


class GaussianActor(nn.Module):
    """Actor: MLP 128-128 LN, mean head + softplus std head (App. F.4; official Actor)."""
    n_actions: int
    hidden: int = 128

    @nn.compact
    def __call__(self, x, use_ln):
        h = MLPTrunk(self.hidden, name="trunk")(x, use_ln)
        mu = nn.Dense(self.n_actions, name="mu")(h)
        std = softplus_torch(nn.Dense(self.n_actions, name="std")(h))
        return mu, std


# ------------------------------------------------------------------ initialization
INIT_SPARSE, INIT_DENSE0, INIT_TORCH = 0, 1, 2


def _fan_in(kernel_shape) -> int:
    return int(math.prod(kernel_shape[:-1]))  # Dense (in, out): in; Conv (kh, kw, cin, out): kh*kw*cin


def sparse_kernel(key, shape, sparsity: float = SPARSITY, dtype=jnp.float32):
    """SparseInit for one kernel: U(+-1/sqrt(fan_in)), then for EACH output unit an independent uniformly random
    subset of ceil(s * fan_in) incoming weights is set to 0 (official sparse_init.py; )."""
    fan_in = _fan_in(shape)
    fan_out = shape[-1]
    n_zero = int(math.ceil(sparsity * fan_in))
    ku, kp = jax.random.split(key)
    bound = 1.0 / math.sqrt(fan_in)
    w = jax.random.uniform(ku, (fan_in, fan_out), dtype, -bound, bound)
    # rank of a uniform key per (unit, input) = a uniform random permutation per output unit
    ranks = jnp.argsort(jnp.argsort(jax.random.uniform(kp, (fan_out, fan_in)), axis=1), axis=1)
    keep = (ranks >= n_zero).T  # (fan_in, fan_out)
    return jnp.where(keep, w, 0.0).reshape(shape)


def dense_kernel(key, shape, dtype=jnp.float32):
    bound = 1.0 / math.sqrt(_fan_in(shape))
    return jax.random.uniform(key, shape, dtype, -bound, bound)


def _layer_items(params):
    """Sorted list of (path, layer_dict) for every dict that holds a 'kernel'."""
    out = []

    def walk(d, path):
        if "kernel" in d:
            out.append((path, d))
            return
        for k in sorted(d):
            walk(d[k], path + (k,))

    walk(params, ())
    return out


def init_params(key, template, scheme: int):
    """Re-initialize every layer of a flax param tree (template from module.init) with one scheme.
    Layer i (sorted path order) uses fold_in(key, i) -> split(2) = (kernel key, bias key)."""
    params = jax.tree.map(lambda x: x, template)
    for i, (path, layer) in enumerate(_layer_items(template)):
        kk, kb = jax.random.split(jax.random.fold_in(key, i))
        shape = layer["kernel"].shape
        if scheme == INIT_SPARSE:
            k = sparse_kernel(kk, shape)
            b = jnp.zeros_like(layer["bias"])
        elif scheme == INIT_DENSE0:
            k = dense_kernel(kk, shape)
            b = jnp.zeros_like(layer["bias"])
        elif scheme == INIT_TORCH:
            k = dense_kernel(kk, shape)
            bound = 1.0 / math.sqrt(_fan_in(shape))
            b = jax.random.uniform(kb, layer["bias"].shape, jnp.float32, -bound, bound)
        else:
            raise ValueError(scheme)
        d = params
        for p in path[:-1]:
            d = d[p]
        d[path[-1]] = {"kernel": k, "bias": b}
    return params


def init_select(key, template, scheme_id):
    """Per-lane dynamic choice among the three schemes (all computed, one selected): keeps every arm in one program."""
    trees = [init_params(key, template, s) for s in (INIT_SPARSE, INIT_DENSE0, INIT_TORCH)]
    return jax.tree.map(lambda a, b, c: jnp.where(scheme_id == 0, a, jnp.where(scheme_id == 1, b, c)), *trees)


def template(module: nn.Module, obs_shape) -> dict:
    """Unbatched param template (values irrelevant; init_params overwrites all of them)."""
    return module.init(jax.random.PRNGKey(0), jnp.zeros(obs_shape, jnp.float32), 1.0)["params"]
