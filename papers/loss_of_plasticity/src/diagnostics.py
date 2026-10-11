"""Plasticity diagnostics as reusable, jittable JAX functions.

All functions take plain arrays, carry no state, and vmap over a leading run axis.

* ``effective_rank(phi)``      Roy & Vetterli (2007) / paper Eq. 1: exp(entropy of the normalised singular values),
                               all min(N, n) singular values, 0 log 0 = 0 (official
                               utils/miscellaneous.py compute_effective_rank).
* ``scaled_effective_rank``    100 * erank / n_units  (paper "scaled to [0, 100]"; official plots/all_metrics.py
                               divides the layer sum by 3 * 2000 / 100 = mean over layers of 100 * erank_l / n_l).
* ``dead_units(h)``            unit is dead iff sum_x |h_i(x)| == 0 over the sample (paper §4; online_expr.py).
* ``saturated_units(h, lo, hi, eps)``  tanh/sigmoid analogue: every output within eps*(hi-lo) of an extreme
                               (ASSUMED; diagnostic only).
* ``mean_abs_weight(kernels)`` average |w| over all weight matrices, biases excluded (paper Fig. 3 / 6b).
* ``layer_diagnostics(hs)``    dict of the above per hidden layer for a list of (N, n_l) activation matrices.
"""
from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp


def singular_values(phi):
    return jnp.linalg.svd(phi, compute_uv=False)


def effective_rank_from_sv(sv):
    sv = jnp.abs(sv)
    p = sv / jnp.sum(sv)
    ent = -jnp.sum(jnp.where(p > 0, p * jnp.log(jnp.where(p > 0, p, 1.0)), 0.0))
    return jnp.exp(ent)


def effective_rank(phi):
    """phi: (N, n) feature matrix -> effective rank (scalar)."""
    return effective_rank_from_sv(singular_values(phi))


def scaled_effective_rank(phi):
    return 100.0 * effective_rank(phi) / phi.shape[-1]


def dead_units(h):
    """h: (N, n) -> bool (n,) dead mask (output exactly 0 on every sample)."""
    return jnp.sum(jnp.abs(h), axis=0) == 0


def saturated_units(h, lo: float, hi: float, eps: float = 0.01):
    """h: (N, n) -> bool (n,) saturated mask: all samples within eps*(hi-lo) of lo, or all within it of hi."""
    tol = eps * (hi - lo)
    return jnp.all(h <= lo + tol, axis=0) | jnp.all(h >= hi - tol, axis=0)


def mean_abs_weight(kernels: Sequence):
    return sum(jnp.sum(jnp.abs(k)) for k in kernels) / sum(k.size for k in kernels)


def layer_diagnostics(hs: Sequence):
    """hs: list of (N, n_l) activations -> per-layer arrays and network aggregates (official definitions)."""
    dead = jnp.stack([jnp.sum(dead_units(h)) for h in hs]).astype(jnp.float32)
    er = jnp.stack([effective_rank(h) for h in hs])
    widths = jnp.asarray([h.shape[-1] for h in hs], jnp.float32)
    return {
        "dead_per_layer": dead,
        "erank_per_layer": er,
        "dead_pct": 100.0 * jnp.sum(dead) / jnp.sum(widths),
        "erank_scaled": jnp.mean(100.0 * er / widths),
    }
