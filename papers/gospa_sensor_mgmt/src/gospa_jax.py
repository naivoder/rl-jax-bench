"""gospa_jax.py -- GOSPA (paper eq. 18, p = 2, alpha = 2) in JAX for small fixed-size sets, with the loc / missed / false split.

Formulation (exact): d^2(X, Y) = c^2/2 (|X| + |Y|) + min_gamma sum_{(i,j) in gamma} (d_ij^2 - c^2); only pairs with d_ij^2 < c^2 can
help.  The minimization is a dynamic program over the SET OF TARGETS already used (2^K states, K = 5 compacted truth slots)
scanned over the estimate slots, so it is jit/vmap friendly and exact (no Hungarian needed for |truth| <= K).  The split is the
authors' (gospapy): loc = sum of d^2 over assigned pairs, missed = c^2/2 (#unassigned truths), false = c^2/2 (#unassigned
estimates); total = loc + missed + false.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def _masks(K: int):
    """For each target t: (source masks without bit t, destination masks with bit t)."""
    allm = np.arange(2 ** K)
    src, dst = [], []
    for t in range(K):
        s = allm[(allm >> t) & 1 == 0]
        src.append(s)
        dst.append(s | (1 << t))
    return src, dst


def gospa_parts(tx, tmask, ex, emask, c: float, idx=None):
    """GOSPA^2 and parts between truth set tx[K, D] (valid tmask[K]) and estimate set ex[E, D] (valid emask[E]).

    idx: coordinate indices entering the single-target distance (None = all of D).  Returns (total, loc, missed, false).
    Requires the number of valid truths <= K (K = tx.shape[0]).
    """
    K = tx.shape[0]
    E = ex.shape[0]
    if idx is not None:
        tx = tx[:, idx]
        ex = ex[:, idx]
    d2 = jnp.sum((tx[:, None, :] - ex[None, :, :]) ** 2, axis=-1)          # [K, E]
    c2 = c * c
    gain = d2 - c2
    allow = tmask[:, None] & emask[None, :] & (d2 < c2)
    src, dst = _masks(K)
    big = jnp.asarray(1e30, tx.dtype)
    obj0 = jnp.full((2 ** K,), big, tx.dtype).at[0].set(0.0)
    loc0 = jnp.zeros((2 ** K,), tx.dtype)
    np0 = jnp.zeros((2 ** K,), tx.dtype)

    def step(carry, e):
        obj, loc, npair = carry
        o2, l2, n2 = obj, loc, npair
        for t in range(K):
            s, d = src[t], dst[t]
            ok = allow[t, e]
            cand = obj[s] + gain[t, e]
            better = ok & (cand < o2[d]) & (obj[s] < big / 2)
            o2 = o2.at[d].set(jnp.where(better, cand, o2[d]))
            l2 = l2.at[d].set(jnp.where(better, loc[s] + d2[t, e], l2[d]))
            n2 = n2.at[d].set(jnp.where(better, npair[s] + 1.0, n2[d]))
        return (o2, l2, n2), None

    (obj, loc, npair), _ = jax.lax.scan(step, (obj0, loc0, np0), jnp.arange(E))
    i = jnp.argmin(obj)
    nt = jnp.sum(tmask).astype(tx.dtype)
    ne = jnp.sum(emask).astype(tx.dtype)
    npr = npair[i]
    l = loc[i]
    missed = 0.5 * c2 * (nt - npr)
    false = 0.5 * c2 * (ne - npr)
    return l + missed + false, l, missed, false
