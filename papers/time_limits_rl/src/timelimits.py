"""Float64 numpy reference helpers for time-limit analysis (no jax).

* ``gae_two_flag``: a reference GAE that tells a TERMINATION (``terms``, V(s') := 0) from an episode RESET for any
  reason (``resets``, trace cut). ``peb=True`` bootstraps through time-limit truncations (paper Eq. 6);
  ``peb=False`` treats every reset as terminal (the standard agent, Eq. 4) -- i.e. a single-``done`` GAE.
  ``final_values[t]`` is V of the TRUE (pre-reset) next observation of transition t. It equals the JAX
  ``src/algorithm.py::gae_peb`` (pinned by tests/test_timelimits.py).
* ``aliasing_cost``: R^2 of a leave-one-row-out bin-mean lookup of the return on one time variable minus that on a
  coarser one -- the share of return variance a critic that sees only the coarser time cannot represent. Rows are
  independent environments.
* ``boundary_windows`` / ``adv_share``: the share of (centered) advantage variance carried by the K slots ending each
  episode, the statistic that shows a critic's level error at a time limit being paid by the last actions.
"""
from __future__ import annotations

import numpy as np


def gae_two_flag(rewards, values, final_values, terms, resets, gamma: float, lam: float, peb: bool = True):
    """Arrays (M, ...) with time on axis 0. Returns (advantages, value targets = A + V), float64."""
    r = np.asarray(rewards, np.float64)
    v = np.asarray(values, np.float64)
    fv = np.asarray(final_values, np.float64)
    te = np.asarray(terms, np.float64)
    rs = np.asarray(resets, np.float64)
    term_eff = te if peb else rs
    delta = r + gamma * (1.0 - term_eff) * fv - v
    adv = np.zeros_like(v)
    nxt = np.zeros_like(v[0])
    for t in range(v.shape[0] - 1, -1, -1):
        nxt = delta[t] + gamma * lam * (1.0 - rs[t]) * nxt
        adv[t] = nxt
    return adv, adv + v


def loo_bin_mean(G, live, key, n_keys: int, fallback: float):
    """Leave-one-ROW-out mean of G per integer bin ``key``; G, live, key are (N, T) with rows = environments."""
    G = np.asarray(G, np.float64)
    live = np.asarray(live, bool)
    N = G.shape[0]
    k = np.where(live, key, 0).astype(np.int64)
    S = np.zeros((N, n_keys))
    C = np.zeros((N, n_keys))
    for e in range(N):
        m = live[e]
        S[e] = np.bincount(k[e][m], weights=G[e][m], minlength=n_keys)
        C[e] = np.bincount(k[e][m], minlength=n_keys)
    St, Ct = S.sum(0), C.sum(0)
    rows = np.arange(N)[:, None]
    s_loo = St[k] - S[rows, k]
    c_loo = Ct[k] - C[rows, k]
    return np.where(c_loo > 0, s_loo / np.maximum(c_loo, 1), fallback)


def lookup_r2(g, b) -> float:
    """1 - Var(g - b) / Var(g) (explained variance of a baseline b for returns g)."""
    g = np.asarray(g, np.float64)
    return float(1.0 - np.var(g - np.asarray(b, np.float64)) / np.var(g))


def aliasing_cost(G, live, fine_time, coarse_time, bin_width: float):
    """R^2(LOO lookup on fine_time) - R^2(LOO lookup on coarse_time), plus both R^2 values.

    fine_time: the time variable a time-aware critic would need;
    coarse_time: the one the critic has (e.g. time within the episode). Bins of ``bin_width``."""
    G = np.asarray(G, np.float64)
    live = np.asarray(live, bool)
    gm = G[live].mean()
    kf = (np.asarray(fine_time) // bin_width).astype(np.int64)
    kc = (np.asarray(coarse_time) // bin_width).astype(np.int64)
    nf = int(np.max(np.where(live, kf, 0))) + 1
    nc = int(np.max(np.where(live, kc, 0))) + 1
    bf = loo_bin_mean(G, live, kf, nf, gm)
    bc = loo_bin_mean(G, live, kc, nc, gm)
    r2f, r2c = lookup_r2(G[live], bf[live]), lookup_r2(G[live], bc[live])
    return dict(A_alias=r2f - r2c, R2_fine=r2f, R2_coarse=r2c)


def boundary_windows(done, K: int):
    """bool (N, T): the K slots ending at (and including) each done slot of the same row."""
    done = np.asarray(done) > 0
    W = np.zeros(done.shape, bool)
    for j in range(K):
        W[:, :done.shape[1] - j] |= done[:, j:]
    return W


def adv_share(A, W) -> float:
    """Share of the centered advantage variance carried by the slots in mask W."""
    c = np.asarray(A, np.float64)
    c = c - c.mean()
    return float(np.sum(c[W] ** 2) / np.sum(c ** 2))
