"""Lagrangian dual lower bound V^D of the paper (§III-A,B, eqs 11-20) for the SCALAR model, used only by Figs 5-6.

    v^lam(P)   = min_a { d P + lam a + beta v^lam(phi^a(P)) }                      (16)
    V^L(lam;P0) = sum_n v_n^lam(P_{n,0}) - K lam / (1 - beta)                      (13),(17)
    V^D(P0)    = max_{lam >= 0} V^L(lam; P0)                                        (14),(20)   [concave in lam]

Two independent computation paths (two-path agreement):
  path A  `vi_dual`        value iteration on a P grid + golden-section maximisation over lam
  path B  `threshold_dual` v^lam(P0) = min over threshold policies z of F(P0,z) + lam G(P0,z) by deterministic rollouts
                           (tau = 400); exact iff threshold policies are optimal (PCL-indexability), and in general
                           an UPPER value of the single-target problem: path B >= path A, equality = threshold optimality.

Out-of-grid next states (passive growth beyond the cap): v(x) is replaced by the 'track at x' value d x + lam + beta v(phi^1(x))
(phi^1(x) <= R lies inside the grid).  This keeps every interpolation a convex combination (a contraction) and is exact whenever
tracking is optimal above the cap; the visited-state check (`visited_max`) logs whether the optimal policy ever gets near it.
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import model as M
from model import Bank

P_CAP = 1e6                  # ASSUMED
GRID_UNIFORM_TO = 20.0       # ASSUMED: uniform spacing 0.01 on (0, 20] (paper §IV-A spacing 1e-2)
GRID_STEP = 0.01             # paper §IV-A (1e-2); extent ASSUMED
GRID_RATIO = 1.002           # ASSUMED: geometric tail
VI_TOL = 1e-9                # ASSUMED
VI_MAX_SWEEPS = 5000         # ASSUMED
LAMBDA_TOL = 1e-8            # ASSUMED
ORACLE_TAU = 400             # ASSUMED


def make_grid() -> np.ndarray:
    uni = np.arange(GRID_STEP, GRID_UNIFORM_TO + 1e-9, GRID_STEP)
    n_geo = int(np.ceil(np.log(P_CAP / GRID_UNIFORM_TO) / np.log(GRID_RATIO)))
    geo = GRID_UNIFORM_TO * GRID_RATIO ** np.arange(1, n_geo + 1)
    return np.concatenate([uni, geo])


class Tables(NamedTuple):
    grid: jnp.ndarray     # (G,)
    c: jnp.ndarray        # (Tp,G)   d * P_i
    j: jnp.ndarray        # (2,Tp,G) lower interpolation index of phi^a(P_i)  (a = 0 uses the in-grid part only)
    w: jnp.ndarray        # (2,Tp,G) interpolation weight
    out0: jnp.ndarray     # (Tp,G)   passive next state above the cap
    jo: jnp.ndarray       # (Tp,G)   index of phi^1(phi^0(P_i)) for out-of-grid next states
    wo: jnp.ndarray       # (Tp,G)
    co: jnp.ndarray       # (Tp,G)   d * phi^0(P_i) for out-of-grid next states


def unique_problems(types, qct, d):
    """Distinct (type, Q^CT, d) triples and the target -> problem id map."""
    keys = list(zip(types, [float(q) for q in qct], [float(x) for x in d]))
    uniq = sorted(set(keys))
    tid = np.array([uniq.index(k) for k in keys], np.int64)
    return uniq, tid


def build_tables(uniq) -> Tables:
    grid = make_grid()
    bank = M.scalar_bank([u[0] for u in uniq], [u[1] for u in uniq], [u[2] for u in uniq])
    Tp = len(uniq)
    d = np.array([u[2] for u in uniq])
    g = jnp.asarray(grid)

    def nxt(act, x):  # x (Tp,G) -> phi^act(x)
        f = lambda col: M.phi(bank, col[:, None, None], jnp.full(Tp, act))[:, 0, 0]  # noqa: E731
        return jax.vmap(f, in_axes=1, out_axes=1)(x)

    X = jnp.broadcast_to(g, (Tp, grid.size))
    x0, x1 = nxt(0.0, X), nxt(1.0, X)

    def interp_idx(x):
        j = jnp.clip(jax.vmap(lambda xi: jnp.searchsorted(g, xi, side="right") - 1)(x.reshape(-1)).reshape(x.shape), 0, grid.size - 2)
        w = (x - g[j]) / (g[j + 1] - g[j])
        return j, w

    out0 = x0 > grid[-1]
    j0, w0 = interp_idx(jnp.where(out0, grid[-1], x0))
    j1, w1 = interp_idx(x1)
    xo1 = nxt(1.0, jnp.where(out0, x0, grid[-1]))
    jo, wo = interp_idx(xo1)
    return Tables(g, jnp.asarray(d)[:, None] * X, jnp.stack([j0, j1]), jnp.stack([w0, w1]), out0, jo, wo,
                  jnp.asarray(d)[:, None] * x0)


def _take(v, j):
    return jnp.take_along_axis(v, jnp.broadcast_to(j, v.shape), axis=-1)


@functools.partial(jax.jit, static_argnames=("beta", "tol", "max_sweeps"))
def vi(tb: Tables, lam, v0, beta: float = M.BETA, tol: float = VI_TOL, max_sweeps: int = VI_MAX_SWEEPS):
    """Value iteration for every (batch b, problem p): v (B,Tp,G); lam (B,).  Returns (v, sweeps, last change, greedy a (B,Tp,G))."""
    lam_ = lam[:, None, None]

    def branches(v):
        def interp(j, w):
            return (1 - w) * _take(v, j) + w * _take(v, j + 1)
        q1 = tb.c + lam_ + beta * interp(tb.j[1], tb.w[1])
        v_out = tb.co + lam_ + beta * interp(tb.jo, tb.wo)            # track-at-x value for next states above the cap
        q0 = tb.c + beta * jnp.where(tb.out0, v_out, interp(tb.j[0], tb.w[0]))
        return q0, q1

    def cond(s):
        _, k, ch = s
        return (k < max_sweeps) & (ch > 0.0)

    def body(s):
        v, k, _ = s
        q0, q1 = branches(v)
        vn = jnp.minimum(q0, q1)
        ch = jnp.max(jnp.abs(vn - v))
        # stop (ch -> 0) once the sup-norm change is below tol * max|v|
        ch = jnp.where(ch <= tol * jnp.max(jnp.abs(vn)), 0.0, ch)
        return vn, k + 1, ch

    v, k, ch = jax.lax.while_loop(cond, body, (v0, jnp.zeros((), jnp.int32), jnp.ones(())))
    q0, q1 = branches(v)
    return v, k, ch, (q1 < q0).astype(jnp.int8)


def _interp_P0(tb: Tables, v, P0, tid):
    """v (B,Tp,G) at the initial states: P0 (B,N) -> per-target v (B,N) (linear interpolation / extrapolation at the grid ends)."""
    g = tb.grid
    j = jnp.clip(jnp.searchsorted(g, P0, side="right") - 1, 0, g.size - 2)
    w = (P0 - g[j]) / (g[j + 1] - g[j])
    vt = v[:, tid, :]                                                 # (B,N,G)
    return (1 - w) * jnp.take_along_axis(vt, j[..., None], -1)[..., 0] + w * jnp.take_along_axis(vt, (j + 1)[..., None], -1)[..., 0]


def vi_dual(types, qct, d, P0, K: int, beta: float = M.BETA, iters: int = 80):
    """Path A.  P0 (R,N) (a replication axis; each replication gets its own lam*).  Returns dict with
    V_D (R,) = V^L(lam*;P0), lam (R,), pvis (R,) = max state visited by the lam*-greedy policy from P0 (cap check)."""
    uniq, tid = unique_problems(types, qct, d)
    tb = build_tables(uniq)
    R, N = P0.shape
    Tp, G = len(uniq), tb.grid.size
    tid_j = jnp.asarray(tid)
    P0 = jnp.asarray(P0)

    def VL(lam, v_init):
        v, k, ch, a = vi(tb, lam, v_init, beta)
        val = jnp.sum(_interp_P0(tb, v, P0, tid_j), axis=1) - K * lam / (1 - beta)
        return val, v, a

    v = jnp.zeros((R, Tp, G))
    # bracket the maximiser: V^L is concave in lam; grow hi until V^L(hi) < V^L(hi/2)
    hi = jnp.full(R, 1.0)
    prev, v, _ = VL(hi / 2, v)
    cur, v, _ = VL(hi, v)
    for _ in range(60):
        grow = cur >= prev
        if not bool(jnp.any(grow)):
            break
        hi = jnp.where(grow, hi * 2, hi)
        prev = jnp.where(grow, cur, prev)
        cur_new, v, _ = VL(hi, v)
        cur = jnp.where(grow, cur_new, cur)
    a, b = jnp.zeros(R), hi
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a); dd = a + gr * (b - a)
    fc, vc, _ = VL(c, v)
    fd, vd, _ = VL(dd, vc)
    n_it = 0
    for n_it in range(iters):
        left = fc > fd                                   # maximiser lies in [a, d]
        a_n = jnp.where(left, a, c); b_n = jnp.where(left, dd, b)
        c_n = jnp.where(left, b_n - gr * (b_n - a_n), dd)
        d_n = jnp.where(left, c, a_n + gr * (b_n - a_n))
        new_pt = jnp.where(left, c_n, d_n)
        f_new, v_new, _ = VL(new_pt, jnp.where(left[:, None, None], vc, vd))
        fc_n = jnp.where(left, f_new, fd); vc_n = jnp.where(left[:, None, None], v_new, vd)
        fd_n = jnp.where(left, fc, f_new);  vd_n = jnp.where(left[:, None, None], vc, v_new)
        a, b, c, dd, fc, fd, vc, vd = a_n, b_n, c_n, d_n, fc_n, fd_n, vc_n, vd_n
        if bool(jnp.all((b - a) <= LAMBDA_TOL * jnp.maximum(1.0, b))):
            break
    best_c = fc > fd
    lam = jnp.where(best_c, c, dd)
    V_D, v, act = VL(lam, jnp.where(best_c[:, None, None], vc, vd))
    pvis = visited_max(tb, act, P0, tid_j, uniq)
    return {"V_D": V_D, "lam": lam, "pvis": pvis, "n_golden": n_it + 1, "n_problems": Tp, "grid_size": G}


def visited_max(tb: Tables, act, P0, tid, uniq, steps: int = 400):
    """Largest state visited by the greedy (lam*-optimal) policy of each replication: the P_cap check."""
    bank = M.scalar_bank([u[0] for u in uniq], [u[1] for u in uniq], [u[2] for u in uniq])
    bank_n = Bank(*[x[tid] if x.ndim > 1 and x.shape[0] == len(uniq) else x for x in bank])
    g = tb.grid

    def one(a_r, p0):                                 # a_r (Tp,G), p0 (N,)
        def body(carry, _):
            p, mx = carry
            j = jnp.clip(jnp.searchsorted(g, p, side="right") - 1, 0, g.size - 1)
            aa = a_r[tid, j].astype(p.dtype)
            return (M.phi(bank_n, p[:, None, None], aa)[:, 0, 0], jnp.maximum(mx, jnp.max(p))), None
        (_, mx), _ = jax.lax.scan(body, (p0, jnp.zeros(())), None, length=steps)
        return mx
    return jax.vmap(one)(act, P0)


# ----------------------------------------------------------------------------------------------- path B: threshold-min dual
def threshold_dual(types, qct, d, P0, K: int, beta: float = M.BETA, tau: int = ORACLE_TAU, n_z: int = 4000, iters: int = 200):
    """Path B.  For each distinct target problem and each replication's P0 evaluate F(P0,z), G(P0,z) of the z-threshold policy
    (action at k = 0 also by the threshold) for z in a log grid plus {always, never}; V^L(lam) = sum_n min_z (F + lam G) - K lam/(1-beta);
    maximised over lam (concave, piecewise linear) by dense search + golden refinement in numpy.  P0 (R,N)."""
    import algorithm as A
    uniq, tid = unique_problems(types, qct, d)
    R, N = P0.shape
    zs = np.concatenate([[-1.0], np.geomspace(1e-3, 1e4, n_z), [1e30]])
    bank = M.scalar_bank([types[i] for i in range(N)], list(qct), list(d))

    @jax.jit
    def FG_all(p0):                                  # p0 (N,) -> F,G (Z,N)
        def one(z):
            zz = jnp.full(N, z)
            P = p0[:, None, None]
            a0 = (M.stat(P) > zz).astype(p0.dtype)
            return A.rollout_FG(bank, P, zz, a0, tau, beta)
        return jax.lax.map(one, jnp.asarray(zs))
    F = np.empty((R, len(zs), N)); Gw = np.empty_like(F)
    for r in range(R):
        f, g = FG_all(jnp.asarray(P0[r]))
        F[r], Gw[r] = np.asarray(f), np.asarray(g)
    out_V, out_lam = np.empty(R), np.empty(R)
    for r in range(R):
        def VLr(lam):
            return float(np.sum(np.min(F[r] + lam * Gw[r], axis=0)) - K * lam / (1 - beta))
        hi = 1.0
        while VLr(hi) >= VLr(hi / 2) and hi < 1e12:
            hi *= 2
        a, b = 0.0, hi
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a); dd = a + gr * (b - a); fc, fd = VLr(c), VLr(dd)
        for _ in range(iters):
            if fc > fd:
                b, dd, fd = dd, c, fc; c = b - gr * (b - a); fc = VLr(c)
            else:
                a, c, fc = c, dd, fd; dd = a + gr * (b - a); fd = VLr(dd)
            if b - a <= LAMBDA_TOL * max(1.0, b):
                break
        lam = c if fc > fd else dd
        out_lam[r], out_V[r] = lam, max(fc, fd)
    return {"V_D": out_V, "lam": out_lam}
