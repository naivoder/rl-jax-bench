"""filter.py -- Gaussian multi-Bernoulli (MB) filter with sequential per-sensor PMB updates (paper SS II-B, eqs. 3-17, SS V-C).

Fixed-shape, float64, jit/vmap friendly.  One filter lane = (r[NB], x[NB,4], P[NB,4,4], n); slots >= n are inert (r = 0,
x = 0, P = I).  Per step: predict (+ birth Bernoulli appended at slot n) -> [planner] -> for each sensor in turn: Bernoulli
update with data association marginals by loopy belief propagation (LBP, [40]) and MB projection (TOMB, eqs. 15-17) ->
merge (Mahalanobis < 1) -> prune -> region -> estimate.  The CFG-O branches replicate
the authors' behavior where it differs from the text (README differences D4/D20/D21/D22/D30), the CFG-P branches follow the text.

Official-code facts replicated in CFG-O (verified by tests/test_official_parity.py):
  * the Bernoulli detection probability is evaluated per BERNOULLI at its mean (hard 40 m FOV), never per measurement;
  * clutter weight wnew = lambda_C / (pi 40^2) for every measurement column; LBP messages mu, nu as in lbp.py;
  * D30 (found while implementing): lbp.py aliases `mu_old = mu`, so its convergence loop runs EXACTLY ONE sweep ('one_sweep');
  * TOMB prunes r <= 1e-4 (new-track hypotheses carry r = 0 and are all dropped); merge is greedy in index order with the
    leader's full 4-D covariance; weights are the existence probabilities and r = sum r (capped only when cfg says so).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from src import model as M
from src.model import Cfg

NB = M.NB_MAX
IDX = jnp.arange(NB)
_PI = float(np.pi)


class FState(NamedTuple):
    r: jnp.ndarray   # [NB]
    x: jnp.ndarray   # [NB, 4]
    P: jnp.ndarray   # [NB, 4, 4]
    n: jnp.ndarray   # int32 scalar


def _consts():
    F = jnp.asarray(M.F_ncv())
    Q = jnp.asarray(M.Q_ncv())
    return F, Q


def empty_state(cfg: Cfg) -> FState:
    r = jnp.zeros((NB,))
    x = jnp.zeros((NB, 4))
    P = jnp.tile(jnp.eye(4), (NB, 1, 1))
    n = jnp.asarray(0, jnp.int32)
    if cfg.dummy_init:
        P = P.at[0].set(jnp.zeros((4, 4)))   # official main.py: covariances_existing = zeros, prob_existence = [0]
        n = jnp.asarray(1, jnp.int32)
    return FState(r, x, P, n)


def _benign(fs: FState) -> FState:
    act = IDX < fs.n
    return FState(jnp.where(act, fs.r, 0.0), jnp.where(act[:, None], fs.x, 0.0),
                  jnp.where(act[:, None, None], fs.P, jnp.eye(4)), fs.n)


def predict(fs: FState, cfg: Cfg):
    """eqs. 4-7: survive + Kalman predict, then append the birth Bernoulli at slot n.  Returns (state, overflow flag)."""
    F, Q = _consts()
    act = IDX < fs.n
    r = jnp.where(act, M.P_SURV * fs.r, 0.0)
    x = jnp.where(act[:, None], fs.x @ F.T, 0.0)
    P = jnp.where(act[:, None, None], jnp.einsum("ab,nbc,dc->nad", F, fs.P, F) + Q, jnp.eye(4))
    bm = jnp.asarray(M.BIRTH_MEAN)
    bc = M.BIRTH_VAR * jnp.eye(4)
    over = fs.n >= NB
    # out-of-range scatter is dropped, so an overflowing birth is simply lost (and counted)
    r = r.at[fs.n].set(cfg.r_b, mode="drop")
    x = x.at[fs.n].set(bm, mode="drop")
    P = P.at[fs.n].set(bc, mode="drop")
    return FState(r, x, P, jnp.minimum(fs.n + 1, NB)), over


def _compact(keep, *arrs):
    """Stable (order-preserving) compaction of the rows where keep is True; returns the arrays and the count."""
    order = jnp.argsort(~keep, stable=True)
    n = jnp.sum(keep).astype(jnp.int32)
    act = IDX < n
    out = []
    for a, fill in arrs:
        g = a[order]
        shp = (NB,) + (1,) * (a.ndim - 1)
        out.append(jnp.where(act.reshape(shp), g, fill))
    return out, n


def _inv2(S):
    a, b, c, d = S[..., 0, 0], S[..., 0, 1], S[..., 1, 0], S[..., 1, 1]
    det = a * d - b * c
    inv = jnp.stack([jnp.stack([d, -b], -1), jnp.stack([-c, a], -1)], -2) / det[..., None, None]
    return inv, det


def det_prob(spos, xp, cfg: Cfg):
    """Filter detection probability at the Bernoulli mean (D4): hard 40 m FOV (CFG-O) or eq. 48 at the mean (CFG-P)."""
    dist = jnp.sqrt(jnp.sum((xp - spos[None, :]) ** 2, axis=-1))
    if cfg.filt_pd == "hard":
        return jnp.where(dist < M.FOV, M.PD_MAX, 0.0)
    return M.PD_MAX * jnp.exp(-0.5 * (dist / M.FOV) ** 2)


def lbp_marginals(w0, w, wnew, cfg: Cfg):
    """Marginal association probabilities (pupd0[n], pupd[n, m]) from weights w0 (miss), w (detections), wnew (clutter).

    mu_{j->i} = 1 / (wnew_j + sum_{i' != i} nu_{i'->j}),  nu_{i->j} = w_ij / (w_i0 + sum_{j' != j} w_ij' mu_{j'->i})  ([40]; lbp.py).
    Returns also the number of sweeps run and a convergence flag.
    """
    def sweep(mu):
        prd = w * mu
        s = w0[:, None] + jnp.sum(prd, axis=1, keepdims=True)
        nu = w / (s - prd)
        mu_new = 1.0 / (wnew[None, :] + jnp.sum(nu, axis=0, keepdims=True) - nu)
        return mu_new

    mu0 = jnp.ones_like(w)
    if cfg.lbp_mode == "one_sweep":
        mu = sweep(mu0)
        conv = jnp.asarray(True)
        iters = jnp.asarray(1)
    else:
        def cond(c):
            mu, mu_old, it = c
            return (jnp.max(jnp.abs(mu - mu_old)) > cfg.lbp_tol) & (it < cfg.lbp_max_iter)

        def body(c):
            mu, _, it = c
            return sweep(mu), mu, it + 1

        mu, mu_old, iters = lax.while_loop(cond, body, (sweep(mu0), mu0, jnp.asarray(1)))
        conv = jnp.max(jnp.abs(mu - mu_old)) <= cfg.lbp_tol
    wm = w * mu
    s = w0 + jnp.sum(wm, axis=1)
    return w0 / s, wm / s[:, None], iters, conv


def update_sensor(fs: FState, z, zv, spos, lam_dens, cfg: Cfg):
    """One sensor's PMB update + MB projection (TOMB).  z[M,2] measurements, zv[M] validity, spos[2], lam_dens = clutter density.

    Returns (new state, info dict of scalars).
    """
    R = cfg.R_var * jnp.eye(2)
    act = IDX < fs.n
    xp = fs.x[:, M.POS_IDX]
    pD = jnp.where(act, det_prob(spos, xp, cfg), 0.0)
    r = fs.r
    w0 = jnp.where(act, 1.0 - r + r * (1.0 - pD), 1.0)
    r0 = jnp.where(act, r * (1.0 - pD) / w0, 0.0)
    Pxx = fs.P[:, M.POS_IDX][:, :, M.POS_IDX]
    S = Pxx + R
    Sinv, detS = _inv2(S)
    PHt = fs.P[:, :, M.POS_IDX]                                    # [NB, 4, 2]
    K = jnp.einsum("nab,nbc->nac", PHt, Sinv)                      # [NB, 4, 2]
    Pplus = fs.P - jnp.einsum("nab,nbc->nac", K, fs.P[:, M.POS_IDX, :])
    v = z[None, :, :] - xp[:, None, :]                             # [NB, M, 2]
    maha = jnp.einsum("nmi,nij,nmj->nm", v, Sinv, v)
    w = (r * pD)[:, None] * jnp.exp(-0.5 * maha) / (2.0 * _PI * jnp.sqrt(detS))[:, None]
    w = jnp.where(act[:, None] & zv[None, :], w, 0.0)
    wnew = jnp.full((z.shape[0],), lam_dens)
    p0, pj, iters, conv = lbp_marginals(w0, w, wnew, cfg)
    pr0 = p0 * r0
    prj = pj                                                       # rupd[:, j>0] = 1
    rn = pr0 + jnp.sum(prj, axis=1)
    rs = jnp.where(rn != 0.0, rn, 1.0)
    vbar = jnp.einsum("nm,nmi->ni", prj, v)                        # sum_j pr_j v_j
    a0 = vbar / rs[:, None]
    xn = fs.x + jnp.einsum("nab,nb->na", K, a0)
    aj = a0[:, None, :] - v                                        # [NB, M, 2]
    inner = (pr0 / rs)[:, None, None] * jnp.einsum("ni,nj->nij", a0, a0) + \
        jnp.einsum("nm,nmi,nmj->nij", prj / rs[:, None], aj, aj)
    Pn = (pr0 / rs)[:, None, None] * fs.P + ((rn - pr0) / rs)[:, None, None] * Pplus + \
        jnp.einsum("nab,nbc,ndc->nad", K, inner, K)
    keep = act & (rn > M.R_PRUNE)                                  # NaN compares False -> dropped, as in numpy
    (r2, x2, P2), n2 = _compact(keep, (rn, 0.0), (xn, 0.0), (Pn, jnp.eye(4)))
    # P2 fill: _compact fills with the scalar; restore identity on inactive slots
    act2 = IDX < n2
    P2 = jnp.where(act2[:, None, None], P2, jnp.eye(4))
    info = dict(lbp_iters=iters, lbp_conv=conv)
    return FState(r2, x2, P2, n2), info


def merge(fs: FState, cap: bool):
    """Greedy Mahalanobis merge (paper SS V-C, official utils.merge_gaussians), then prune r > 1e-4 and the region test."""
    act = IDX < fs.n
    Pinv = jnp.linalg.inv(fs.P)
    diff = fs.x[:, None, :] - fs.x[None, :, :]
    d = jnp.sqrt(jnp.einsum("ija,iab,ijb->ij", diff, Pinv, diff))
    close = d < M.MERGE_MAHA

    def body(i, c):
        used, G = c
        lead = act[i] & ~used[i]
        g = lead & close[i] & ~used & act & (IDX > i)
        g = g.at[i].set(lead)
        return used | g, G.at[i].set(g)

    used, G = lax.fori_loop(0, fs.n, body, (jnp.zeros((NB,), bool), jnp.zeros((NB, NB), bool)))
    Gf = G.astype(fs.r.dtype)
    W = Gf * fs.r[None, :]
    mw = jnp.sum(W, axis=1)
    mws = jnp.where(G[IDX, IDX], mw, 1.0)
    mean = (W @ fs.x) / mws[:, None]
    dm = fs.x[None, :, :] - mean[:, None, :]
    cov = jnp.einsum("ij,ijab->iab", W / mws[:, None],
                     fs.P[None, :, :, :] + jnp.einsum("ija,ijb->ijab", dm, dm))
    lead = G[IDX, IDX]
    rm = jnp.where(lead, mw, 0.0)
    if cap:
        rm = jnp.minimum(rm, 1.0)
    return _post_merge(rm, mean, cov, lead)


def _post_merge(rm, mean, cov, lead):
    keep = lead & (rm > M.R_PRUNE)
    inreg = (jnp.abs(mean[:, 0]) <= M.REGION) & (jnp.abs(mean[:, 2]) <= M.REGION)
    # the region test in the official code is x_condition = (x >= -250) & (x <= 250), inclusive
    keep = keep & inreg
    (r2, x2, P2), n2 = _compact(keep, (rm, 0.0), (mean, 0.0), (cov, 0.0))
    act2 = IDX < n2
    P2 = jnp.where(act2[:, None, None], P2, jnp.eye(4))
    return FState(r2, x2, P2, n2)


def estimate(fs: FState, cfg: Cfg = None):
    """Report the Bernoullis with r > 0.5 (official existence_threshold; est_rule='half', the default) or with r > Gamma*_d(tr P)
    (est_rule='gamma': the optimal-MSGOSPA threshold of eq. 30, Gamma*_d = 1 / (2 - min(2 tr P / c^2, 1)), tr over the full 4-D covariance
    when cfg.plan_trace == 'full' else over the position block).  Returns est[E_MAX,4], emask[E_MAX], overflow flag."""
    act = IDX < fs.n
    if cfg is not None and cfg.est_rule == "gamma":
        idx = np.arange(4) if cfg.plan_trace == "full" else M.POS_IDX
        t = jnp.sum(jnp.stack([fs.P[:, i, i] for i in idx], axis=0), axis=0)
        gam = 1.0 / (2.0 - jnp.fmin(2.0 * t / (M.GOSPA_C * M.GOSPA_C), 1.0))
        sel = act & (fs.r > gam)
    else:
        sel = act & (fs.r > M.EST_THRESH)
    order = jnp.argsort(~sel, stable=True)
    ne = jnp.sum(sel)
    ex = fs.x[order][: M.E_MAX]
    em = jnp.arange(M.E_MAX) < ne
    return ex, em, ne > M.E_MAX
