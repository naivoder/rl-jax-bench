"""Classical allocation rules (paper §VII-B1) and the two labelled reproduction additions (A-48).

All take the env View (prior estimates only, never ground truth) and return dwells tau [K] summing to delta T0 over
the confirmed tracks (all-scan when there is no track). Pure, jit/vmap-able.
  EA  : delta T0 / n_trk each.
  DA  : delta T0 * r_hat / sum r_hat (prior range, A-31).
  OA  : min sum_n w_n c_ss(tau_n, r_hat_n) s.t. sum tau = delta T0 (A-32): bisection on the
        multiplier mu (outer, log scale) of per-target KKT roots found by bisection (inner). The paper's subgradient
        procedure is `oa_subgradient` (toy-test agreement only).
  OA-myopic (addition): same, with the exact one-step posterior trace given the current prior.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import physics as ph
from .common import EnvCfg


def ea(cfg: EnvCfg, view, delta: float):
    n = view.act.sum()
    return jnp.where(view.act, delta * cfg.T0 / jnp.maximum(n, 1), 0.0)


def da(cfg: EnvCfg, view, delta: float):
    r = jnp.where(view.act, view.r_hat, 0.0)
    return jnp.where(view.act, delta * cfg.T0 * r / jnp.maximum(r.sum(), 1e-300), 0.0)


def _tau_of_mu(dc, w, mu, D, T0, n_inner):
    """Per-target argmin_{0<=tau<=D} w c(tau) + mu tau / T0 for convex decreasing c (bisection on the derivative)."""
    lo = jnp.zeros_like(w)
    hi = jnp.full_like(w, D)

    def body(_, lh):
        lo, hi = lh
        mid = 0.5 * (lo + hi)
        fp = w * dc(mid) + mu / T0
        return jnp.where(fp > 0, lo, mid), jnp.where(fp > 0, mid, hi)

    lo, hi = jax.lax.fori_loop(0, n_inner, body, (lo, hi))
    return 0.5 * (lo + hi)


def allocate(dc, w, act, D, T0, n_outer: int = 64, n_inner: int = 52):
    """REFERENCE solver (toy tests only): min sum w c(tau) s.t. sum tau = D by bisection on log mu with bisection
    inner roots. Exact to ~1e-12 but ~3300 sequential steps per decision (too slow inside 1e4-slot rollouts).
    Returns (tau, mu)."""
    w = jnp.where(act, w, 0.0)

    def body(_, lh):
        lo, hi = lh
        mid = 0.5 * (lo + hi)
        s = _tau_of_mu(dc, w, jnp.exp(mid), D, T0, n_inner).sum()
        return jnp.where(s > D, mid, lo), jnp.where(s > D, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_outer, body, (jnp.log(1e-3), jnp.log(1e14)))
    mu = jnp.exp(0.5 * (lo + hi))
    tau = jnp.where(act, _tau_of_mu(dc, w, mu, D, T0, n_inner), 0.0)
    tau = tau * D / jnp.maximum(tau.sum(), 1e-300)          # remove the residual bisection error
    return jnp.where(act.any(), tau, 0.0), mu


def allocate_newton(dc, d2c, w, act, D, T0, n_outer: int = 12, n_inner: int = 4):
    """PRODUCTION solver (same problem as `allocate`): Newton on log mu (outer) with warm-started Newton on
    u = log tau (inner) for F_n(u) = log(-w_n c_n'(e^u)) - log(mu / T0) = 0, which is near-linear in u for the
    power-law-like costs here; dS/dlog mu by implicit differentiation (du/dm = 1/F'(u)); clamps at
    tau in [1e-10, D] (a clamped target has zero derivative). Unrolled (no while loops), ~50 cost evaluations.
    Agreement with `allocate` to <= 1e-6 relative is checked by a toy test. Returns (tau, mu)."""
    n_act = act.sum()
    w = jnp.where(act, w, 1.0)
    lo, hi = jnp.log(1e-10), jnp.log(D)

    def F_and_dF(u, m):
        tau = jnp.exp(u)
        c1, c2 = dc(tau), d2c(tau)
        c1 = jnp.minimum(c1, -1e-300)
        F = jnp.log(w) + jnp.log(-c1) - m + jnp.log(T0)
        dF = c2 * tau / c1                                    # < 0
        return F, jnp.minimum(dF, -1e-12)

    # start: equal split; m0 = log-mean of the per-target multipliers at the equal split
    u = jnp.full(w.shape, jnp.log(D / jnp.maximum(n_act, 1)))
    m_i = jnp.log(-w * T0 * jnp.minimum(dc(jnp.exp(u)), -1e-300))
    m = (jnp.where(act, m_i, 0.0)).sum() / jnp.maximum(n_act, 1)
    for _ in range(n_outer):
        for _ in range(n_inner):
            F, dF = F_and_dF(u, m)
            u = jnp.clip(u - F / dF, lo, hi)
        F, dF = F_and_dF(u, m)
        free = ~(((u >= hi) & (F > 0)) | ((u <= lo) & (F < 0))) & act
        tau = jnp.where(act, jnp.exp(u), 0.0)
        ssum = tau.sum()
        dS = jnp.where(free, tau / dF, 0.0).sum() / jnp.maximum(ssum, 1e-300)
        S = jnp.log(jnp.maximum(ssum, 1e-300)) - jnp.log(D)
        step = jnp.where(dS < 0, S / jnp.minimum(dS, -1e-300), 0.0)
        m = m - jnp.clip(step, -3.0, 3.0)
    for _ in range(n_inner):
        F, dF = F_and_dF(u, m)
        u = jnp.clip(u - F / dF, lo, hi)
    tau = jnp.where(act, jnp.exp(u), 0.0)
    tau = tau * D / jnp.maximum(tau.sum(), 1e-300)
    tau = jnp.where(n_act == 1, jnp.where(act, D, 0.0), tau)
    return jnp.where(act.any(), tau, 0.0), jnp.exp(m)


def ss_d2cost(cfg: EnvCfg, snr0, r_hat, factor):
    g2 = jax.grad(jax.grad(lambda tau, r, f: ph.c_ss(tau, r, snr0, cfg, f)))
    return lambda tau: jax.vmap(g2)(jnp.maximum(tau, 1e-12), r_hat, factor)


def ss_dcost(cfg: EnvCfg, snr0, r_hat, factor):
    """d c_ss / d tau at the estimated ranges (vectorised over K)."""
    g = jax.grad(lambda tau, r, f: ph.c_ss(tau, r, snr0, cfg, f))
    return lambda tau: jax.vmap(g)(jnp.maximum(tau, 1e-12), r_hat, factor)


def oa(cfg: EnvCfg, view, delta: float, snr0, informed: bool = True):
    fac = view.factor_hat if informed else jnp.ones_like(view.r_hat)
    tau, _ = allocate_newton(ss_dcost(cfg, snr0, view.r_hat, fac), ss_d2cost(cfg, snr0, view.r_hat, fac), view.w,
                             view.act, delta * cfg.T0, cfg.T0)
    return tau


def myopic_cost(cfg: EnvCfg, snr0, xp, Pp, factor, tau):
    """Exact one-step posterior position trace given the prior, measurement noise from the ESTIMATED range (no
    truth). Covariance form with an explicit 2x2 inverse (no linalg calls; equal to the information form
    tr_pos((Pp^-1 + tau H^T R1^-1 H)^-1), checked in tests):
        P+ = Pp - Pp H^T (H Pp H^T + R1 / tau)^-1 H Pp."""
    H = ph.H_jac(xp)
    r = jnp.hypot(xp[0], xp[1])
    snr1 = ph.track_snr(1.0, r, snr0, cfg.r0, cfg.tau0, cfg.r_min_snr)
    vr, vt = ph.meas_var(snr1, cfg.sigma_r0_sq, cfg.sigma_th0_sq)
    PH = Pp @ H.T                                            # [4, 2]
    S = H @ PH + jnp.diag(jnp.array([vr * factor, vt * factor])) / tau
    det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
    Sinv = jnp.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
    G = PH[:2] @ Sinv @ PH[:2].T                             # position block of Pp H^T S^-1 H Pp
    return Pp[0, 0] + Pp[1, 1] - G[0, 0] - G[1, 1]


def myopic_cost_info(cfg: EnvCfg, snr0, xp, Pp, factor, tau):
    """Information-form reference of myopic_cost (tests only)."""
    H = ph.H_jac(xp)
    r = jnp.hypot(xp[0], xp[1])
    snr1 = ph.track_snr(1.0, r, snr0, cfg.r0, cfg.tau0, cfg.r_min_snr)
    vr, vt = ph.meas_var(snr1, cfg.sigma_r0_sq, cfg.sigma_th0_sq)
    Rinv1 = jnp.diag(jnp.array([1.0 / (vr * factor), 1.0 / (vt * factor)]))
    J = jnp.linalg.inv(Pp) + tau * H.T @ Rinv1 @ H
    return ph.pos_trace(jnp.linalg.inv(J))


def oa_myopic(cfg: EnvCfg, view, delta: float, snr0, informed: bool = True):
    fac = view.factor_hat if informed else jnp.ones_like(view.r_hat)
    # inactive slots carry an identity prior: harmless (weight 0) and invertible
    Pp = jnp.where(view.act[:, None, None], view.Pp, jnp.eye(4))
    xp = jnp.where(view.act[:, None], view.xp, jnp.array([1.0e4, 0.0, 0.0, 0.0]))
    g = jax.grad(lambda tau, x, P, f: myopic_cost(cfg, snr0, x, P, f, tau))
    g2 = jax.grad(g)
    dc = lambda tau: jax.vmap(g)(jnp.maximum(tau, 1e-12), xp, Pp, fac)
    d2c = lambda tau: jax.vmap(g2)(jnp.maximum(tau, 1e-12), xp, Pp, fac)
    tau, _ = allocate_newton(dc, d2c, view.w, view.act, delta * cfg.T0, cfg.T0)
    return tau


def oa_subgradient(dc, w, act, D, T0, mu0=5.0e3, eta0=1.0e5, n_iter=20000, n_inner=52):
    """The paper's procedure (§VII-B1): mu <- max(0, mu + eta_k (sum tau(mu)/T0 - D/T0)), eta_k = eta0/sqrt(k+1).
    toy tests only (must agree with the bisection)."""
    w = jnp.where(act, w, 0.0)

    def body(k, mu):
        s = _tau_of_mu(dc, w, mu, D, T0, n_inner).sum()
        return jnp.maximum(0.0, mu + eta0 / jnp.sqrt(k + 1.0) * (s / T0 - D / T0))

    mu = jax.lax.fori_loop(0, n_iter, body, mu0)
    return jnp.where(act, _tau_of_mu(dc, w, mu, D, T0, n_inner), 0.0), mu
