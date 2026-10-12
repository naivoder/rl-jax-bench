"""Toy checks: exactly computable problems that exercise the SAME instrument code paths
(algorithm.pg_vector, surrogate_grid, gae, lambda_return, cos_* / snr_*, mre, model.gauss_logp).

Each function returns a dict of numbers plus boolean ``pass_*`` flags; tests/test_grads.py runs them small,
scripts/toy_checks.py runs them at full size and records the numbers in runs/toys/toys.json.
"""
from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from . import algorithm as A
from .model import gauss_logp


# ------------------------------------------------------------------ 6.1 Gaussian bandit
class BanditNets:
    """Duck-typed 'nets' for a one-step Gaussian policy a = mu + sigma*eps, theta = (mu, log sigma)."""

    @staticmethod
    def logp(p, obs, act):
        return gauss_logp(p[0][None, None] * jnp.ones_like(act), p[1][None] * jnp.ones((act.shape[-1],)), act)


def bandit_closed_form(d, s):
    return dict(
        g=(-2 * d, -2 * s ** 2), J=-(d ** 2 + s ** 2),
        var={"zero": (d ** 4 / s ** 2 + 14 * d ** 2 + 15 * s ** 2, 2 * d ** 4 + 60 * d ** 2 * s ** 2 + 74 * s ** 4),
             "V": (8 * d ** 2 + 10 * s ** 2, 40 * d ** 2 * s ** 2 + 56 * s ** 4),
             "bstar": (8 * d ** 2 + 6 * s ** 2, 40 * d ** 2 * s ** 2 + 24 * s ** 4)},
        bstar=(-d ** 2 - 3 * s ** 2, -d ** 2 - 5 * s ** 2))


def bandit_estimator(key, N, d=1.0, s=1.0, c=0.0):
    """Per-sample eq. (1) estimates (r - b) * grad log pi via pg_vector with n=1 (raw advantages), for b in
    {0, V, b*}; compares mean and variance with the sympy closed forms within 4 SE."""
    mu, ls = c + d, math.log(s)
    p = jnp.array([mu, ls], jnp.float32)
    a = mu + s * jax.random.normal(key, (N,))
    r = -(a - c) ** 2
    cf = bandit_closed_form(d, s)
    obs = jnp.zeros((1, 1))

    def one(ai, adv):
        return A.pg_vector(BanditNets, p, obs, ai[None, None], adv[None], normalize=False)

    out, ok = {}, True
    for name in ("zero", "V", "bstar"):
        if name == "zero":
            b = (0.0, 0.0)
        elif name == "V":
            b = (cf["J"], cf["J"])
        else:
            b = cf["bstar"]
        g_mu = jax.vmap(one)(a, r - b[0])[:, 0]
        g_s = jax.vmap(one)(a, r - b[1])[:, 1]
        res = {}
        for k, (gk, tg, tv) in enumerate(((g_mu, cf["g"][0], cf["var"][name][0]), (g_s, cf["g"][1], cf["var"][name][1]))):
            x = np.asarray(gk, np.float64)
            m, v = x.mean(), x.var(ddof=1)
            se_m = math.sqrt(v / N)
            m4 = np.mean((x - m) ** 4)
            se_v = math.sqrt(max(m4 - v ** 2, 0) / N)
            z_m, z_v = (m - tg) / se_m, (v - tv) / se_v
            res["mu" if k == 0 else "logsig"] = dict(mean=m, mean_true=tg, z_mean=z_m, var=v, var_true=tv, z_var=z_v)
            ok &= abs(z_m) < 4 and abs(z_v) < 4
        out[name] = res
    out["pass_bandit_moments"] = bool(ok)
    return out


def bandit_landscape(key, N, d=1.0, s=1.0, c=0.0, delta=(-0.4, -0.3), alphas=(0.0, 0.5, 1.0, 2.0, 3.0)):
    """On a single state, E_{a~pi0}[rho_theta (r - V0)] = J(theta) - J(theta0) exactly, and with normalised
    advantages L_norm = (J - J0)/sigma_A. Uses algorithm.surrogate_grid (TRPO surrogate) on a fixed N-sample."""
    mu, ls = c + d, math.log(s)
    p0 = jnp.array([mu, ls], jnp.float32)
    a = mu + s * jax.random.normal(key, (N,))
    r = -(a - c) ** 2
    J = lambda m, lsig: -((m - c) ** 2 + math.exp(2 * lsig))
    lp0 = gauss_logp(p0[0] * jnp.ones((N, 1)), p0[1] * jnp.ones((1,)), a[:, None])
    b = A.Batch(jnp.zeros((N, 1)), a[:, None], lp0, r - J(mu, ls), r, jnp.zeros(N), jnp.ones(N))
    al = jnp.asarray(alphas, jnp.float32)
    S = A.surrogate_grid(BanditNets, "trpo", 0.2, p0, jnp.asarray(delta, jnp.float32), jnp.zeros(2), al,
                         jnp.zeros(1), b)[:, 0]
    sig = float(jnp.std(b.adv, ddof=1))
    mu_hat = float(jnp.mean(b.adv))
    rows, ok = [], True
    nadv = np.asarray(A.adv_normalize(b.adv), np.float64)
    for i, al_ in enumerate(alphas):
        th = (mu + al_ * delta[0], ls + al_ * delta[1])
        exact = (J(*th) - J(mu, ls) - mu_hat) / sig      # E_pi0[rho (A - mu_hat)]/sigma_hat, E[rho] = 1
        lp = gauss_logp(th[0] * jnp.ones((N, 1)), th[1] * jnp.ones((1,)), a[:, None])
        w = np.asarray(jnp.exp(lp - lp0), np.float64) * nadv
        se = w.std(ddof=1) / math.sqrt(N)
        z = (float(S[i]) - exact) / max(se, 1e-12)
        rows.append(dict(alpha=al_, surrogate=float(S[i]), exact=exact, se=se, z=z))
        ok &= abs(z) < 4
    return dict(rows=rows, sigma_A=sig, pass_landscape_identity=bool(ok))


# ------------------------------------------------------------------ 6.1 cosine / SNR model at d = 1e4
def cosine_model(key, d=10_000, n_half=5000.0, sizes=(500, 2000, 5000, 20000), m=10, N=1_000_000, reps=200):
    """Gaussian g_hat_n = g + N(0, Sigma/n), ||g|| = 1, tr Sigma = n_half; the instrument
    (algorithm.cos_matrix / pairwise_mean) must match n/(n+n_half) and 1/sqrt((1+n_half/n)(1+n_half/N))."""
    kg, ks, kd = jax.random.split(key, 3)
    g = jax.random.normal(kg, (d,))
    g = g / jnp.linalg.norm(g)
    sd = jax.random.uniform(ks, (d,), minval=0.2, maxval=1.8)
    sd = sd * jnp.sqrt(n_half / jnp.sum(sd ** 2))          # tr Sigma = n_half
    rows, ok = [], True
    for i, n in enumerate(sizes):
        k = jax.random.fold_in(kd, i)

        def rep(kk):
            k1, k2 = jax.random.split(kk)
            G = g + sd * jax.random.normal(k1, (m, d)) / math.sqrt(n)
            gr = g + sd * jax.random.normal(k2, (d,)) / math.sqrt(N)
            Gn = G / jnp.linalg.norm(G, axis=1, keepdims=True)
            C = Gn @ Gn.T
            iu = jnp.triu_indices(m, 1)
            return C[iu].mean(), jnp.mean(Gn @ gr / jnp.linalg.norm(gr))

        pw, tt = jax.vmap(rep)(jax.random.split(k, reps))
        pw, tt = np.asarray(pw, np.float64), np.asarray(tt, np.float64)
        e_pw, e_tt = A.snr_pairwise(n, n_half), A.snr_to_true(n, n_half, N)
        z1 = (pw.mean() - e_pw) / (pw.std(ddof=1) / math.sqrt(reps))
        z2 = (tt.mean() - e_tt) / (tt.std(ddof=1) / math.sqrt(reps))
        rows.append(dict(n=n, pairwise=pw.mean(), pairwise_model=e_pw, z_pw=z1, to_true=tt.mean(), to_true_model=e_tt,
                         z_tt=z2))
        ok &= abs(z1) < 4 and abs(z2) < 4
    # variance ratio read-back from two cosines with known ratio 0.25
    c0, cb = A.snr_pairwise(2000, n_half), A.snr_pairwise(2000, 0.25 * n_half)
    vr = A.variance_ratio_from_cos(cb, c0)
    ok &= abs(vr - 0.25) < 1e-9
    # instrument path itself (numpy cos_stats) on one draw vs the jnp computation
    G = np.asarray(g + sd * jax.random.normal(jax.random.fold_in(kd, 99), (m, d)) / math.sqrt(2000), np.float64)
    st = A.cos_stats(G, np.asarray(g), n_boot=50)
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    ok &= abs(st["pairwise"] - (Gn @ Gn.T)[np.triu_indices(m, 1)].mean()) < 1e-12
    return dict(rows=rows, variance_ratio_readback=vr, pass_cosine_model=bool(ok))


# ------------------------------------------------------------------ 6.2 two-state MDP
P = np.array([[[0.9, 0.1], [0.2, 0.8]], [[0.7, 0.3], [0.05, 0.95]]])
R = np.array([[0.0, 1.0], [2.0, 0.5]])
GAMMA, HORIZON = 0.9, 10
THETA = np.array([[0.3, -0.2], [0.1, 0.4]])
GRAD_J = np.array([-1.448360, 1.448360, 0.628427, -0.628427])
G_PRAC = np.array([-2.044889, 2.044889, 1.083924, -1.083924])
COS_PRAC_TRUE = 0.996957


def mdp_J(theta):
    """Exact J (gamma^t-weighted, rho0 = delta_0, horizon 10) by forward DP, in JAX (differentiable)."""
    th = theta.reshape(2, 2)
    pi = jax.nn.softmax(th, -1)
    Pj, Rj = jnp.asarray(P), jnp.asarray(R)
    d = jnp.array([1.0, 0.0])
    tot = 0.0
    for t in range(HORIZON):
        tot = tot + GAMMA ** t * jnp.einsum("s,sa,sa->", d, pi, Rj)
        d = jnp.einsum("s,sa,sat->t", d, pi, Pj)
    return tot


def mdp_values(theta):
    pi = np.exp(theta) / np.exp(theta).sum(1, keepdims=True)
    V = np.zeros((HORIZON + 1, 2))
    for t in reversed(range(HORIZON)):
        Q = R + GAMMA * np.einsum("sat,t->sa", P, V[t + 1])
        V[t] = (pi * Q).sum(1)
    return V


class TabNets:
    """Duck-typed nets: obs = one-hot state [2], logits = theta[s]; Gaussian-free categorical log-prob."""

    @staticmethod
    def logp(p, obs, act):
        logits = obs @ p.reshape(2, 2)
        return jnp.sum(jax.nn.log_softmax(logits, -1) * act, -1)


def mdp_sample(key, n_traj, theta):
    pi = jnp.asarray(np.exp(theta) / np.exp(theta).sum(1, keepdims=True))
    Pj, Rj = jnp.asarray(P), jnp.asarray(R)

    def traj(k):
        def step(s, kk):
            k1, k2 = jax.random.split(kk)
            a = jax.random.categorical(k1, jnp.log(pi[s]))
            s1 = jax.random.categorical(k2, jnp.log(Pj[s, a]))
            return s1, (s, a, Rj[s, a])
        _, (s, a, r) = jax.lax.scan(step, jnp.int32(0), jax.random.split(k, HORIZON))
        return s, a, r

    return jax.vmap(traj)(jax.random.split(key, n_traj))


def mdp_pipeline(key, n_traj=200_000):
    """(a) GAE(lambda=1) with the exact time-indexed V + score pipeline -> g_prac (within 4 SE); (b) check_grads /
    exact grad J; (c) cosine instrument returns cos(g_prac, grad J) = 0.996957."""
    from jax.test_util import check_grads
    s, a, r = mdp_sample(key, n_traj, THETA)
    V = mdp_values(THETA)
    vals = jnp.asarray(V[np.arange(HORIZON)[None, :], np.asarray(s)], jnp.float32)
    nd = jnp.ones((n_traj, HORIZON)).at[:, -1].set(0.0)
    adv, _ = jax.vmap(lambda rr, vv, dd: A.gae(rr, vv, dd, GAMMA, 1.0))(r.astype(jnp.float32), vals, nd)
    obs = jax.nn.one_hot(s, 2)
    act = jax.nn.one_hot(a, 2)
    p = jnp.asarray(THETA.ravel(), jnp.float32)
    # per-trajectory estimate sum_t psi_t A_t  (= T * pg_vector on that trajectory, raw advantages)
    per = jax.vmap(lambda o, ac, ad: HORIZON * A.pg_vector(TabNets, p, o, ac, ad, normalize=False))(obs, act, adv)
    per = np.asarray(per, np.float64)
    m, se = per.mean(0), per.std(0, ddof=1) / math.sqrt(n_traj)
    z = (m - G_PRAC) / se
    gJ = np.asarray(jax.grad(mdp_J)(jnp.asarray(THETA.ravel())), np.float64)
    check_grads(mdp_J, (jnp.asarray(THETA.ravel(), jnp.float32),), order=1, modes=["rev"], atol=2e-2, rtol=2e-2)
    cos = float(A.cosine(jnp.asarray(G_PRAC), jnp.asarray(GRAD_J)))
    return dict(g_prac_est=m.tolist(), g_prac_true=G_PRAC.tolist(), se=se.tolist(), z=z.tolist(),
                grad_J_autodiff=gJ.tolist(), grad_J_ref=GRAD_J.tolist(), grad_J_maxerr=float(np.abs(gJ - GRAD_J).max()),
                cos_instrument=cos, cos_ref=COS_PRAC_TRUE,
                pass_mdp=bool(np.all(np.abs(z) < 4) and np.abs(gJ - GRAD_J).max() < 1e-5
                              and abs(cos - COS_PRAC_TRUE) < 1e-6))


def gae_identity(key, n_traj=64, T=200, gamma=0.99, lam=0.95):
    """: V_old + A^GAE == TD(lambda) return (algorithm.lambda_return), and G^lam - G =
    sum_l (g*l)^l g(1-l)(V_old(s_{t+l+1}) - G_{t+l+1}) inside each terminated episode, on random episodes."""
    k1, k2, k3 = jax.random.split(key, 3)
    r = jax.random.normal(k1, (n_traj, T))
    v = jax.random.normal(k2, (n_traj, T)) * 3
    nd = (jax.random.uniform(k3, (n_traj, T)) > 0.03).astype(jnp.float32).at[:, -1].set(0.0)
    adv, ret = jax.vmap(lambda a, b, c: A.gae(a, b, c, gamma, lam))(r, v, nd)
    lr = jax.vmap(lambda a, b, c: A.lambda_return(a, b, c, gamma, lam))(r, v, nd)
    err1 = float(jnp.max(jnp.abs(v + adv - lr)))
    # explicit sum in float64
    r_, v_, nd_, y_, G_ = (np.asarray(x, np.float64) for x in (r, v, nd, v + adv, ret))
    err2 = 0.0
    for i in range(n_traj):
        t = 0
        while t < T:
            e = t
            while nd_[i, e] > 0.5:
                e += 1
            for u in range(t, e + 1):
                rhs = sum((gamma * lam) ** l * gamma * (1 - lam) * (v_[i, u + l + 1] - G_[i, u + l + 1])
                          for l in range(0, e - u))
                err2 = max(err2, abs((y_[i, u] - G_[i, u]) - rhs))
            t = e + 1
    weight = gamma * (1 - lam) / (1 - gamma * lam)
    return dict(max_abs_gae_vs_lambda_return=err1, max_abs_D16=err2, total_weight=weight,
                pass_gae_identity=bool(err1 < 1e-4 and err2 < 1e-3))


# ------------------------------------------------------------------ 6.3 Returns-MRE floor
def mre_toys(key, N=2_000_000):
    """ two-point example (exact), a well-posed continuous case, and the ill-posed case.

    Note: with r = 10 - (a-c)^2 the return crosses 0 with non-zero density, so
    |V - r|/|r| ~ 1/|r - 0| has a NON-integrable singularity and E[MRE] = +inf; quad's 0.1112 is a numerical
    artefact and the Monte-Carlo mean creeps up with N. The check uses r = 10 + (a-c)^2 (> 0, finite MRE)."""
    two_v = float(A.mre(jnp.array([2.0, 2.0]), jnp.array([1.0, 3.0])))
    two_1 = float(A.mre(jnp.array([1.0, 1.0]), jnp.array([1.0, 3.0])))
    from scipy import integrate, stats
    d, s, c = 1.0, 0.5, 0.0
    a = c + d + s * jax.random.normal(key, (N,))
    # well-posed: r = 10 + (a-c)^2, V = 10 + d^2 + s^2
    r = 10.0 + (a - c) ** 2
    Vt = 10.0 + d ** 2 + s ** 2
    x = np.asarray(jnp.abs(Vt - r) / jnp.abs(r), np.float64)
    m, se = float(A.mre(Vt * jnp.ones_like(r), r)), x.std(ddof=1) / math.sqrt(N)
    f = lambda e: abs(Vt - (10 + (d + s * e) ** 2)) / (10 + (d + s * e) ** 2) * stats.norm.pdf(e)
    ref = integrate.quad(f, -40, 40, limit=400, points=[(-math.sqrt(1.25) - d) / s, (math.sqrt(1.25) - d) / s])[0]
    # ill-posed original: report the MC mean at growing N (diverges ~ log N)
    r2 = 10.0 - (a - c) ** 2
    y = np.asarray(jnp.abs(8.75 - r2) / jnp.abs(r2), np.float64)
    growth = {str(n): float(y[:n].mean()) for n in (10_000, 100_000, N)}
    return dict(two_point_exact_V=two_v, two_point_const_1=two_1, bandit_pos_mre=m, bandit_pos_mre_se=se,
                bandit_pos_mre_quad=ref, illposed_mre_mc_by_N=growth, illposed_max_term=float(y.max()),
                pass_mre=bool(abs(two_v - 2 / 3) < 1e-6 and abs(two_1 - 1 / 3) < 1e-6 and abs(m - ref) < 4 * se + 1e-5))
