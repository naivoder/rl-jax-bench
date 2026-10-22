"""LBP / data-association marginals (paper eq. 17, [40]) against exhaustive enumeration of all global hypotheses, and the D30 finding
(the authors' lbp.py runs exactly one sweep)."""
import itertools
import numpy as np
import jax, jax.numpy as jnp
import pytest
from src import filter as FL
from src import model as M


def exact_marginals(w0, w, wnew):
    n, m = w.shape
    psi = w / wnew[None, :]
    tot = 0.0
    marg0 = np.zeros(n); margj = np.zeros((n, m))
    for a in itertools.product(range(m + 1), repeat=n):
        used = [x for x in a if x > 0]
        if len(used) != len(set(used)):
            continue
        wt = 1.0
        for i, ai in enumerate(a):
            wt *= w0[i] if ai == 0 else psi[i, ai - 1]
        tot += wt
        for i, ai in enumerate(a):
            if ai == 0:
                marg0[i] += wt
            else:
                margj[i, ai - 1] += wt
    return marg0 / tot, margj / tot


def _lbp(cfg, w0, w, wnew):
    p0, pj, it, conv = FL.lbp_marginals(jnp.asarray(w0), jnp.asarray(w), jnp.asarray(wnew), cfg)
    return np.asarray(p0), np.asarray(pj), int(it), bool(conv)


def _case(rng, n, m, dense):
    w0 = rng.uniform(0.1, 1.0, n)
    w = rng.uniform(0.0, 3.0, (n, m)) * (rng.random((n, m)) < (0.9 if dense else 0.4))
    wnew = np.full(m, rng.uniform(0.05, 0.6))
    return w0, w, wnew


def test_converged_lbp_exact_on_trees():
    rng = np.random.default_rng(0)
    cfg = M.CFG_P
    for _ in range(60):
        # single track, any number of measurements: the association graph is a star (a tree) -> BP is exact
        n, m = 1, int(rng.integers(1, 6))
        w0, w, wnew = _case(rng, n, m, True)
        p0, pj, it, conv = _lbp(cfg, w0, w, wnew)
        e0, ej = exact_marginals(w0, w, wnew)
        assert np.max(np.abs(p0 - e0)) < 1e-9 and np.max(np.abs(pj - ej)) < 1e-9 and conv
        # two tracks with disjoint measurement supports -> also a forest
        n, m = 2, 4
        w0, w, wnew = _case(rng, n, m, True)
        w[0, 2:] = 0.0; w[1, :2] = 0.0
        p0, pj, it, conv = _lbp(cfg, w0, w, wnew)
        e0, ej = exact_marginals(w0, w, wnew)
        assert np.max(np.abs(p0 - e0)) < 1e-9 and np.max(np.abs(pj - ej)) < 1e-9


def test_converged_lbp_accuracy_on_loopy_cases_and_one_sweep_differs():
    rng = np.random.default_rng(1)
    err_conv, err_one, nloop = [], [], 0
    for _ in range(150):
        n, m = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        w0, w, wnew = _case(rng, n, m, True)
        e0, ej = exact_marginals(w0, w, wnew)
        p0, pj, _, conv = _lbp(M.CFG_P, w0, w, wnew)
        o0, oj, it, _ = _lbp(M.CFG_O, w0, w, wnew)
        assert conv and it >= 1 and CFG_O_ITERS == 1 == it
        err_conv.append(max(np.max(np.abs(p0 - e0)), np.max(np.abs(pj - ej))))
        err_one.append(max(np.max(np.abs(o0 - e0)), np.max(np.abs(oj - ej))))
    print(f"[METRIC] LBP max abs marginal error vs exact enumeration: converged mean {np.mean(err_conv):.2e} max {np.max(err_conv):.2e}; "
          f"one-sweep (official D30) mean {np.mean(err_one):.2e} max {np.max(err_one):.2e}")
    assert np.max(err_conv) < 0.3                       # loopy BP is approximate (these random dense cases are adversarial)
    assert np.mean(err_one) > np.mean(err_conv)          # the one-sweep variant is the less accurate one


CFG_O_ITERS = 1


def _bp_reference(w0, w, wnew, iters=500):
    """plain-loop fixed-point BP of [40] (independent of src/filter.py): returns the converged marginals."""
    n, m = w.shape
    mu = np.ones((n, m))
    for _ in range(iters):
        nu = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                nu[i, j] = w[i, j] / (w0[i] + sum(w[i, jj] * mu[i, jj] for jj in range(m) if jj != j))
        mu_new = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                mu_new[i, j] = 1.0 / (wnew[j] + sum(nu[ii, j] for ii in range(n) if ii != i))
        if np.max(np.abs(mu_new - mu)) < 1e-14:
            mu = mu_new
            break
        mu = mu_new
    s = np.array([w0[i] + (w[i] * mu[i]).sum() for i in range(n)])
    return w0 / s, w * mu / s[:, None]


def test_converged_lbp_equals_reference_fixed_point():
    rng = np.random.default_rng(2)
    cfg = M.CFG_P.replace(lbp_tol=1e-13, lbp_max_iter=5000)
    worst = 0.0
    for _ in range(60):
        n, m = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        w0, w, wnew = _case(rng, n, m, True)
        r0, rj = _bp_reference(w0, w, wnew)
        p0, pj, _, conv = _lbp(cfg, w0, w, wnew)
        if conv:
            worst = max(worst, np.max(np.abs(p0 - r0)), np.max(np.abs(pj - rj)))
    assert worst < 1e-8, worst
