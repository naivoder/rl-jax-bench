"""Tests of the full-scale helper scripts (the scripts that test the paper's theory claims and the Figs 5-6 gap-level hypotheses).  Small instances only (< 30 s CPU)."""
import importlib.util
import os

import numpy as np
import jax.numpy as jnp

import model as M
import algorithm as A
import bound as B

HERE = os.path.dirname(__file__)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", "scripts", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BT = _load("bound_tests")
TH = _load("theory_checks")


def test_soft_dual_gradient_is_the_adjoint_of_the_backward_recursion():
    """d D_temp / d p_t (expected activations - K, from the forward mass transport) equals a central finite difference of the smoothed dual."""
    tabs = [BT.Tab("reckless", 4.0, 1.0), BT.Tab("cautious", 2.0, 5.0)]
    P0 = np.array([0.7, 1.3]); T, beta, K = 6, 0.9, 1
    rng = np.random.default_rng(0); p = rng.uniform(0.5, 3.0, T)
    for temp in (0.5, 0.1):
        D0, g = BT.soft_dual(tabs, P0, p, K, T, beta, temp)
        for t in (0, 2, 5):
            e = np.zeros(T); e[t] = 1e-5
            fd = (BT.soft_dual(tabs, P0, p + e, K, T, beta, temp)[0] - BT.soft_dual(tabs, P0, p - e, K, T, beta, temp)[0]) / 2e-5
            assert abs(fd - g[t]) < 2e-5 * max(1.0, abs(g[t])), (temp, t, fd, g[t])


def test_soft_dual_is_a_lower_bound_and_the_discounted_price_reproduces_V_D():
    """soft-min <= min (every temperature is a valid lower bound); with p_t = lambda* beta^t the finite-horizon hard dual equals the infinite-horizon V^D of bound.py to 5e-4."""
    types, qct, d, P0, K = ["reckless", "cautious", "reckless", "cautious"], [4.0] * 4, [5.0, 1.0, 5.0, 1.0], np.array([0.5, 1.2, 0.01, 1.9]), 1
    a = B.vi_dual(types, qct, d, P0[None], K)
    VD, lam = float(a["V_D"][0]), float(a["lam"][0])
    tabs = [BT.Tab(types[n], qct[n], d[n]) for n in range(4)]
    T, beta = 100, 0.9
    p = lam * beta ** np.arange(T)
    hard = BT.soft_dual(tabs, P0, p, K, T, beta, 0.0)[0]
    soft = BT.soft_dual(tabs, P0, p, K, T, beta, 0.3)[0]
    assert abs(hard - VD) / VD < 5e-4, (hard, VD)
    assert soft <= hard + 1e-9


def test_vi_whittle_index_matches_rollout_mp_index():
    """Whittle index by indifference lambda of the value iteration vs the rollout MP index f/g (claim 2), two points, tolerance 1e-3."""
    Pq = np.array([1.0, 5.0])
    for typ, q in (("reckless", 4), ("cautious", 10)):
        w = TH.vi_whittle(typ, q, Pq, n_bis=34)
        b = M.scalar_bank([typ] * 2, [float(q)] * 2, [1.0] * 2)
        mp, _ = A.index_whittle(b, jnp.asarray(Pq)[:, None, None], A.Cfg(guard=False))
        np.testing.assert_allclose(w, np.asarray(mp), rtol=1e-3)


def test_ceil_grid_mapping_is_an_upper_bias_of_the_exact_dual():
    """The 'ceil' grid (T5 hypothesis) rounds states up: its dual value is >= the interpolated one (value is increasing in P), the nearest-grid value is not ordered."""
    types, qct, d, P0, K, T, beta = ["reckless"] * 2, [4.0, 6.0], [1.0, 1.0], np.array([0.3, 0.9]), 1, 40, 0.9
    p = 20.0 * beta ** np.arange(T)
    exact = BT.soft_dual([BT.Tab(types[n], qct[n], d[n]) for n in range(2)], P0, p, K, T, beta, 0.0)[0]
    ceil = BT.soft_dual_grid([BT.TabGrid(types[n], qct[n], d[n], 0.1, "ceil") for n in range(2)], P0, p, K, T, beta)
    assert ceil >= exact - 1e-9
