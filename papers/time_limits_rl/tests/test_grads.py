"""toy stage -- exact / closed-form correctness .

The paper introduces no new loss or gradient: time-awareness changes the INPUT and PEB changes the TD/GAE TARGET.
So, as the loop brief allows for methods without a new gradient, this file tests the estimators and fixed points
against closed forms (float64), plus finite-difference `check_grads` on the two autodiff paths that exist here:
the PPO loss as composed by our trainer (vendored PPO-details terms, with the time input) and the exact Queue-of-Cars
return used to find the best memoryless policy.

Run: JAX_PLATFORMS=cpu uv run python -m pytest tests/test_grads.py -q
"""
import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src import exact as X  # noqa: E402
from src.ppo8 import algorithm as P8  # noqa: E402

G = 0.99


# ------------------------------------------------------------------ 1. Last Moment 
@pytest.mark.parametrize("gamma,T,ta,best", [(1.0, 2, 1.0, 0.25), (1.0, 3, 1.0, 0.0), (1.0, 4, 1.0, 0.0),
                                             (1.0, 5, 1.0, 0.0), (1.0, 10, 1.0, 0.0), (0.99, 2, 0.99, 0.252525),
                                             (0.99, 3, 0.9801, 5.0e-5), (0.99, 4, 0.970299, 0.0),
                                             (0.99, 10, 0.913517, 0.0)])
def test_last_moment_table(gamma, T, ta, best):
    Q = X.finite_horizon_q(D.last_moment(), gamma, T)
    assert abs(float(Q[T][0].max()) - ta) < 1e-6           # time-aware optimum gamma^(T-1), from the DP
    assert abs(gamma ** (T - 1) - ta) < 1e-6
    p, J = X.last_moment_best_memoryless(T, gamma)
    assert abs(float(J) - best) < 1e-6
    if T == 2:
        assert abs(float(p) - 1 / (2 * gamma)) < 1e-4       # p* = 1/(2 gamma)
    if T >= 3 and gamma == 1.0:
        assert float(p) == 0.0                              # 'never jump' (exact at gamma = 1)


def test_last_moment_ta_policy_is_stay_then_jump():
    Q = X.finite_horizon_q(D.last_moment(), 1.0, 5)
    acts = [int(Q[k][0].argmax()) for k in range(5, 0, -1)]  # k = remaining steps, t = 0..4
    assert acts == [0, 0, 0, 0, 1]


# ------------------------------------------------------------------ 2. Two-Goal Gridworld 
@pytest.fixture(scope="module")
def grid_ref(tmp_path_factory):
    """Run the exact-solver numpy reference solver (scripts/exact_solvers/grid.py) and load its tables."""
    d = tmp_path_factory.mktemp("grid")
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "exact_solvers", "grid.py")], cwd=d, check=True,
                   capture_output=True)
    return dict(np.load(d / "grid_fp.npz"))


@pytest.fixture(scope="module")
def tg():
    return D.two_goal()


def test_two_goal_tables_match_reference(grid_ref, tg):
    pb, _ = X.bootstrap_fraction(tg, 3)
    np.testing.assert_allclose(pb, grid_ref["pboot"], atol=1e-12)
    np.testing.assert_allclose(np.asarray(X.infinite_q(tg, G, pb=pb)), grid_ref["Qstd"], atol=1e-6)
    np.testing.assert_allclose(np.asarray(X.infinite_q(tg, G)), grid_ref["Qpeb"], atol=1e-6)
    np.testing.assert_allclose(np.asarray(X.finite_horizon_q(tg, G, 3)[3]), grid_ref["Qta3"], atol=1e-6)


def test_two_goal_paper_numbers(tg):
    pb, _ = X.bootstrap_fraction(tg, 3)
    ng = ~tg.is_terminal
    assert sorted(set(np.round(pb[ng], 6))) == [0.666667, 0.675676, 0.714286, 0.725806]
    Qs = np.asarray(X.infinite_q(tg, G, pb=pb))
    adj = {(0, 4): 49.0, (1, 5): 49.0, (5, 1): 19.0}            # paper §2.2 'converge to 49 and 19'
    for cell, v in adj.items():
        assert abs(Qs[tg.names.index(cell)].max() - v) < 1e-9
    Qta = X.finite_horizon_q(tg, G, 3)
    pi_ta = jnp.stack([X.greedy_onehot(Qta[3 - t]) for t in range(3)])
    ps, pp = X.greedy_onehot(jnp.asarray(Qs)), X.greedy_onehot(X.infinite_q(tg, G))
    m = lambda V: float(np.asarray(V)[ng].mean())  # noqa: E731
    assert abs(m(X.policy_value_finite(tg, G, pi_ta)) - 17.4256) < 1e-4
    assert abs(m(X.policy_value_finite(tg, G, jnp.stack([ps] * 3))) - 16.0000) < 1e-4
    assert abs(m(X.policy_value_finite(tg, G, jnp.stack([pp] * 3))) - 11.1065) < 1e-4
    assert abs(m(X.policy_value_infinite(tg, G, pp)) - 42.4902) < 1e-4
    assert abs(m(X.policy_value_infinite(tg, G, ps)) - 35.6883) < 1e-4
    diff = [tg.names[i] for i in np.flatnonzero(ng) if
            np.asarray(X.infinite_q(tg, G))[i, int(np.argmax(Qs[i]))] < np.asarray(X.infinite_q(tg, G))[i].max() - 1e-9]
    assert diff == [(3, 0), (3, 1), (5, 1), (5, 2), (5, 3)]


# ------------------------------------------------------------------ 3. Queue of Cars 
def test_queue_of_cars_dp():
    out = D.qoc_outcomes(D.QoCParams())
    v, P = X.qoc_optimal(out, 0.99, 15)
    assert abs(v - 0.528557) < 1e-5
    assert abs(X.qoc_optimal(out, 1.0, 15)[0] - 0.602) < 5e-4
    row = lambda t: "".join("-" if p == 0.5 else str(int(p)) for p in P[t])  # noqa: E731
    assert [row(t) for t in (0, 3, 6, 9, 14)] == ["000000000", "100000000", "111000000", "---110000", "--------1"]
    # policy evaluation of the optimal table reproduces V*
    assert abs(float(X.qoc_policy_value(jnp.asarray(np.where(P == 0.5, 0.0, P)), out, 0.99)[0]) - v) < 1e-9
    const = lambda p: float(X.qoc_policy_value(jnp.full((15, 9), p), out, 0.99)[0])  # noqa: E731
    assert abs(const(0.4708) - 0.398990) < 2e-6
    assert abs(const(0.0) - 0.267876) < 1e-6 and abs(const(1.0) - 0.316404) < 1e-6
    det = jnp.asarray(np.broadcast_to(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0.0]), (15, 9)))
    assert abs(float(X.qoc_policy_value(det, out, 0.99)[0]) - 0.385153) < 1e-6


def test_queue_of_cars_best_memoryless():
    val, p = X.qoc_best_memoryless(D.qoc_outcomes(D.QoCParams()), 0.99, 15, n_restarts=16, steps=1500)
    assert abs(val - 0.398990) < 1e-5 and np.all(np.abs(p - 0.4708) < 2e-3)


def test_check_grads_qoc_exact_return():
    out = D.qoc_outcomes(D.QoCParams())
    f = lambda lg: X.qoc_policy_value(jax.nn.sigmoid(lg), out, 0.99)[0]  # noqa: E731
    lg = jax.random.normal(jax.random.PRNGKey(0), (15, 9), jnp.float64)
    check_grads(f, (lg,), order=2, modes=["rev"], atol=1e-6, rtol=1e-6)


# ------------------------------------------------------------------ 4. constant chain through OUR GAE 
@pytest.mark.parametrize("gamma,T,td0,gae,mc", [(0.99, 1000, 90.9918, 90.8604, 90.1004),
                                                (0.99, 300, 75.1880, 74.1645, 68.6183),
                                                (0.99, 50, 33.5570, 27.0097, 21.7912),
                                                (1.0, 1000, 1000.0, 981.000, 500.5),
                                                (1.0, 300, 300.0, 281.000, 150.5),
                                                (1.0, 50, 50.0, 35.168, 25.5)])
def test_constant_chain_aliased_critic(gamma, T, td0, gae, mc):
    for lam, want in ((0.0, td0), (0.95, gae), (1.0, mc)):
        got = float(X.chain_fixed_point(gamma, lam, T, peb=False))
        assert abs(got - want) < 1e-3 * max(1.0, abs(want)) * 0.1 + 1e-3, (lam, got, want)
        assert abs(got - X.chain_fixed_point_closed_form(gamma, lam, T)) < 1e-6 * max(1, want)
    if gamma < 1:
        for lam in (0.0, 0.95, 1.0):  # PEB is unbiased for every lambda: v = 1/(1-gamma)
            assert abs(float(X.chain_fixed_point(gamma, lam, T, peb=True)) - 1 / (1 - gamma)) < 1e-6


# ------------------------------------------------------------------ 5. Difficult Gridworld 
def test_difficult_grid():
    dg = D.difficult_grid()
    Q = X.infinite_q(dg, 1.0, iters=2000)
    assert float(Q[D.DGRID_START].max()) == -54.0
    assert abs(float(X.dgrid_eps_greedy_return(dg, Q, 0.1, 200, D.DGRID_START)) - (-60.2794)) < 1e-4
    assert abs(float(X.dgrid_eps_greedy_return(dg, Q, 0.1, 200, D.DGRID_START, first=False)) - (-60.1328)) < 1e-4


# ------------------------------------------------------------------ 6. GAE with PEB: hand example + baselines parity
def test_gae_peb_hand_example():
    """5 steps, gamma 0.9, lambda 0.5; env termination at t=1, timeout at t=3 (hand-computed)."""
    r = jnp.ones(5)
    V = jnp.array([1.0, 2, 3, 4, 5])
    F = jnp.array([2.0, 10, 4, 20, 6])        # V(s'_t): = V[t+1] unless reset; t=1, t=3 pre-reset; t=4 closes
    term = jnp.array([0.0, 1, 0, 0, 0])
    reset = jnp.array([0.0, 1, 0, 1, 0])
    adv, ret = A.gae_peb(r, V, F, term, reset, 0.9, 0.5, 1.0)
    np.testing.assert_allclose(np.asarray(adv), [1.35, -1.0, 8.35, 15.0, 1.4], atol=1e-12)
    np.testing.assert_allclose(np.asarray(ret), np.asarray(adv + V), atol=1e-12)
    adv0, _ = A.gae_peb(r, V, F, term, reset, 0.9, 0.5, 0.0)
    np.testing.assert_allclose(np.asarray(adv0), [1.35, -1.0, 0.25, -3.0, 1.4], atol=1e-12)


def test_gae_no_peb_equals_vendored_baselines_gae():
    """peb = 0 is the paper's standard agent = baselines ppo2 GAE (the PPO-details reproduction vendored `gae`, dones-with-obs)."""
    rng = np.random.default_rng(0)
    M, N = 50, 3
    reset = (rng.random((M, N)) < 0.2).astype(np.float64)
    term = reset * (rng.random((M, N)) < 0.5)
    V = rng.normal(size=(M, N))
    nv = rng.normal(size=N)
    F = np.concatenate([V[1:], nv[None]], 0)
    F = np.where(reset > 0, rng.normal(size=(M, N)), F)  # pre-reset obs values are irrelevant without PEB
    rw = rng.normal(size=(M, N))
    a_ours, r_ours = A.gae_peb(jnp.asarray(rw), jnp.asarray(V), jnp.asarray(F), jnp.asarray(term), jnp.asarray(reset),
                               0.99, 0.95, 0.0)
    dones = np.concatenate([np.zeros((1, N)), reset[:-1]], 0)
    a_ref, r_ref = P8.gae(jnp.asarray(rw), jnp.asarray(V), jnp.asarray(dones), jnp.asarray(nv), jnp.asarray(reset[-1]),
                          0.99, 0.95)
    np.testing.assert_allclose(np.asarray(a_ours), np.asarray(a_ref), atol=1e-10)
    # and PEB equals the vendored 'truncation-correct' variant (PPO-details extension) with truncs = timeouts
    trunc = reset * (1 - term)
    a_p, _ = A.gae_peb(jnp.asarray(rw), jnp.asarray(V), jnp.asarray(F), jnp.asarray(term), jnp.asarray(reset),
                       0.99, 0.95, 1.0)
    a_t, _ = P8.gae(jnp.asarray(rw), jnp.asarray(V), jnp.asarray(dones), jnp.asarray(nv), jnp.asarray(reset[-1]),
                    0.99, 0.95, truncs=jnp.asarray(trunc), final_values=jnp.asarray(F))
    np.testing.assert_allclose(np.asarray(a_p), np.asarray(a_t), atol=1e-10)


# ------------------------------------------------------------------ 7. time feature (paper §1; Tonic TimeFeature)
def test_time_feature_modes():
    T = 8
    t = jnp.arange(T + 1)
    tf = lambda m: np.asarray(jax.vmap(lambda x: D.time_feature(x, T, D.TIME_MODES[m]))(t))  # noqa: E731
    np.testing.assert_allclose(tf("elapsed_pm1"), -1 + 2 * np.arange(T + 1) / T, atol=1e-6)
    np.testing.assert_allclose(tf("remaining_pm1"), 1 - 2 * np.arange(T + 1) / T, atol=1e-6)
    np.testing.assert_allclose(tf("remaining_raw"), T - np.arange(T + 1), atol=1e-6)
    assert np.all(tf("none") == 0)


# ------------------------------------------------------------------ 8. check_grads on the trainer's PPO loss
@pytest.mark.parametrize("cont", [False, True])
def test_check_grads_trainer_ppo_loss(cont):
    from src import model as M
    from src.train import PPO, load_config
    cfg = load_config(os.path.join(REPO, "src", "configs", "e3_queue_of_cars.yaml"))
    ppo = PPO(cfg)
    if cont:
        ppo.cont, ppo.net = True, M.build(2, True, 16)
    else:
        ppo.net = M.build(2, False, 16)
    k = jax.random.split(jax.random.PRNGKey(3), 6)
    B = 24
    x = jax.random.normal(k[0], (B, 10), jnp.float64)
    params = jax.tree.map(lambda z: z.astype(jnp.float64), ppo.net.init(k[1], x[:1]))
    params = jax.tree.map(lambda z: z + 0.1 * jax.random.normal(k[2], z.shape, z.dtype), params)
    dist, v = ppo.net.apply(params, x)
    a = M.sample(k[3], dist, cont)
    mb = dict(x=x, a=a, logp=M.log_prob(dist, a, cont) + 0.05 * jax.random.normal(k[4], (B,), jnp.float64),
              v=v + 0.1, ret=jax.random.normal(k[5], (B,), jnp.float64), adv=jax.random.normal(k[4], (B,)))
    check_grads(lambda p: ppo._loss(p, mb)[0], (params,), order=2, modes=["rev"], atol=1e-5, rtol=1e-5)
