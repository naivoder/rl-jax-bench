"""Toy tests with closed-form / exactly computable answers and finite-difference gradient
checks. Run on CPU in float64: JAX_PLATFORMS=cpu pytest tests/test_grads.py
"""
import numpy as np
import pytest
from scipy.stats import ncx2

from src import common  # noqa: F401  (x64)
import jax
import jax.numpy as jnp
import optax
from jax.test_util import check_grads

from src import algorithm as alg
from src import baselines as bl
from src import data
from src import env as E
from src import eval as ev
from src import physics as ph
from src.common import EnvCfg, load_yaml

CFG = EnvCfg()


# ----------------------------------------------------------------------------- SS
@pytest.mark.parametrize("snr0,tau,r,ref", [(1, 1.5, 17000, 61952.535258), (1, 0.625, 9500, 8485.404618),
                                            (10, 1.5, 17000, 9663.171842), (10, 2.0, 13000, 2054.405844),
                                            (100, 0.625, 9500, 149.008559)])
def test_g1_ss_closed_form_matches_worked_example(snr0, tau, r, ref):
    assert abs(float(ph.c_ss(tau, r, snr0, CFG)) - ref) <= 1e-6   # worked example printed to 6 decimals


def test_g1_ss_ekf_converges_to_closed_form():
    """Our EKF at a FIXED linearisation point (zero velocity, zero innovation) converges to (<= 1e-9 rel)."""
    rng = np.random.default_rng(1)
    worst = 0.0
    step = jax.jit(lambda x, P, vr, vt: ph.ekf_update(*ph.predict(x, P, CFG.T0, CFG.sigma_w_sq),
                                                       ph.h_polar(ph.predict(x, P, CFG.T0, CFG.sigma_w_sq)[0]), vr, vt))
    for _ in range(8):
        tau, r, th, snr0 = rng.uniform(0.1, 2.5), rng.uniform(3000, 20000), rng.uniform(-np.pi, np.pi), \
            10 ** rng.uniform(-0.5, 2)
        x = jnp.array([r * np.cos(th), r * np.sin(th), 0.0, 0.0])
        P = jnp.diag(jnp.array(CFG.P0_diag))
        snr = ph.track_snr(tau, r, snr0, CFG.r0, CFG.tau0, CFG.r_min_snr)
        vr, vt = ph.meas_var(snr, CFG.sigma_r0_sq, CFG.sigma_th0_sq)
        for _k in range(3000):
            x, P = step(x, P, vr, vt)
        ref = float(ph.c_ss(tau, r, snr0, CFG))
        worst = max(worst, abs(float(ph.pos_trace(P)) - ref) / ref)
    print("SS worst rel", worst)
    assert worst <= 1e-9


def test_eq6_division_monotone():
    """A-01: variance decreases with dwell and increases with range."""
    s1 = ph.track_snr(0.5, 10000.0, 2.0, CFG.r0, CFG.tau0, 100.0)
    s2 = ph.track_snr(1.0, 10000.0, 2.0, CFG.r0, CFG.tau0, 100.0)
    s3 = ph.track_snr(1.0, 15000.0, 2.0, CFG.r0, CFG.tau0, 100.0)
    v1, v2, v3 = (ph.meas_var(s, 16.0, 1e-6)[0] for s in (s1, s2, s3))
    assert v2 < v1 and v3 > v2


# ----------------------------------------------------------------------------- PD
def test_g1_pd_lut():
    tab = {0.01: 0.7394, 0.10: 2.6327, 0.50: 6.5199, 0.90: 12.0417, 0.99: 17.4666}
    for pd, snr in tab.items():
        assert abs(float(ph.shnidman_snr(pd, 1e-3)) - snr) < 1e-4
    thr = -2 * np.log(1e-3)
    snrs = ph.shnidman_snr(ph.PD_GRID, 1e-3)
    exact = ncx2.sf(thr, 2, 2 * snrs)
    assert np.max(np.abs(exact - ph.PD_GRID)) <= 0.02
    lut = ph.make_lut(1e-3)
    assert np.allclose(np.asarray(ph.pd_lookup(jnp.asarray(snrs), lut)), ph.PD_GRID)
    assert float(ph.pd_lookup(jnp.array(0.0), lut)) == 0.01 and float(ph.pd_lookup(jnp.array(1e6), lut)) == 0.99


# ----------------------------------------------------------------------------- 3-of-4 chain
@pytest.mark.parametrize("p", [1.0, 0.9, 0.5, 0.2])
def test_g1_m4_chain(p):
    """3-of-4 chain with the env's transition tables, constant P_d p, 1e5 MC targets: E[scans] within 3 SE of."""
    n = 100_000

    def body(c, k):
        st, done, cnt = c
        det = jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(7), k), (n,)) < p
        nxt = jnp.where(det, ph.NEXT_DET[st], ph.NEXT_MISS[st]).astype(jnp.int32)
        cnt = cnt + (~done)
        done = done | (nxt == 6)
        return (jnp.where(nxt == 6, 0, nxt), done, cnt), None

    (st, done, cnt), _ = jax.lax.scan(body, (jnp.zeros(n, jnp.int32), jnp.zeros(n, bool), jnp.zeros(n)), jnp.arange(4000))
    assert bool(done.all())
    m, se = float(cnt.mean()), float(cnt.std() / np.sqrt(n))
    ex = ph.expected_scans(p)
    print("chain", p, m, ex, se)
    assert abs(m - ex) <= max(3 * se, 1e-12)


def test_g1_m4_exact_values():
    for p, ref in [(1.0, 3.0), (0.9, 3.3774), (0.5, 8.7692), (0.2, 68.1488), (0.1, 424.9867)]:
        assert abs(ph.expected_scans(p) - ref) < 1e-4


# ----------------------------------------------------------------------------- OA
def _oa_example():
    r = jnp.array([17200.0, 9500.0, 1e4, 1e4, 1e4])
    act = jnp.array([1, 1, 0, 0, 0], bool)
    w = jnp.ones(5)
    one = jnp.ones(5)
    return r, act, w, bl.ss_dcost(CFG, 2.8, r, one), bl.ss_d2cost(CFG, 2.8, r, one)


def test_g1_oa_worked_example():
    r, act, w, dc, d2c = _oa_example()
    for tau, mu in [bl.allocate(dc, w, act, 2.25, 2.5), bl.allocate_newton(dc, d2c, w, act, 2.25, 2.5)]:
        t = np.asarray(tau) / 2.5
        assert abs(t[0] - 0.737538) < 1e-6 and abs(t[1] - 0.162462) < 1e-6
        assert abs(float(mu) - 26624.96) / 26624.96 < 1e-6
        costs = [float(ph.c_ss(tau[i], r[i], 2.8, CFG)) for i in range(2)]
        assert abs(costs[0] - 24347.10) < 0.01 and abs(costs[1] - 5148.46) < 0.01
    # KKT: -T0 dc/dtau equal across active targets (= mu)
    tau, mu = bl.allocate(dc, w, act, 2.25, 2.5)
    k = -2.5 * np.asarray(dc(tau))[:2]
    assert abs(k[0] - k[1]) / k[0] < 1e-6


def test_g1_oa_subgradient_agrees():
    r, act, w, dc, _ = _oa_example()
    t_b, mu_b = bl.allocate(dc, w, act, 2.25, 2.5)
    t_s, mu_s = jax.jit(lambda: bl.oa_subgradient(dc, w, act, 2.25, 2.5))()
    assert np.max(np.abs(np.asarray(t_s - t_b))) / 2.25 < 1e-4
    assert abs(float(mu_s - mu_b)) / float(mu_b) < 1e-3


def test_g1_oa_newton_equals_bisection_random():
    rng = np.random.default_rng(0)
    fb = jax.jit(lambda r, w, act, s: bl.allocate(bl.ss_dcost(CFG, s, r, jnp.ones(5)), w, act, 2.25, 2.5)[0])
    fn = jax.jit(lambda r, w, act, s: bl.allocate_newton(bl.ss_dcost(CFG, s, r, jnp.ones(5)),
                                                         bl.ss_d2cost(CFG, s, r, jnp.ones(5)), w, act, 2.25, 2.5)[0])
    worst = 0.0
    for _ in range(100):
        r = jnp.asarray(rng.uniform(200, 20000, 5))
        w = jnp.asarray(rng.choice([1.0, 5.0, 10.0], 5))
        act = jnp.asarray(rng.random(5) < 0.6)
        s = float(rng.choice([0.5, 1.86, 30.0]))
        worst = max(worst, float(jnp.max(jnp.abs(fb(r, w, act, s) - fn(r, w, act, s)))) / 2.25)
    assert worst < 1e-6


def test_oa_myopic_forms_agree_and_ss_ordering():
    rng = np.random.default_rng(3)
    for _ in range(20):
        A = rng.normal(size=(4, 4))
        P = jnp.asarray(A @ A.T * 100 + np.diag([1e3, 1e3, 10, 10]))
        x = jnp.asarray([rng.uniform(3e3, 2e4), rng.uniform(-1e4, 1e4), 1.0, -1.0])
        tau = rng.uniform(0.01, 2.5)
        a = float(bl.myopic_cost(CFG, 1.86, x, P, 1.0, tau))
        b = float(bl.myopic_cost_info(CFG, 1.86, x, P, 1.0, tau))
        assert abs(a - b) / b < 1e-8


# ----------------------------------------------------------------------------- DUAL
def test_g1_dual_sequence():
    lam = jnp.float64(5000.0)
    seq = [float(lam)]
    for g in [-0.9, -0.9, 0.1, 0.1, 0.1]:
        lam = E.dual_update(CFG, lam, g)
        seq.append(float(lam))
    assert seq == [5000.0, 500.0, 0.0, 500.0, 1000.0, 1500.0]


def test_a13_overflow_and_a12_empty():
    act = jnp.array([1, 1, 1, 0, 0], bool)
    tau, ts = E.execute_dwells(CFG, act, jnp.array([2.0, 2.0, 1.0, 2.5, 2.5]))
    assert abs(float(tau.sum()) - 2.5) < 1e-12 and float(ts) == 0.0
    assert float(tau[3]) == 0.0 and float(tau[4]) == 0.0
    assert np.allclose(np.asarray(tau[:3]), np.array([2.0, 2.0, 1.0]) * 2.5 / 5.0)
    tau, ts = E.execute_dwells(CFG, act, jnp.array([0.5, 0.5, 0.5, 2.5, 2.5]))
    assert abs(float(ts) - 1.0) < 1e-12


# ----------------------------------------------------------------------------- CRN
def test_g1_crn_fixed_delta_baselines_identical_initiation():
    """: EA, DA, OA at the same delta have identical N^miss sequences and confirmation slots (CRN)."""
    y = load_yaml()
    lut = ph.make_lut(CFG.pfa)
    consts = dict(snr0=jnp.float64(1.86), s_ref=jnp.float64(5.0), delta=jnp.float64(0.9))
    for i in range(3):
        k = jax.random.PRNGKey(100 + i)
        scen = data.spawn_scenario(jax.random.fold_in(k, 0x5A), CFG, 600, 3)
        res = []
        for kind in ("ea", "da", "oa"):
            outs, esT = jax.jit(lambda s, kk, kind=kind: ev.rollout(CFG, lut, ev.make_policy(kind, CFG, y), None,
                                                                    consts, s, kk, 600))(scen, k)
            res.append((np.asarray(outs["nmiss"]), np.asarray(esT.conf_slot)))
        for r_ in res[1:]:
            assert np.array_equal(r_[0], res[0][0]) and np.array_equal(r_[1], res[0][1])


# ----------------------------------------------------------------------------- DPG
def test_g1_dpg_analytic_actor_gradient():
    """Q(s,a) = -||a - A s||^2, mu = sigmoid(W s + b) T0: autodiff = closed form. Convergence check with a
    REPRESENTABLE target c(s) = T0 sigmoid(W* s + b*) (an affine A s is not representable by a sigmoid actor):
    Adam on the DPG objective drives mu to c(s)."""
    T0 = 2.5
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.uniform(0, 0.4, (5, 11)))
    Wst, bst = jnp.asarray(rng.normal(0, 1.0, (5, 11))), jnp.asarray(rng.normal(0, 0.5, 5))
    S = jnp.asarray(rng.uniform(0, 1, (64, 11)))
    p = dict(W=jnp.asarray(rng.normal(0, 0.1, (5, 11))), b=jnp.zeros(5))
    actor = lambda p, s: jax.nn.sigmoid(s @ p["W"].T + p["b"]) * T0
    critic = lambda pc, s, a: -jnp.sum((a - s @ A.T) ** 2, -1)
    g = jax.grad(alg.actor_loss)(p, None, dict(s=S), actor, critic)
    z = S @ p["W"].T + p["b"]
    mu = jax.nn.sigmoid(z) * T0
    dmu = T0 * jax.nn.sigmoid(z) * (1 - jax.nn.sigmoid(z))
    gW = (2 * (mu - S @ A.T) * dmu).T @ S / S.shape[0]          # gradient of -J
    assert np.allclose(np.asarray(g["W"]), np.asarray(gW), rtol=1e-10, atol=1e-12)
    tgt = lambda s: jax.nn.sigmoid(s @ Wst.T + bst) * T0
    critic2 = lambda pc, s, a: -jnp.sum((a - tgt(s)) ** 2, -1)
    opt = optax.adam(0.05)
    os_ = opt.init(p)

    @jax.jit
    def stepf(p, os_):
        g = jax.grad(alg.actor_loss)(p, None, dict(s=S), actor, critic2)
        u, os_ = opt.update(g, os_, p)
        return optax.apply_updates(p, u), os_
    for _ in range(4000):
        p, os_ = stepf(p, os_)
    assert float(jnp.max(jnp.abs(actor(p, S) - tgt(S)))) < 1e-3


def test_g1_dpg_check_grads_losses():
    """Finite-difference check (order 2, float64) of the critic and actor losses with small float64 MLPs."""
    rng = np.random.default_rng(1)
    def mlp(p, x):
        return jnp.tanh(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
    pa = dict(W1=jnp.asarray(rng.normal(0, .3, (11, 8))), b1=jnp.zeros(8), W2=jnp.asarray(rng.normal(0, .3, (8, 5))),
              b2=jnp.zeros(5))
    pc = dict(W1=jnp.asarray(rng.normal(0, .3, (16, 8))), b1=jnp.zeros(8), W2=jnp.asarray(rng.normal(0, .3, (8, 1))),
              b2=jnp.zeros(1))
    actor = lambda p, s: jax.nn.sigmoid(mlp(p, s)) * 2.5
    critic = lambda p, s, a: mlp(p, jnp.concatenate([s, a / 2.5], -1))[..., 0]
    B = 16
    batch = dict(s=jnp.asarray(rng.normal(size=(B, 11))), a=jnp.asarray(rng.uniform(0, 2.5, (B, 5))),
                 r=jnp.asarray(rng.normal(size=B)), s2=jnp.asarray(rng.normal(size=(B, 11))), disc=jnp.ones(B))
    st = alg.DDPGState(pa, pc, pa, pc, None, None, jnp.zeros((), jnp.int32))
    check_grads(lambda q: alg.critic_loss(q, st, batch, actor, critic, 0.9)[0], (pc,), order=2, modes=["rev"])
    check_grads(lambda q: alg.actor_loss(q, pc, batch, actor, critic), (pa,), order=2, modes=["rev"])


# ----------------------------------------------------------------------------- DQN
def test_g1_dqn_max_target_vs_numpy():
    y = load_yaml()
    qcfg = alg.make_dql_cfg(y, 5, 64)
    st = alg.drqn.init(jax.random.PRNGKey(0), qcfg)
    st = st._replace(target_params=alg.drqn.init(jax.random.PRNGKey(1), qcfg).params)
    rng = np.random.default_rng(0)
    B = 32
    obs = rng.normal(size=(B, 2, 12)).astype(np.float32)
    act = rng.integers(0, 11, (B, 1)).astype(np.int32)
    rew = rng.normal(size=(B, 1)).astype(np.float32)
    batch = dict(inputs=jnp.asarray(obs), actions=jnp.asarray(act), rewards=jnp.asarray(rew),
                 discounts=jnp.full((B, 1), 0.9, jnp.float32), resets=jnp.zeros((B, 2), bool))
    loss, aux = alg.drqn.sequence_loss(st.params, st.target_params, batch, qcfg)
    q_on = np.asarray(alg.dql_q(qcfg, st.params, jnp.asarray(obs[:, 0])))
    q_tg = np.asarray(alg.dql_q(qcfg, st.target_params, jnp.asarray(obs[:, 1])))
    y_np = rew[:, 0] + 0.9 * q_tg.max(-1)
    l_np = np.mean((q_on[np.arange(B), act[:, 0]] - y_np) ** 2)
    assert abs(float(loss) - l_np) / l_np < 1e-5
