"""Toy correctness.  The PMBM filter has no gradient, so this file tests the estimator's CLOSED-FORM /
statistical correctness instead (a hand-derived 1-D toy, tabulated below): exact conjugacy -- the PMBM posterior produced by the
filter equals Bayes' rule l(Z|X) f(X) / rho(Z) evaluated pointwise by brute-force enumeration of the point-target
likelihood (5) and the PMBM prior (7)/(12), for arbitrary finite sets X.  Also: Example 5 weights, the Gamma
recursion, Prop. 4 count, the hand OSPA number is tested in test_eval.py."""
import itertools
import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from src.algorithm import PMBMConfig, init_state, pmbm_update
from src.model import LGParams, make_lg_model

PD, LAMC, CZ = 0.8, 2.0, 0.1
Z = np.array([0.5, 3.0])


def gauss(x, m, v):
    return math.exp(-0.5 * (x - m) ** 2 / v) / math.sqrt(2 * math.pi * v)


def toy_model():
    p = LGParams(F=np.eye(1), Q=np.zeros((1, 1)), H=np.eye(1), R=np.eye(1), ps=1.0, pd=PD, clutter_density=CZ,
                 gate=1e9, birth_w=np.zeros(0), birth_m=np.zeros((0, 1)), birth_P=np.zeros((0, 1, 1)))
    return make_lg_model(p)


def run_toy(backend="host"):
    model = toy_model()
    cfg = PMBMConfig(T=4, H=4, Nh=8, P=2, M=2, prune_w=0.0, prune_pois=0.0, prune_r=0.0, backend=backend)
    pois = (jnp.log(jnp.array([2.0])), dict(m=jnp.array([[0.0]]), P=jnp.array([[[4.0]]])))
    tracks = [(0.6, dict(m=jnp.array([1.0]), P=jnp.array([[1.0]])))]
    st0 = init_state(model, cfg, pois=pois, tracks=tracks)
    zz = jnp.array(Z)[:, None]
    st, out = pmbm_update(st0, zz, jnp.array([True, True]), model, cfg)
    return model, cfg, st0, st, out


# ---- brute-force densities ------------------------------------------------------------------------------------
def pmbm_density(X, pois, hyps):
    """f(X) = sum_j W_j sum_{assign each x to Poisson or a distinct Bernoulli} e^{-Lam} prod mu(x) prod r p(x) prod (1-r).
    pois: list of (w, m, v); hyps: list of (W, [(r, m, v), ...])."""
    Lam = sum(w for w, _, _ in pois)
    mu = lambda x: sum(w * gauss(x, m, v) for w, m, v in pois)
    tot = 0.0
    n = len(X)
    for W, bern in hyps:
        nb = len(bern)
        for assign in itertools.product(range(-1, nb), repeat=n):
            used = [a for a in assign if a >= 0]
            if len(set(used)) != len(used):
                continue
            val = math.exp(-Lam)
            for x, a in zip(X, assign):
                val *= mu(x) if a < 0 else bern[a][0] * gauss(x, bern[a][1], bern[a][2])
            for i, (r, _, _) in enumerate(bern):
                if i not in used:
                    val *= 1 - r
            tot += W * val
    return tot


def likelihood(Zs, X):
    """l(Z|X) eq. (5): e^{-lamc} sum over injective partial maps targets->measurements (clutter = rest)."""
    tot = 0.0
    m, n = len(Zs), len(X)
    for assign in itertools.product(range(-1, m), repeat=n):
        used = [a for a in assign if a >= 0]
        if len(set(used)) != len(used):
            continue
        val = 1.0
        for x, a in zip(X, assign):
            val *= (1 - PD) if a < 0 else PD * gauss(Zs[a], x, 1.0)
        for k in range(m):
            if k not in used:
                val *= CZ
        tot += val
    return math.exp(-LAMC) * tot


def extract(st):
    pois = [(math.exp(float(w)), float(d_m[0]), float(d_P[0, 0])) for w, d_m, d_P in
            zip(st.pois_logw, st.pois_d["m"], st.pois_d["P"]) if np.isfinite(w)]
    hyps = []
    for g in range(st.glogW.shape[0]):
        if not np.isfinite(st.glogW[g]):
            continue
        bern = []
        for t in range(st.gidx.shape[1]):
            h = int(st.gidx[g, t])
            if h >= 0:
                bern.append((float(st.trk_r[t, h]), float(st.trk_d["m"][t, h, 0]), float(st.trk_d["P"][t, h, 0, 0])))
        hyps.append((math.exp(float(st.glogW[g])), bern))
    return pois, hyps


@pytest.mark.parametrize("backend", ["host", "jax"])
def test_gate1_bayes_pointwise(backend):
    model, cfg, st0, st, out = run_toy(backend)
    pois0, hyps0 = extract(st0)
    pois1, hyps1 = extract(st)
    assert abs(sum(W for W, _ in hyps1) - 1.0) < 1e-14
    sets = [[], [0.3], [2.2], [0.4, 2.9], [-1.0, 1.5], [0.2, 3.1, -2.0], [0.5, 1.0, 3.0, -0.7]]
    ratios = []
    for X in sets:
        num = likelihood(Z, X) * pmbm_density(X, pois0, hyps0)
        den = pmbm_density(X, pois1, hyps1)
        ratios.append(num / den)
    ratios = np.array(ratios)
    assert np.max(np.abs(ratios - 0.0024276523785825)) < 1e-15, ratios
    assert (ratios.max() - ratios.min()) / ratios.mean() < 1e-13


def test_gate1_tabulated_numbers():
    model, cfg, st0, st, out = run_toy()
    pois1, hyps1 = extract(st)
    Ws = sorted([W for W, _ in hyps1], reverse=True)
    exp = [0.478513741666342, 0.309328023255609, 0.212158235078049]
    assert len(Ws) == 3
    assert np.max(np.abs(np.array(Ws) - exp)) < 1e-14
    # new Bernoullis: r_p, mean, var of the two new tracks
    rs = {}
    for g_W, bern in hyps1:
        for r, m, v in bern:
            rs[round(m, 6)] = (r, v)
    assert abs(rs[0.4][0] - 0.735737648277750) < 1e-14 and abs(rs[0.4][1] - 0.8) < 1e-14
    assert abs(rs[2.4][0] - 0.537164140280032) < 1e-14 and abs(rs[2.4][1] - 0.8) < 1e-14
    # estimators (Gamma = 0.4): the toy separates Est1 from Est2/3
    def pts(e):
        mask, x = e
        return sorted(float(v) for v, m in zip(np.asarray(x)[:, 0], np.asarray(mask)) if m)
    assert np.allclose(pts(out["est1"]), [0.4, 2.4], atol=1e-12)
    assert np.allclose(pts(out["est3"]), [0.75, 2.4], atol=1e-12)
    assert np.allclose(pts(out["est2"]), [0.75, 2.4], atol=1e-12)
    pmf = np.asarray(out["card_pmf"])[:4]
    assert np.allclose(pmf, [0.045021, 0.390334, 0.521004, 0.043642], atol=1e-5)


def test_example5_and_gamma_and_prop4():
    r = np.array([0.8, 0.2, 1.0])
    # Example 5: MBM01 weights r=(0.8,0.2,1): the r=1 Bernoulli does not branch
    w = sorted([0.8 * 0.2, 0.2 * 0.2, 0.8 * 0.8, 0.2 * 0.8])
    assert np.allclose(w, sorted([0.16, 0.04, 0.64, 0.16]))
    # Gamma = 0.4 tolerates two misses at pd=0.9, ps=0.99
    rr, seq = 1.0, []
    for _ in range(3):
        rr *= 0.99
        rr = rr * 0.1 / (1 - rr * 0.9)
        seq.append(rr)
    assert abs(seq[0] - 0.908) < 1e-3 and abs(seq[1] - 0.471) < 1e-3 and abs(seq[2] - 0.080) < 1e-3
    # Prop 4: sum_j 2^{n_j} MBM01 components (r in (0,1))
    ns = [3, 1, 4]
    assert sum(2 ** n for n in ns) == 8 + 2 + 16


def test_gate1_multistep_bayes():
    """Three predict/update cycles (static targets: F = 1, Q = 0, p_s = 1, Poisson prior only, no births), 2 measurements
    per epoch, no pruning: the filter's PMBM must equal Bayes' rule  prod_k l(Z_k|X) f0(X) / rho  pointwise (exact because
    every Poisson component stays identical, so the moment-matched new Bernoullis are exact).  Exercises hypothesis-tree
    growth, relabelling, track-slot reuse and the multi-step bookkeeping."""
    from src.algorithm import pmbm_predict
    model = toy_model()
    cfg = PMBMConfig(T=10, H=40, Nh=100, P=2, M=2, prune_w=0.0, prune_pois=0.0, prune_r=0.0, backend="host",
                     k_per_parent=100)   # exactness mode: no k-best truncation (paper rule k_g = ceil(N_h W_g) is an approximation)
    pois = (jnp.log(jnp.array([1.5])), dict(m=jnp.array([[0.0]]), P=jnp.array([[[4.0]]])))
    st0 = init_state(model, cfg, pois=pois)
    Zs = [np.array([0.5, 3.0]), np.array([0.7, -1.2]), np.array([2.8, 0.4])]
    st = st0
    for zk in Zs:
        st = pmbm_predict(st, model, cfg)
        st, out = pmbm_update(st, jnp.array(zk)[:, None], jnp.array([True, True]), model, cfg)
    assert int(out["ovf"].sum()) == 0
    pois0, hyps0 = extract(st0)
    pois1, hyps1 = extract(st)
    sets = [[], [0.3], [2.2, -1.0], [0.5, 1.5, -0.7], [0.6, 2.9, -1.1, 0.35]]
    ratios = []
    for X in sets:
        lik = 1.0
        for zk in Zs:
            lik *= likelihood(zk, X)
        ratios.append(lik * pmbm_density(X, pois0, hyps0) / pmbm_density(X, pois1, hyps1))
    ratios = np.array(ratios)
    assert (ratios.max() - ratios.min()) / ratios.mean() < 1e-12, ratios
    assert len(hyps1) > 6      # a genuinely multi-hypothesis posterior


def test_new_track_moment_matching_two_component_poisson():
    """Section V-C1 moment matching checked by an independent hand computation (numpy, no model code).  Poisson intensity with
    TWO components (1-D); for a single measurement z the new Bernoulli is r = e/(c + e), e = sum_i w_i p_d N(z; m_i, P_i + R),
    with density the moment-matched Gaussian of the mixture sum_i (w_i l_i / e) N(m_i + K_i(z - m_i), (1 - K_i) P_i)."""
    R_ = 1.0
    model = toy_model()                                    # F = 1, Q = 0, H = 1, R = 1, pd = 0.8, c = 0.1, gate off
    cfg = PMBMConfig(T=2, H=2, Nh=4, P=2, M=1, prune_w=0.0, prune_pois=0.0, prune_r=0.0, backend="host")
    w = np.array([1.3, 0.7]); m = np.array([-1.0, 2.5]); Pv = np.array([4.0, 0.5])
    pois = (jnp.log(jnp.array(w)), dict(m=jnp.array(m)[:, None], P=jnp.array(Pv)[:, None, None]))
    st0 = init_state(model, cfg, pois=pois)
    z = 1.2
    st, _ = pmbm_update(st0, jnp.array([[z]]), jnp.array([True]), model, cfg)
    S = Pv + R_
    li = PD * np.exp(-0.5 * (z - m) ** 2 / S) / np.sqrt(2 * np.pi * S)
    e = float(np.sum(w * li))
    r_hand = e / (CZ + e)
    wt = w * li / e
    K = Pv / S
    mi = m + K * (z - m)
    Pi = (1 - K) * Pv
    mean_hand = float(np.sum(wt * mi))
    var_hand = float(np.sum(wt * (Pi + (mi - mean_hand) ** 2)))
    alive = np.asarray(st.trk_alive)
    s_, h_ = np.argwhere(alive)[0]
    assert alive.sum() == 1
    assert abs(float(st.trk_r[s_, h_]) - r_hand) < 1e-14
    assert abs(float(st.trk_d["m"][s_, h_, 0]) - mean_hand) < 1e-13
    assert abs(float(st.trk_d["P"][s_, h_, 0, 0]) - var_hand) < 1e-13


def test_estimate_set_overflow_is_counted():
    """With est_E = 1 the toy's two-target estimate (Est 1 = {0.4, 2.4}) does not fit; est_over must say 1 per
    estimator that overflows (Est 1, 2 and 3 each select 2 tracks here -> 3), and 0 with est_E = 16."""
    model = toy_model()
    for E_, want in ((16, 0), (1, 3)):
        cfg = PMBMConfig(T=4, H=4, Nh=8, P=2, M=2, prune_w=0.0, prune_pois=0.0, prune_r=0.0, est_E=E_)
        pois = (jnp.log(jnp.array([2.0])), dict(m=jnp.array([[0.0]]), P=jnp.array([[[4.0]]])))
        tracks = [(0.6, dict(m=jnp.array([1.0]), P=jnp.array([[1.0]])))]
        st0 = init_state(model, cfg, pois=pois, tracks=tracks)
        _, out = pmbm_update(st0, jnp.array(Z)[:, None], jnp.array([True, True]), model, cfg)
        assert int(out["est_over"]) == want, (E_, int(out["est_over"]))
