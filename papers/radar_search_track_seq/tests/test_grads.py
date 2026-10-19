"""toy-check toys (README) + finite-difference gradient checks. CPU, float64 where the toy is exact.

A   scan-value closed forms (p_max, Static = q, Coverage period mean, Random expectation) vs the environment
B1  UKF == KF on a linear measurement (worked example + random 6-D cases with the Stone Soup kappa = -3 weights)
B2  GNN brute force vs scipy linear_sum_assignment; greedy NN vs a numpy transcription of Stone Soup's loop
B3  printed-reward (flipped) revisit table on the 1-D CV toy through the environment's rtl_flip_terms
C1  masked BiGRU: padding invariance + differential vs flax.linen.GRUCell
C2  MHSA: permutation equivariance, padding invariance, differential vs a numpy multi-head attention
D   GOSPA worked example (36.742346)
E1  PPO clipped-surrogate gradient (closed form + check_grads); E2 GAE identities
G   check_grads of the full PPO loss through every encoder (float64 params)
"""
import math
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common  # noqa: F401,E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.test_util import check_grads  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from src.configs.config import EnvConfig, NetConfig  # noqa: E402
from src import data, algorithm as alg  # noqa: E402
from src.model import ActorCritic, BiGRU, MHSA, dummy_obs  # noqa: E402

CFG = replace(EnvConfig(), gospa_metrics=False)


# ----------------------------------------------------------------------------- A: scan value
def np_ssv_series(looks, cfg):
    """Independent float64 numpy: SSV before the push for a sequence of looks (deg pairs)."""
    s0, g, N = cfg.sigma0_deg, cfg.gamma_sv, cfg.n_sv
    pmax = sum(1 / (2 * math.pi * (s0 * (1 + g) ** k) ** 2) for k in range(N))
    out = []
    for t in range(len(looks)):
        v = 0.0
        for k in range(1, min(N, t) + 1):
            sig = s0 * (1 + g) ** k
            d2 = (looks[t][0] - looks[t - k][0]) ** 2 + (looks[t][1] - looks[t - k][1]) ** 2
            v += math.exp(-d2 / (2 * sig ** 2)) / (2 * math.pi * sig ** 2)
        out.append(v / pmax)
    return np.array(out)


def env_ssv_series(actions, cfg):
    s = data.reset(jax.random.PRNGKey(0), cfg)

    def body(s, a):
        s, _, info = data.step(s, a, cfg)
        return s, info["r_sv"]
    _, r = jax.lax.scan(body, s, jnp.asarray(actions, jnp.int32))
    return -np.asarray(r)


def test_A_ssv_closed_forms():
    cfg = CFG
    q = (1 + cfg.gamma_sv) ** -2
    assert abs(cfg.sigma0_deg - 1.83843) < 1e-5
    assert abs(data.ssv_pmax(cfg) - (1 - q ** cfg.n_sv) / (1 - q) / (2 * math.pi * cfg.sigma0_deg ** 2)) < 1e-12
    cen = np.linspace(-60, 60, 19)
    # Static: steady state = q exactly
    st = env_ssv_series(np.tile([[4, 11]], (300, 1)), cfg)
    assert abs(st[-1] - q) < 1e-12 and abs(q - 0.980296) < 1e-6
    # Coverage (row-major raster): exact against independent numpy over 2 periods, period mean vs README
    r = np.arange(722)
    acts = np.stack([r % 19, (r // 19) % 19], -1)
    e = env_ssv_series(acts, cfg)
    n = np_ssv_series([(cen[a], cen[b]) for a, b in acts], cfg)
    assert np.max(np.abs(e - n)) < 1e-12
    assert abs(e[361:722].mean() - 0.0002009) < 5e-8
    # Random: exact expectation (README) vs the environment's mean over 20000 i.i.d. looks (MC, 4 SE)
    M = 361
    grid = np.array([(cen[i], cen[j]) for j in range(19) for i in range(19)])
    d2 = ((grid[:, None, :] - grid[None, :, :]) ** 2).sum(-1)
    s0, g = cfg.sigma0_deg, cfg.gamma_sv
    ex = 0.0
    for k in range(1, cfg.n_sv + 1):
        sig = s0 * (1 + g) ** k
        ex += (np.exp(-d2 / (2 * sig ** 2)) / (2 * math.pi * sig ** 2)).sum() / M ** 2
    ex /= data.ssv_pmax(cfg)
    assert abs(ex - 0.0036314) < 5e-8
    acts = np.asarray(jax.random.randint(jax.random.PRNGKey(5), (20100, 2), 0, 19))
    e = env_ssv_series(acts, cfg)[100:]
    se = e.std() / math.sqrt(len(e)) * 3   # autocorrelation allowance
    assert abs(e.mean() - ex) < 4 * se, (e.mean(), ex, se)


# ----------------------------------------------------------------------------- B1: UKF == KF on linear h
def test_B1_ukf_equals_kf_linear():
    H = jnp.array([[1.0, 0.0]])
    cfg2 = replace(CFG, ukf_kappa=1.0)            # Stone Soup default kappa = 3 - n with n = 2
    P = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    x = jnp.array([0.3, -0.2])
    R = jnp.array([[1.0]])
    zh, S, C = data.ukf_meas_pred(x, P, R, cfg2, h=lambda X: X @ H.T, angle_idx=())
    K = C @ jnp.linalg.inv(S)
    Pp = P - K @ S @ K.T
    assert abs(float(S[0, 0]) - 3.0) < 1e-12
    np.testing.assert_allclose(np.asarray(K).ravel(), [2 / 3, 1 / 6], atol=1e-12)
    np.testing.assert_allclose(np.asarray(Pp), [[0.666666666667, 0.166666666667], [0.166666666667, 0.916666666667]],
                               atol=1e-11)
    rng = np.random.default_rng(1)
    for _ in range(50):                            # 6-D, the shipped kappa = -3 (n + lambda = 0.75 > 0)
        A = rng.normal(size=(6, 6))
        P6 = jnp.asarray(A @ A.T + 6 * np.eye(6))
        H6 = jnp.asarray(rng.normal(size=(3, 6)))
        x6 = jnp.asarray(rng.normal(size=6))
        R6 = jnp.eye(3) * 0.7
        zh, S, C = data.ukf_meas_pred(x6, P6, R6, CFG, h=lambda X: X @ H6.T, angle_idx=())
        np.testing.assert_allclose(np.asarray(zh), np.asarray(H6 @ x6), atol=1e-10)
        np.testing.assert_allclose(np.asarray(S), np.asarray(H6 @ P6 @ H6.T + R6), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.asarray(C), np.asarray(P6 @ H6.T), rtol=1e-10, atol=1e-10)


# ----------------------------------------------------------------------------- B2: association
def ss_greedy_nn(D, elig):
    """numpy transcription of stonesoup NearestNeighbour.associate with gate == missed_distance."""
    K, M = D.shape
    assign = -np.ones(K, int)
    used = set()
    todo = set(range(K))
    while todo:
        best = None
        for k in todo:
            for m in range(M):
                if elig[k, m] and m not in used and (best is None or D[k, m] < best[0]):
                    best = (D[k, m], k, m)
        if best is None:
            break
        _, k, m = best
        assign[k] = m
        used.add(m)
        todo.discard(k)
    return assign


def test_B2_association_vs_bruteforce_and_scipy():
    rng = np.random.default_rng(2)
    K, M, miss = CFG.k_max, CFG.n_tgt, CFG.missed
    n = 10_000
    D = rng.uniform(0, 4, (n, K, M))
    el = (rng.uniform(size=(n, K, M)) < 0.25) & (D < 3)
    gnn = np.asarray(jax.jit(jax.vmap(lambda d, e: data.associate_gnn(d, e, miss)))(jnp.asarray(D), jnp.asarray(el)))
    nnj = np.asarray(jax.jit(jax.vmap(data.associate_nn))(jnp.asarray(D), jnp.asarray(el)))
    big = 1e6
    for i in range(n):
        cost = np.where(el[i], D[i] - miss, big)       # assigning a detection replaces the missed cost
        aug = np.concatenate([cost, np.zeros((K, K))], 1)
        r, c = linear_sum_assignment(aug)
        ref = aug[r, c].sum()
        a = gnn[i]
        ours = sum(D[i, k, a[k]] - miss for k in range(K) if a[k] >= 0)
        assert all(el[i, k, a[k]] for k in range(K) if a[k] >= 0)
        assert abs(ours - ref) < 1e-9
        if i < 2000:
            np.testing.assert_array_equal(nnj[i], ss_greedy_nn(D[i], el[i]))


# ----------------------------------------------------------------------------- B3: printed reward on the 1-D toy
B3_TABLE = {1: (0.0, 0.0, 54.81), 2: (0.0, 0.8082, 91.05), 3: (0.6653, 1.3164, 122.77), 5: (1.5302, 2.0263, 179.22),
            10: (2.8712, 3.2113, 300.16), 20: (4.5591, 4.7890, 504.23), 100: (11.4156, 11.4965, 1737.76)}


def b3_rates(k, burn=3000, n=1000, dt=0.05, q=1.0, r=2500.0):
    F = np.array([[1, dt], [0, 1]])
    Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    H = np.array([[1.0, 0.0]])
    P = np.diag([50.0 ** 2, 100.0 ** 2])
    pn_prev, det_prev = np.linalg.norm(P), False
    tot_u, tot_t, norms = 0.0, 0.0, []
    for t in range(1, burn + n + 1):
        P = F @ P @ F.T + Q
        hit = (t % k == 0)
        if hit:
            S = H @ P @ H.T + r
            Kg = P @ H.T / S
            P = P - Kg @ S @ Kg.T
        pn = np.linalg.norm(P)
        tu = float(data.rtl_flip_terms(jnp.array(True), jnp.array(hit), jnp.array(det_prev), pn_prev, pn))
        tt = float(data.rtl_flip_terms(jnp.array(True), jnp.array(hit), jnp.array(False), pn_prev, pn))
        if t > burn:
            tot_u += tu
            tot_t += tt
            norms.append(pn)
        pn_prev, det_prev = pn, hit
    return tot_u / n, tot_t / n, float(np.mean(norms))


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10, 20, 100])
def test_B3_reward_revisit_table(k):
    # 1000 is not a multiple of 3: the README table's k = 3 row is the 1000-step window starting at burn-in 3002
    # (window-phase artifact, ~0.5 %); the exact per-period rate is checked separately below.
    u, t, m = b3_rates(k, burn=3002 if k == 3 else 3000)
    eu, et, em = B3_TABLE[k]
    assert abs(u - eu) < 1e-4 and abs(t - et) < 1e-4 and abs(m - em) < 1e-2
    if k <= 2:
        assert abs(u) < 1e-9          # continuous tracking earns exactly 0 under the printed reward
    if k == 3:
        assert abs(b3_rates(3, n=999)[0] - 0.6620525) < 1e-6   # exact steady-state rate G(3)/3


# ----------------------------------------------------------------------------- C1: masked BiGRU
def test_C1_bigru_padding_and_flax_parity():
    import flax.linen as nn
    rng = np.random.default_rng(3)
    K, D, H = 17, 7, 16
    net = BiGRU(H)
    x = jnp.asarray(rng.normal(size=(K, D)), jnp.float32)
    p = net.init(jax.random.PRNGKey(0), x, jnp.ones(K, bool))
    nv = 6
    mask = jnp.arange(K) < nv
    out1 = net.apply(p, x, mask)
    x2 = x.at[nv:].set(jnp.asarray(rng.normal(size=(K - nv, D)) * 5, jnp.float32))
    out2 = net.apply(p, x2, mask)
    np.testing.assert_allclose(np.asarray(out1), np.asarray(out2), atol=1e-6)
    out3 = net.apply(p, x[:nv], jnp.ones(nv, bool))       # same as the unpadded sequence
    np.testing.assert_allclose(np.asarray(out1), np.asarray(out3), atol=1e-6)
    # differential vs flax GRUCell with the same weights
    cell = nn.GRUCell(H)
    pp = p["params"]
    for d, seq in (("fwd", x[:nv]), ("bwd", x[:nv][::-1])):
        wi, bi, wh, bhn = (pp[f"{d}_{n}"] for n in ("wi", "bi", "wh", "bhn"))
        fp = {"params": {"ir": {"kernel": wi[:, :H], "bias": bi[:H]}, "iz": {"kernel": wi[:, H:2 * H], "bias": bi[H:2 * H]},
                         "in": {"kernel": wi[:, 2 * H:], "bias": bi[2 * H:]}, "hr": {"kernel": wh[:, :H]},
                         "hz": {"kernel": wh[:, H:2 * H]}, "hn": {"kernel": wh[:, 2 * H:], "bias": bhn}}}
        h = jnp.zeros(H, jnp.float32)
        for t in range(nv):
            h, _ = cell.apply(fp, h, seq[t])
        ref = h
        got = out1[:H] if d == "fwd" else out1[H:]
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-5)


# ----------------------------------------------------------------------------- C2: MHSA
def np_mha(e, mask, p, heads):
    """Independent numpy multi-head attention (Vaswani et al.) from flax MultiHeadDotProductAttention weights."""
    q = np.einsum("kd,dhf->khf", e, p["query"]["kernel"]) + p["query"]["bias"]
    k = np.einsum("kd,dhf->khf", e, p["key"]["kernel"]) + p["key"]["bias"]
    v = np.einsum("kd,dhf->khf", e, p["value"]["kernel"]) + p["value"]["bias"]
    dk = q.shape[-1]
    logits = np.einsum("qhf,khf->hqk", q, k) / math.sqrt(dk)
    logits = np.where(mask[None, None, :], logits, -1e30)
    w = np.exp(logits - logits.max(-1, keepdims=True))
    w = w / w.sum(-1, keepdims=True)
    o = np.einsum("hqk,khf->qhf", w, v)
    return np.einsum("qhf,hfd->qd", o, p["out"]["kernel"]) + p["out"]["bias"]


def test_C2_mhsa_equivariance_padding_and_reference():
    rng = np.random.default_rng(4)
    K, d = 17, 64
    m = MHSA(d, 4)
    e = jnp.asarray(rng.normal(size=(K, d)), jnp.float32)
    nv = 7
    mask = jnp.arange(K) < nv
    p = m.init(jax.random.PRNGKey(1), e, mask)
    out = np.asarray(m.apply(p, e, mask))
    perm = np.concatenate([rng.permutation(nv), np.arange(nv, K)])
    outp = np.asarray(m.apply(p, e[perm], mask))
    np.testing.assert_allclose(outp[:nv], out[perm[:nv]], atol=1e-5)
    e2 = e.at[nv:].set(jnp.asarray(rng.normal(size=(K - nv, d)) * 3, jnp.float32))
    np.testing.assert_allclose(np.asarray(m.apply(p, e2, mask))[:nv], out[:nv], atol=1e-5)
    pp = jax.tree.map(np.asarray, p["params"])
    a = np_mha(np.asarray(e, np.float64), np.asarray(mask), pp["mha"], 4)
    y = np.asarray(e, np.float64) + a
    mu, var = y.mean(-1, keepdims=True), y.var(-1, keepdims=True)
    ref = (y - mu) / np.sqrt(var + m.ln_eps) * pp["ln"]["scale"] + pp["ln"]["bias"]
    np.testing.assert_allclose(out[:nv], ref[:nv], atol=2e-4)


# ----------------------------------------------------------------------------- D: GOSPA worked example
def test_D_gospa_worked_example():
    X = jnp.array([[0.0, 0, 0, 0, 0, 0], [100.0, 0, 0, 0, 0, 0]])
    Y = jnp.zeros((3, 6)).at[0, 0].set(10.0)
    v = jnp.array([True, False, False])
    d, loc, miss, false = (float(a) for a in data.gospa(X, Y, v, 50.0, 2))
    assert abs(d - 36.742346) < 1e-6 and abs(loc - 100) < 1e-9 and abs(miss - 1250) < 1e-9 and false == 0.0


def test_P_GOSPA_static_ratio():
    """P-GOSPA: a step with no tracks has missed / distance = c sqrt(N/2) (= 1.2247e4 m at c = 10 km, N = 3)."""
    X = jnp.ones((3, 6))
    d, _, miss, _ = data.gospa(X, jnp.zeros((CFG.k_max, 6)), jnp.zeros(CFG.k_max, bool), CFG.gospa_c, 2)
    assert abs(float(miss) / float(d) - 1e4 * math.sqrt(1.5)) < 1e-6


# ----------------------------------------------------------------------------- E: PPO estimator
def test_E1_ppo_clip_gradient():
    def pg(theta):
        la = theta[None, :]
        le = jnp.zeros((1, 1))
        a = jnp.array([[0, 0]])
        old = jnp.log(jnp.array([0.5]))
        p, _, _, _ = alg.ppo_loss_terms(la, le, jnp.zeros(1), a, old, jnp.array([1.0]), jnp.zeros(1), 0.2,
                                        normalize_adv=False)
        return p
    g0 = np.asarray(jax.grad(pg)(jnp.array([0.0, 0.0])))
    np.testing.assert_allclose(g0, [-0.5, 0.5], atol=1e-12)        # pg is the negated objective
    g1 = np.asarray(jax.grad(pg)(jnp.array([1.0, 0.0])))
    np.testing.assert_allclose(g1, [0.0, 0.0], atol=1e-12)
    check_grads(pg, (jnp.array([0.1, -0.2]),), order=1, modes=["rev"])


def test_E2_gae_identities():
    rng = np.random.default_rng(5)
    T, gam, lam = 50, 0.99, 0.95
    r, v = rng.normal(size=T), rng.normal(size=T)
    d = (rng.uniform(size=T) < 0.1).astype(float)
    lv = rng.normal()
    adv, ret = (np.asarray(a) for a in alg.gae(jnp.asarray(r), jnp.asarray(v), jnp.asarray(d), jnp.asarray(lv), gam, lam))
    vn = np.append(v[1:], lv)
    delta = r + gam * vn * (1 - d) - v
    fwd = np.zeros(T)
    for t in range(T):
        acc, c = 0.0, 1.0
        for l in range(t, T):
            acc += c * delta[l]
            if d[l]:
                break
            c *= gam * lam
        fwd[t] = acc
    np.testing.assert_allclose(adv, fwd, atol=1e-12)
    np.testing.assert_allclose(ret, adv + v, atol=1e-12)


# ----------------------------------------------------------------------------- G: check_grads through every encoder
@pytest.mark.parametrize("enc", ["flat", "bigru", "mhsa"])
def test_G_ppo_loss_check_grads(enc):
    ecfg = replace(CFG, n_sv=8, k_max=5)
    net = ActorCritic(replace(NetConfig(), encoder=enc), ecfg)
    rng = np.random.default_rng(6)
    B = 4
    obs = dummy_obs(ecfg, (B,))
    obs = obs._replace(tl=jnp.asarray(rng.uniform(-1, 1, obs.tl.shape), jnp.float32),
                       tl_mask=jnp.asarray(np.arange(ecfg.k_max)[None] < np.array([[0], [2], [3], [5]])),
                       sv_az=jnp.asarray(rng.uniform(-60, 60, obs.sv_az.shape), jnp.float32),
                       sv_el=jnp.asarray(rng.uniform(-60, 60, obs.sv_el.shape), jnp.float32),
                       sv_age=jnp.asarray(rng.integers(1, 50, obs.sv_age.shape), jnp.float32),
                       sv_valid=jnp.ones(obs.sv_valid.shape, bool))
    p = net.init(jax.random.PRNGKey(0), obs)
    # Fresh init has zero biases and a mostly-zero raster, so most conv pre-activations sit exactly on the ReLU kink
    # (a non-differentiable point, where finite differences disagree with any subgradient). Jitter every parameter
    # so the check runs at a generic, differentiable point.
    leaves, tdef = jax.tree.flatten(p)
    jit_rng = np.random.default_rng(7)
    p64 = jax.tree.unflatten(tdef, [jnp.asarray(np.asarray(x, np.float64) + jit_rng.normal(0, 0.05, x.shape))
                                    for x in leaves])
    a = jnp.asarray(rng.integers(0, 19, (B, 2)), jnp.int32)
    old = jnp.asarray(rng.normal(-5.9, 0.1, B))
    adv, ret = jnp.asarray(rng.normal(size=B)), jnp.asarray(rng.normal(size=B))

    def loss(params):
        la, le, v = net.apply(params, obs)
        pg, vl, el, _ = alg.ppo_loss_terms(la, le, v, a, old, adv, ret, 0.2)
        return pg + 0.5 * vl + 0.01 * el
    assert jax.eval_shape(loss, p64).dtype == jnp.float64
    check_grads(loss, (p64,), order=1, modes=["rev"], eps=1e-6, atol=1e-5, rtol=1e-4)
