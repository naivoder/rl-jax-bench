"""toy-test: closed-form / exactly computable toys (README T1-T9 + §8 additions) and finite-difference gradient
checks. Run on CPU:  JAX_PLATFORMS=cpu uv run python -m pytest tests/test_grads.py -q

T7 (Lagrange multiplier) is N/A: constraint-mode regularisers are out of scope (UNREPRODUCIBLE(budget), README).
"""
import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
enable_x64 = lambda: jax.enable_x64(True)  # noqa: E731
from jax.test_util import check_grads  # noqa: E402
from scipy import stats as sst  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src import eval as EV  # noqa: E402
from src import model as M  # noqa: E402


def _cfg(**kw):
    base = dict(obs_norm=True, obs_clip=10.0, tanh=True, softplus=True, init_std=0.5, min_std=1e-3, ppo_eps=0.25,
                vclip_eps=0.0, gae_lambda=0.9, gamma=0.99, lr=3e-4, lr_horizon_steps=312640.0, lr_end=0.0,
                adam_b1=0.9, adam_b2=0.999, adam_eps=1e-7, grad_clip=0.5, ent_coef=0.0, value_norm=True,
                adv_norm=False, recompute=True, shared=False, baseline_cost=1.0, reward_norm=False,
                logp_presquash=False)
    base.update(kw)
    return A.LaneCfg(**{k: jnp.asarray(v) for k, v in base.items()})


# ------------------------------------------------------------------ T1 GAE = lambda-mixture on a fragment
def test_T1_gae_worked_example():
    with enable_x64():
        v = jnp.array([0.5, 1.0, 1.5, 2.0])[:, None]
        r = jnp.array([1.0, 0.0, 2.0])[:, None]
        tgt, adv = A.gae(v, r, jnp.zeros((3, 1), bool), 0.9, 0.8)
        assert abs(float(adv[0, 0]) - 2.84432) < 1e-12
        assert abs(float(adv[1, 0]) - 2.006) < 1e-12 and abs(float(adv[2, 0]) - 2.3) < 1e-12
        assert abs(float(tgt[0, 0]) - 3.34432) < 1e-12
        _, adv_t = A.gae(v, r, jnp.array([False, True, False])[:, None], 0.9, 0.8)
        assert abs(float(adv_t[1, 0]) + 1.0) < 1e-12 and abs(float(adv_t[0, 0]) - 0.68) < 1e-12


def _mixture_adv(v, r, d, g, lam):
    """Brute-force (1-lam) sum_N lam^(N-1) Vhat^(N) - V on a fragment, returns truncated at the end / episode end."""
    T = len(r)
    out = np.zeros(T)
    for t in range(T):
        def nstep(N):
            s, disc = 0.0, 1.0
            for i in range(t, t + N):
                if i >= T:
                    return s + disc * v[T]
                s += disc * r[i]
                if d[i]:
                    return s
                disc *= g
            return s + disc * v[t + N]
        tot, w = 0.0, 0.0
        for N in range(1, 400):
            tot += (1 - lam) * lam ** (N - 1) * nstep(N)
            w += (1 - lam) * lam ** (N - 1)
        tot += (1 - w) * nstep(400)
        out[t] = tot - v[t]
    return out


def test_T1_gae_random_fragments():
    rng = np.random.default_rng(0)
    with enable_x64():
        for _ in range(5):
            T = 8
            v, r = rng.normal(size=T + 1), rng.normal(size=T)
            d = rng.random(T) < 0.2
            _, adv = A.gae(jnp.asarray(v)[:, None], jnp.asarray(r)[:, None], jnp.asarray(d)[:, None], 0.97, 0.9)
            np.testing.assert_allclose(np.asarray(adv[:, 0]), _mixture_adv(v, r, d, 0.97, 0.9), atol=1e-10)


# ------------------------------------------------------------------ T2 PPO with the 1/(1+eps) lower bound
def _ppo_official(logp, logp_b, adv, eps):
    log_rho = jax.lax.stop_gradient(logp - logp_b)
    le = jnp.log1p(eps)
    clipped = ((adv > 0) & (log_rho > le)) | ((adv < 0) & (log_rho < -le))
    mask = 1.0 - clipped.astype(adv.dtype)
    return jnp.mean(-logp * jax.lax.stop_gradient(adv * mask * jnp.exp(log_rho * mask)))


def _ppo_min(logp, logp_b, adv, eps):
    rho = jnp.exp(logp - logp_b)
    return jnp.mean(-jnp.minimum(rho * adv, jnp.clip(rho, 1 / (1 + eps), 1 + eps) * adv))


@pytest.mark.parametrize("A_,rho,loss,g", [(1, 1.10, -1.10, -1.10), (1, 1.30, -1.20, 0.0), (-1, 0.85, 0.85, 0.85),
                                            (-1, 0.82, 1 / 1.2, 0.0), (-1, 0.78, 1 / 1.2, 0.0)])
def test_T2_ppo_table(A_, rho, loss, g):
    with enable_x64():
        lb = jnp.zeros(1)
        lp = jnp.log(jnp.array([rho]))
        adv = jnp.array([float(A_)])
        assert abs(float(_ppo_min(lp, lb, adv, 0.2)) - loss) < 1e-9
        g_off = jax.grad(lambda x: _ppo_official(x, lb, adv, 0.2))(lp)
        g_min = jax.grad(lambda x: _ppo_min(x, lb, adv, 0.2))(lp)
        assert abs(float(g_off[0]) - g) < 1e-9 and abs(float(g_min[0]) - g) < 1e-9


def test_T2_gaussian_example_and_check_grads():
    with enable_x64():
        a, s = jnp.array([0.5]), 1.0
        lp_b = jax.scipy.stats.norm.logpdf(a, 0.0, s)
        f = lambda m: _ppo_official(jax.scipy.stats.norm.logpdf(a, m, s), lp_b, jnp.array([1.0]), 0.2)
        assert abs(float(jax.grad(f)(0.1)) + 0.418411) < 1e-6
        a2 = jnp.array([1.0])
        f2 = lambda m: _ppo_official(jax.scipy.stats.norm.logpdf(a2, m, s), jax.scipy.stats.norm.logpdf(a2, 0.0, s),
                                     jnp.array([1.0]), 0.2)
        assert float(jax.grad(f2)(0.5)) == 0.0
        fm = lambda m: _ppo_min(jax.scipy.stats.norm.logpdf(a, m, s), lp_b, jnp.array([1.0]), 0.2)
        check_grads(fm, (0.1,), order=1, modes=["rev"], eps=1e-6)


def test_T2_official_form_gradient_equals_min_form_random():
    with enable_x64():
        k = jax.random.PRNGKey(0)
        lp = jax.random.normal(k, (500,)) * 0.3
        lpb = jax.random.normal(jax.random.fold_in(k, 1), (500,)) * 0.3
        adv = jax.random.normal(jax.random.fold_in(k, 2), (500,))
        g1 = jax.grad(lambda x: _ppo_official(x, lpb, adv, 0.25))(lp)
        g2 = jax.grad(lambda x: _ppo_min(x, lpb, adv, 0.25))(lp)
        np.testing.assert_allclose(np.asarray(g1), np.asarray(g2), atol=1e-12)


def test_T2_full_loss_check_grads():
    """Finite-difference check of the complete per-minibatch loss (tanh + softplus + popart targets + entropy)."""
    with enable_x64():
        k = jax.random.PRNGKey(3)
        p = M.init_params(k, 5, 2, pi_width=8, v_width=8, pi_last_gain=0.5)
        p = jax.tree.map(lambda x: x.astype(jnp.float64), p)
        B = 16
        obs = jax.random.normal(jax.random.fold_in(k, 1), (B, 5))
        act = jnp.tanh(jax.random.normal(jax.random.fold_in(k, 2), (B, 2)))
        mb = A.MB(obs, act, jnp.zeros((B, 2)), jnp.full((B, 2), 0.5), jnp.zeros(B),
                  jax.random.normal(jax.random.fold_in(k, 4), (B,)), jax.random.normal(jax.random.fold_in(k, 5), (B,)))
        cfg = _cfg(ppo_eps=10.0, ent_coef=0.01, vclip_eps=0.0)       # large eps: away from clip kinks
        cfg = cfg._replace(**{f: jnp.asarray(getattr(cfg, f), jnp.float64) for f in cfg._fields
                               if jnp.asarray(getattr(cfg, f)).dtype != bool})
        st = A.Static()
        f = lambda pp: A.loss_fn(pp, mb, cfg, jax.random.PRNGKey(9), st, pg_form="min")[0]
        check_grads(f, (p,), order=1, modes=["rev"], eps=1e-6, atol=1e-5, rtol=1e-5)
        g_off = jax.grad(lambda pp: A.loss_fn(pp, mb, cfg, jax.random.PRNGKey(9), st)[0])(p)
        g_min = jax.grad(f)(p)
        for x, y in zip(jax.tree.leaves(g_off), jax.tree.leaves(g_min)):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-12)
        # and with the clip active (eps 0.05, behaviour far from current): gradients still agree
        cfg2 = cfg._replace(ppo_eps=jnp.asarray(0.05, jnp.float64))
        mb2 = mb._replace(mu_b=jnp.full((B, 2), 0.3))
        g1 = jax.grad(lambda pp: A.loss_fn(pp, mb2, cfg2, jax.random.PRNGKey(9), st)[0])(p)
        g2 = jax.grad(lambda pp: A.loss_fn(pp, mb2, cfg2, jax.random.PRNGKey(9), st, pg_form="min")[0])(p)
        for x, y in zip(jax.tree.leaves(g1), jax.tree.leaves(g2)):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-12)


# ------------------------------------------------------------------ T3 distribution algebra
def test_T3_logprob_entropy_kl():
    with enable_x64():
        mu, sd = jnp.array([0.3, -0.2]), jnp.array([0.8, 1.3])
        a = jnp.array([0.5, -0.7])
        lp = M.log_prob(a, mu, sd, jnp.asarray(False), False)
        assert abs(float(lp) - float(sst.norm.logpdf(np.asarray(a), np.asarray(mu), np.asarray(sd)).sum())) < 1e-12
        #: clip transform -> Gaussian density AT the clipped action
        assert abs(float(M.log_prob(jnp.array([1.0]), jnp.zeros(1), jnp.ones(1), jnp.asarray(False), False)) + 1.418939) < 1e-6
        # tanh log-det vs numerical Jacobian
        x = jnp.linspace(-3, 3, 13)
        num = jnp.log(jax.vmap(jax.grad(jnp.tanh))(x))
        np.testing.assert_allclose(np.asarray(M.tanh_fldj(x)), np.asarray(num), atol=1e-10)
        # tanh inner branch: log p(u) = log N(atanh u) - log(1-u^2)
        u = jnp.array([0.4, -0.9])
        ref = sst.norm.logpdf(np.arctanh(np.asarray(u)), np.asarray(mu), np.asarray(sd)) - np.log(1 - np.asarray(u) ** 2)
        assert abs(float(M.log_prob(u, mu, sd, jnp.asarray(True), False)) - ref.sum()) < 1e-10
        # tail branch at |u| = 0.9995 
        lt = M.log_prob(jnp.array([0.9995]), jnp.array([0.2]), jnp.array([0.7]), jnp.asarray(True), False)
        ref_t = sst.norm.logsf(math.atanh(0.999), 0.2, 0.7) - math.log(1e-3)
        assert abs(float(lt) - ref_t) < 1e-10
        lt2 = M.log_prob(jnp.array([-1.0]), jnp.array([0.2]), jnp.array([0.7]), jnp.asarray(True), False)
        assert abs(float(lt2) - (sst.norm.logcdf(-math.atanh(0.999), 0.2, 0.7) - math.log(1e-3))) < 1e-10
        # Gaussian entropy closed form (clip lanes)
        h = M.entropy(jax.random.PRNGKey(0), jnp.zeros(1), jnp.array([0.5]), jnp.asarray(False))
        assert abs(float(h) - 0.7257913526) < 1e-8
        # tanh entropy: E[one-sample estimate] = H(X) + E log(1 - tanh^2 x); -0.2257 at sigma 0.5 (README)
        hs = jax.vmap(lambda k: M.entropy(k, jnp.zeros(1), jnp.array([0.5]), jnp.asarray(True)))(
            jax.random.split(jax.random.PRNGKey(1), 200000))
        assert abs(float(hs.mean()) - (0.7257913526 - 0.2257)) < 3e-3
        # decoupled KL identity and KL vs Monte-Carlo
        m1, s1, m2, s2 = jnp.zeros(1), jnp.ones(1), jnp.array([0.3]), jnp.array([0.8])
        kl = M.kl_gauss(m1, s1, m2, s2)
        assert abs(float(kl) - 0.1284189) < 1e-7
        assert abs(float(M.kl_gauss(m1, s1, m1, s2) + M.kl_gauss(m1, s2, m2, s2)) - float(kl)) < 1e-12
        xs = jax.random.normal(jax.random.PRNGKey(2), (10 ** 6, 1))
        mc = jnp.mean(sst.norm.logpdf(np.asarray(xs), 0, 1) - sst.norm.logpdf(np.asarray(xs), 0.3, 0.8))
        assert abs(float(mc) - float(kl)) < 4 * 0.0006


def test_T3_std_offsets_and_safe_exp():
    with enable_x64():
        for sp, i0, shift_ref in [(True, 0.5, -0.4352956), (False, 1.0, -0.0010005), (True, 1.0, 0.5397424)]:
            sh = M.std_shift(jnp.asarray(i0), jnp.asarray(1e-3), jnp.asarray(sp))
            assert abs(float(sh) - shift_ref) < 1e-6
            assert abs(float(M.std_of(jnp.zeros(()), sh, 1e-3, jnp.asarray(sp))) - i0) < 1e-12
        g = jax.grad(M.safe_exp)(20.0)
        assert abs(float(g) - math.exp(15.0)) < 1e-6 * math.exp(15.0)
        check_grads(M.safe_exp, (0.3,), order=1, modes=["fwd", "rev"])


# ------------------------------------------------------------------ T4 initial saturation fractions
def test_T4_saturation_fractions():
    k = jax.random.PRNGKey(0)
    p = M.init_params(k, 17, 6, pi_last_gain=0.01)
    obs = jnp.clip(jax.random.normal(jax.random.fold_in(k, 1), (20000, 17)), -10, 10)
    mu, x_rho, _ = M.forward(p, obs, jnp.asarray(False))
    for i0, ref in [(1.0, 0.3173), (0.5, 0.0455)]:
        sd = M.std_of(x_rho, M.std_shift(jnp.asarray(i0), jnp.asarray(1e-3), jnp.asarray(False)), 1e-3, False)
        a = M.sample(jax.random.fold_in(k, 2), mu, sd, jnp.asarray(False))
        frac = float(jnp.mean(jnp.abs(a) >= 1.0))
        se = math.sqrt(ref * (1 - ref) / a.size)
        assert abs(frac - ref) < 4 * se + 2e-3, (i0, frac)


# ------------------------------------------------------------------ T5 normaliser merge (official AverageMeanStd)
def test_T5_normaliser_merge():
    with enable_x64():
        ms = D.meanstd_init(1, jnp.float64)
        m0, s0 = D.meanstd_get(ms)
        assert float(m0[0]) == 0.0 and float(s0[0]) == 1.0                    # mean 0 / std 1 before data
        ms = D.meanstd_update(ms, jnp.array([[1.0], [2.0], [3.0]]))
        ms = D.meanstd_update(ms, jnp.array([[4.0], [5.0]]))
        m, s = D.meanstd_get(ms)
        assert abs(float(m[0]) - 3.0) < 1e-12 and abs(float(s[0]) ** 2 - 2.0) < 1e-12
        rng = np.random.default_rng(0)
        x = rng.normal(3.0, 2.0, (1000, 4))
        ms = D.meanstd_init(4, jnp.float64)
        for chunk in np.array_split(x, [7, 100, 101, 640]):
            ms = D.meanstd_update(ms, jnp.asarray(chunk))
        m, s = D.meanstd_get(ms)
        np.testing.assert_allclose(np.asarray(m), x.mean(0), atol=1e-10)
        np.testing.assert_allclose(np.asarray(s), x.std(0), atol=1e-10)              # population std


# ------------------------------------------------------------------ T6 order-statistic CI coverage
@pytest.mark.parametrize("n", [20, 64, 1000])
@pytest.mark.parametrize("conv", ["official", "lh"])
def test_T6_ci_coverage(n, conv):
    exact = EV.coverage_exact(n, conv)
    if conv == "official":
        assert abs(exact - {20: 0.566, 64: 0.922, 1000: 0.950}[n]) < 1.5e-3
    else:
        assert abs(exact - {20: 0.626, 64: 0.948, 1000: 0.949}[n]) < 1.5e-3
    rng = np.random.default_rng(n)
    R = 100000 if n < 1000 else 20000
    X = np.sort(rng.random((R, n)), axis=1)
    l, h = EV.ci_ranks(n, conv)
    cov = np.mean((X[:, l - 1] <= 0.95) & (X[:, h - 1] >= 0.95))
    se = math.sqrt(exact * (1 - exact) / R)
    assert abs(cov - exact) < 3 * se + 1e-9, (cov, exact)
    # the CI helper returns exactly these order statistics
    lo, hi = EV.percentile_ci(X[0], conv)
    assert lo == X[0, l - 1] and hi == X[0, h - 1]


def test_T6_best_of_20():
    # E[F(max of K)] = K/(K+1): the 95th percentile ~ best of ~20 random configurations (README)
    assert abs(20 / 21 - 0.952) < 1e-3


# ------------------------------------------------------------------ T8 last-layer bound
@pytest.mark.parametrize("g", [0.01, 1.0])
def test_T8_last_layer_bound(g):
    k = jax.random.PRNGKey(4)
    p = M.init_params(k, 17, 6, pi_last_gain=g)
    obs = 10 * jax.random.normal(jax.random.fold_in(k, 1), (10000, 17))
    mu, _, _ = M.forward(p, obs, jnp.asarray(False))
    nrm = float(jnp.max(jnp.linalg.norm(mu, axis=-1)))
    assert nrm <= g * math.sqrt(64) * (1 + 1e-5)
    if g == 0.01:
        assert nrm <= 0.08 * (1 + 1e-5)


# ------------------------------------------------------------------ T9 scores
def test_T9_scores():
    curve = [5.0] + list(range(1, 11))
    assert EV.seed_score(curve) == 5.5
    assert EV.config_score([3.0, 1.0, 2.0]) == 2.0
    assert EV.p95(np.arange(21.0)) == 19.0                  # linear: position 0.95*20 = 19
    assert EV.p95(np.arange(11.0)) == pytest.approx(9.5)


# ------------------------------------------------------------------ implementation-level exactness
def test_value_padding_is_exact():
    """A width-64 value net zero-padded to 256 has identical output and identical (zero-padded) gradients."""
    k = jax.random.PRNGKey(5)
    p64 = M.init_params(k, 11, 3, v_width=64, v_width_max=64)
    p256 = M.init_params(k, 11, 3, v_width=64, v_width_max=256)
    obs = jax.random.normal(jax.random.fold_in(k, 1), (32, 11))
    f = lambda p: jnp.sum(M.forward(p, obs, jnp.asarray(False))[2] ** 2)
    np.testing.assert_allclose(float(f(p64)), float(f(p256)), rtol=1e-6)
    g64, g256 = jax.grad(f)(p64), jax.grad(f)(p256)
    w = g256["v"]["h1"]["kernel"]
    assert float(jnp.abs(w[64:]).max()) == 0.0 and float(jnp.abs(w[:, 64:]).max()) == 0.0
    np.testing.assert_allclose(np.asarray(w[:64, :64]), np.asarray(g64["v"]["h1"]["kernel"]), rtol=1e-5, atol=1e-7)
    assert float(jnp.abs(g256["v"]["out"]["kernel"][64:]).max()) == 0.0
    # policy trunk identical regardless of the value width (pairing)
    np.testing.assert_array_equal(np.asarray(p64["pi"]["h0"]["kernel"]), np.asarray(p256["pi"]["h0"]["kernel"]))


def test_adam_keras_eps_hat_and_lr_decay():
    cfg = _cfg(lr=1e-2, lr_horizon_steps=4.0)
    p = {"w": jnp.array([1.0, -2.0])}
    opt = A.adam_init(p)
    m = v = np.zeros(2)
    w = np.array([1.0, -2.0])
    for t in range(1, 6):
        g = np.array([0.3, -0.1]) * t
        p, opt = A.adam_step(p, {"w": jnp.asarray(g, jnp.float32)}, opt, cfg)
        lr = 1e-2 * (1 - min(t - 1, 4) / 4)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w = w - lr * math.sqrt(1 - 0.999 ** t) / (1 - 0.9 ** t) * m / (np.sqrt(v) + 1e-7)
        np.testing.assert_allclose(np.asarray(p["w"]), w, rtol=1e-5)


def test_grad_clip_global():
    g = {"a": jnp.array([3.0]), "b": jnp.array([4.0])}
    c, n = A.clip_grads(g, jnp.asarray(0.5))
    assert abs(float(n) - 5.0) < 1e-6 and abs(float(A.global_norm(c)) - 0.5) < 1e-6
    c, _ = A.clip_grads(g, jnp.asarray(0.0))                 # <= 0 = off
    assert float(c["a"][0]) == 3.0


def test_popart_semantics():
    """C66: targets normalised with PRE-update stats, advantages divided by the std, stats then updated."""
    with enable_x64():
        v = jnp.array([[1.0], [2.0], [3.0]])
        r = jnp.array([[1.0], [1.0]])
        ms = D.meanstd_update(D.meanstd_init(1, jnp.float64), jnp.array([[10.0], [20.0], [30.0], [40.0]]))
        m0, s0 = D.meanstd_get(ms)
        v_un = v * s0[0] + m0[0]
        tgt, adv = A.gae(v_un, r, jnp.zeros((2, 1), bool), 0.9, 0.8)
        n_t = (tgt - m0[0]) / s0[0]
        n_a = adv / s0[0]
        assert np.all(np.isfinite(np.asarray(n_t))) and float(s0[0]) == pytest.approx(math.sqrt(125.0))
        ms2 = D.meanstd_update(ms, tgt[..., None])
        m2, _ = D.meanstd_get(ms2)
        all_ = np.concatenate([[10, 20, 30, 40], np.asarray(tgt[:, 0])])
        assert float(m2[0]) == pytest.approx(all_.mean())
        assert float(n_a[0, 0]) == pytest.approx(float(adv[0, 0]) / math.sqrt(125.0))
