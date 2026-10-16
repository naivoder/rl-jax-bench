"""Differential test against the OFFICIAL seed_rl modules. Fixtures are produced in the isolated TF
venv by $OFFICIAL_ROOT/scripts/official/difftest_official.py (seed_rl@0e1e0ac):

  cd $OFFICIAL_ROOT && CUDA_VISIBLE_DEVICES=.venv/bin/python scripts/official/difftest_official.py \\
      tests/fixtures/official_fixtures.npz

Compared on identical weights and unrolls (3 cases: REC-like tanh/softplus/PopArt, BASE-like clip/safe_exp/value
clipping/adv-norm/entropy penalty, shared trunk without obs/value normalisation): obs statistics, normalised targets
and advantages, PopArt statistics before/after, total loss, every gradient, the global norm, the clipped gradients and
two keras-Adam steps under PolynomialDecay. Tolerance: float32 (rtol 1e-4 / atol 1e-5 scaled to the magnitude).

STUDY-ERA fixtures (official-code check, README): scripts/official/difftest_studyera.py runs the same HEAD modules with the action
distribution of seed_rl@68c15aa (verbatim, scripts/official/pd_68c15aa.py; stored action = pre-squash x), the configuration the
study's rows were produced with:

  cd $OFFICIAL_ROOT && CUDA_VISIBLE_DEVICES=.venv/bin/python scripts/official/difftest_studyera.py \\
      tests/fixtures/studyera_fixtures.npz

4 full training-step cases (REC-like with forced tanh saturation |x| up to 5; BASE-like with forced clip saturation;
two 16x32 'band' cases whose log-ratios fall inside the PPO clip band, so epsilon is pinned) + a log-prob/gradient
grid on raw x incl. |x| = 12 (4 transform x std-fn combinations). Mutation checks: HEAD semantics, epsilon 0.25 -> 0.3
and lambda must each make a study-era case fail.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src import model as M  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                   "official_fixtures.npz")
CASES = {"rec": dict(tanh=True, softplus=True, init_std=0.5, value_norm=True, vclip=0.0, adv_norm=False, ent=0.0,
                     shared=False, obs_norm=True, eps=0.25, lam=0.9),
         "base": dict(tanh=False, softplus=False, init_std=1.0, value_norm=True, vclip=0.2, adv_norm=True, ent=1e-2,
                      shared=False, obs_norm=True, eps=0.2, lam=0.95),
         "shared": dict(tanh=True, softplus=True, init_std=0.5, value_norm=False, vclip=0.0, adv_norm=False, ent=0.0,
                        shared=True, obs_norm=False, eps=0.25, lam=0.9)}
SEP_PATHS = [("std",), ("pi", "h0", "kernel"), ("pi", "h0", "bias"), ("pi", "h1", "kernel"), ("pi", "h1", "bias"), ("pi", "out", "kernel"),
             ("pi", "out", "bias"), ("v", "out", "kernel"), ("v", "out", "bias"), ("v", "h0", "kernel"), ("v", "h0", "bias"),
             ("v", "h1", "kernel"), ("v", "h1", "bias")]
SH_PATHS = [("std",), ("pi", "out", "kernel"), ("pi", "out", "bias"), ("pi", "h1", "kernel"), ("pi", "h1", "bias"), ("pi", "h0", "kernel"),
            ("pi", "h0", "bias"), ("vs", "kernel"), ("vs", "bias")]

FIX_SE = os.path.join(os.path.dirname(FIX), "studyera_fixtures.npz")
CASES_SE = {"rec_pre": dict(CASES["rec"]), "base_pre": dict(CASES["base"]),
            "band_pre": dict(CASES["rec"], adv_norm=True),
            "bandclip_pre": dict(CASES["base"], value_norm=False, adv_norm=False, ent=0.0)}

pytestmark = pytest.mark.skipif(not os.path.exists(FIX), reason="official fixtures not generated")


def _get(tree, path):
    for p in path:
        tree = tree[p]
    return tree


def _set(tree, path, val):
    if len(path) == 1:
        tree[path[0]] = val
    else:
        _set(tree[path[0]], path[1:], val)


def _close(ours, theirs, name, rtol=1e-4):
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    scale = max(1.0, float(np.abs(theirs).max()) if theirs.size else 1.0)
    np.testing.assert_allclose(ours, theirs, rtol=rtol, atol=1e-5 * scale, err_msg=name)


def _parity(fix, case, c, presquash, eps=None, lam=None, close=None, head_store=False):
    """Run our training-step pieces on the fixture's weights/unroll and compare (close=_close) or collect diffs."""
    close = close or _close
    z = np.load(fix)
    g = lambda k: z[f"{case}/{k}"]
    eps = c["eps"] if eps is None else eps
    lam = c["lam"] if lam is None else lam
    obs = jnp.asarray(g("obs"))
    T1, B, d = obs.shape
    T = T1 - 1
    Adim = g("act").shape[-1]
    paths = SH_PATHS if c["shared"] else SEP_PATHS
    wv = g("var12").shape[0] if not c["shared"] else 8
    params = M.init_params(jax.random.PRNGKey(0), d, Adim, pi_width=8, v_width=wv)
    for i, p in enumerate(paths):
        _set(params, p, jnp.asarray(g(f"var{i}")))
    cfg = A.LaneCfg(obs_norm=jnp.asarray(c["obs_norm"]), obs_clip=jnp.float32(10.0), tanh=jnp.asarray(c["tanh"]),
                    softplus=jnp.asarray(c["softplus"]), init_std=jnp.float32(c["init_std"]), min_std=jnp.float32(1e-3),
                    ppo_eps=jnp.float32(eps), vclip_eps=jnp.float32(c["vclip"]), gae_lambda=jnp.float32(lam),
                    gamma=jnp.float32(0.99), lr=jnp.float32(1e-2), lr_horizon_steps=jnp.float32(4.0),
                    lr_end=jnp.float32(0.0), adam_b1=jnp.float32(0.9), adam_b2=jnp.float32(0.999),
                    adam_eps=jnp.float32(1e-7), grad_clip=jnp.float32(0.5), ent_coef=jnp.float32(c["ent"]),
                    value_norm=jnp.asarray(c["value_norm"]), adv_norm=jnp.asarray(c["adv_norm"]),
                    recompute=jnp.asarray(True), shared=jnp.asarray(c["shared"]), baseline_cost=jnp.float32(1.0),
                    reward_norm=jnp.asarray(False), logp_presquash=jnp.asarray(presquash))
    st = A.Static(num_envs=B, unroll=T, epochs=1, minibatch=T * B)
    # obs statistics: prior update, then the training-step update with observation[:-1]
    obs_ms = D.meanstd_init(d)
    if c["obs_norm"]:
        obs_ms = D.meanstd_update(D.meanstd_update(obs_ms, jnp.asarray(g("prior_obs"))), obs[:-1])
        m, s = D.meanstd_get(obs_ms)
        close(m, g("obs_mean"), "obs mean")
        close(s, g("obs_std"), "obs std")
    val_ms = D.meanstd_init(1)
    if c["value_norm"]:
        val_ms = D.meanstd_update(val_ms, jnp.asarray(g("prior_targets"))[:, None])
    lb = jnp.asarray(g("logits_b"))
    shift = M.std_shift(cfg.init_std, cfg.min_std, cfg.softplus)
    act = jnp.asarray(g("act"))[:-1]
    if head_store:          # what the released HEAD would have stored for the same sample: tanh(x) / clip(x)
        act = jnp.tanh(act) if c["tanh"] else jnp.clip(act, -1.0, 1.0)
    batch = A.Batch(obs=obs, act=act, mu_b=lb[:-1, :, :Adim],
                    std_b=M.std_of(lb[:-1, :, Adim:], shift, cfg.min_std, cfg.softplus),
                    v_b=jnp.asarray(g("base_b"))[:-1], rew=jnp.asarray(g("rew"))[1:], done=jnp.asarray(g("done"))[1:])
    m0, s0 = D.meanstd_get(val_ms)
    if c["value_norm"]:
        close(m0[0], g("pre_mean"), "popart pre mean")
        close(s0[0], g("pre_std"), "popart pre std")
    tgt, adv, val_ms2 = A.compute_advantages(params, val_ms, batch, obs_ms, cfg, st)
    close(tgt, g("n_targets"), "normalised targets")
    close(adv, g("n_adv"), "normalised advantages")
    if c["value_norm"]:
        m2, s2 = D.meanstd_get(val_ms2)
        close(m2[0], g("post_mean"), "popart post mean")
        close(s2[0], g("post_std"), "popart post std")
    flat = lambda x: x.reshape((T * B,) + x.shape[2:])
    obs_n = flat(D.normalize_obs(obs_ms, obs[:-1], cfg.obs_norm, cfg.obs_clip))
    mb = A.MB(obs_n, flat(batch.act), flat(batch.mu_b), flat(batch.std_b), flat(batch.v_b), flat(tgt), flat(adv))
    (loss, aux), grads = jax.value_and_grad(A.loss_fn, has_aux=True)(params, mb, cfg, jax.random.PRNGKey(0), st)
    close(loss, g("loss"), "total loss")
    for i, p in enumerate(paths):
        close(_get(grads, p), g(f"grad{i}"), f"grad {p}")
    cg, gn = A.clip_grads(grads, cfg.grad_clip)
    close(gn, g("gnorm"), "global norm")
    for i, p in enumerate(paths):
        close(_get(cg, p), g(f"cgrad{i}"), f"clipped grad {p}")
    opt = A.adam_init(params)
    p1, opt = A.adam_step(params, cg, opt, cfg)
    p2, opt = A.adam_step(p1, jax.tree.map(lambda x: 0.5 * x, cg), opt, cfg)
    for i, p in enumerate(paths):
        close(_get(p2, p), g(f"w2_{i}"), f"adam 2-step {p}")
    return mb, cfg, params, st


@pytest.mark.parametrize("case", list(CASES))
def test_official_parity(case):
    """Released HEAD (0e1e0ac) semantics: diagnostic arms X3/X4."""
    _parity(FIX, case, CASES[case], presquash=False)


@pytest.mark.skipif(not os.path.exists(FIX_SE), reason="study-era fixtures not generated")
@pytest.mark.parametrize("case", list(CASES_SE))
def test_studyera_parity(case):
    """Study-era (68c15aa) semantics: the default of every paper arm."""
    _parity(FIX_SE, case, CASES_SE[case], presquash=True)


def _fails(**kw):
    try:
        _parity(**kw)
    except AssertionError:
        return True
    return False


@pytest.mark.skipif(not os.path.exists(FIX_SE), reason="study-era fixtures not generated")
@pytest.mark.parametrize("case", ["rec_pre", "base_pre"])
def test_studyera_fixture_rejects_head_semantics(case):
    """Mutation: the released-HEAD storage + log-prob of the same samples (saturated entries) must NOT match."""
    assert _fails(fix=FIX_SE, case=case, c=CASES_SE[case], presquash=False, head_store=True)


@pytest.mark.skipif(not os.path.exists(FIX_SE), reason="study-era fixtures not generated")
@pytest.mark.parametrize("case", ["band_pre", "bandclip_pre"])
def test_studyera_epsilon_and_lambda_pinned(case):
    """The band cases contain log-ratios inside (log 1.25, log 1.3) / (log 1.2, log 1.25): perturbing epsilon by
    +0.05 must fail (toy-test gap: the original fixtures could not see epsilon); so must lambda +0.05."""
    c = CASES_SE[case]
    mb, cfg, params, st = _parity(FIX_SE, case, c, presquash=True)
    mu, x_rho, _ = M.forward(params, mb.obs_n, c["shared"])
    std = M.std_of(x_rho, M.std_shift(cfg.init_std, cfg.min_std, cfg.softplus), cfg.min_std, cfg.softplus)
    lr = M.log_prob(mb.act, mu, std, cfg.tanh, True) - M.log_prob(mb.act, mb.mu_b, mb.std_b, cfg.tanh, True)
    lo, hi = np.log1p(c["eps"]), np.log1p(c["eps"] + 0.05)
    n_band = int(np.sum((np.abs(np.asarray(lr)) > lo) & (np.abs(np.asarray(lr)) < hi)))
    assert n_band >= 5, n_band
    assert _fails(fix=FIX_SE, case=case, c=c, presquash=True, eps=c["eps"] + 0.05)
    assert _fails(fix=FIX_SE, case=case, c=c, presquash=True, lam=c["lam"] + 0.05)


@pytest.mark.skipif(not os.path.exists(FIX_SE), reason="study-era fixtures not generated")
@pytest.mark.parametrize("name,tanh,softplus,init_std", [("tanh_sp05", True, True, 0.5), ("clip_se1", False, False, 1.0),
                                                         ("tanh_se1", True, False, 1.0), ("clip_sp05", False, True, 0.5)])
def test_studyera_logprob_grid(name, tanh, softplus, init_std):
    """68c15aa NormalSquashedDistribution vs ours on raw x incl. |x| = 12: std, postprocess, log-det, log-prob and
    d log-prob / d(mu, rho) (float32)."""
    z = np.load(FIX_SE)
    g = lambda k: jnp.asarray(z[f"lp/{name}/{k}"])
    shift = M.std_shift(jnp.float32(init_std), jnp.float32(1e-3), jnp.asarray(softplus))
    std = M.std_of(g("rho"), shift, jnp.float32(1e-3), jnp.asarray(softplus))
    _close(std, g("std"), "std", rtol=1e-5)
    post = jnp.tanh(g("x")) if tanh else jnp.clip(g("x"), -1.0, 1.0)
    _close(post, g("post"), "postprocess", rtol=1e-6)
    _close(M.tanh_fldj(g("x")) if tanh else jnp.zeros_like(g("x")), g("fldj"), "log-det", rtol=1e-5)

    def lp(mu, rho):
        sd = M.std_of(rho, shift, jnp.float32(1e-3), jnp.asarray(softplus))
        return M.log_prob(g("x"), mu, sd, jnp.asarray(tanh), True)
    _close(lp(g("mu"), g("rho")), g("logp"), "log-prob", rtol=1e-5)
    gm, gr = jax.grad(lambda m, r: jnp.sum(lp(m, r)), argnums=(0, 1))(g("mu"), g("rho"))
    _close(jnp.concatenate([gm, gr], -1), g("grad"), "d logp / d(mu, rho)", rtol=1e-4)


# ------------------------------------------------------------------ metric oracle (the notebook's own statistic code)
FIX_MET = os.path.join(os.path.dirname(FIX), "metric_fixtures.npz")


@pytest.mark.skipif(not os.path.exists(FIX_MET), reason="metric fixtures not generated")
def test_metric_parity_with_notebook():
    """src/eval.py vs the notebook's percentile_ci / plot_top_90 (exec'd verbatim by
    $OFFICIAL_ROOT/scripts/official/metric_oracle.py) on the authors' study rows: 5 choice groups x 3
    envs x every value. Conditional 95th percentile and its CI ends exact (float64); top-5% value shares exact."""
    from src import eval as E
    z = np.load(FIX_MET)
    n_ci = n_top = 0
    for k in z.files:
        if k.startswith("x/"):
            key = k[2:]
            x = z[k]
            p, lo, hi, flag = z[f"ci/{key}/0.95"]
            assert E.p95(x) == pytest.approx(p, rel=0, abs=1e-9), key
            ours_lo, ours_hi = E.percentile_ci(x, "official")
            assert (ours_lo, ours_hi) == (lo, hi), key
            assert bool(flag) == (int(binom_hi(len(x))) > len(x) - 1), key
            n_ci += 1
        elif k.endswith("/values") and k.startswith("top/"):
            g = k[: -len("/values")]
            share = E.top_share(z[g + "/all_values"], z[g + "/all_scores"])
            cnt = dict(zip(z[g + "/values"].tolist(), z[g + "/counts"].tolist()))
            m = sum(cnt.values())
            for v, s in share.items():
                assert s == pytest.approx(cnt.get(v, 0) / m, abs=1e-12), (g, v)
            n_top += 1
    assert n_ci >= 30 and n_top == 15, (n_ci, n_top)


def binom_hi(n):
    from scipy.stats import binom
    return binom.interval(0.95, n, 0.95)[1]


# ------------------------------------------------------------------ multi-epoch loop 
FIX_ME = os.path.join(os.path.dirname(FIX), "multiepoch_fixtures.npz")
CASES_ME = {"me_rec": dict(CASES["rec"]),
            "me_clip": dict(CASES["base"], ent=0.0, adv_norm=True)}


@pytest.mark.skipif(not os.path.exists(FIX_ME), reason="multi-epoch fixtures not generated")
@pytest.mark.parametrize("case", list(CASES_ME))
def test_official_multiepoch_loop(case):
    """The OFFICIAL ppo_training_step (split_with_advantage_recomputation, 3 epochs x 4 minibatches, the shuffle
    replaced by saved permutations; scripts/official/difftest_multiepoch.py) vs our ``algorithm.update`` replaying the same
    permutations: final weights after 12 Adam steps, PopArt stats after 3 recomputes, obs stats. Pins the per-epoch
    recompute with current params, the minibatch gather, per-minibatch adv-norm and the obs-stat ordering."""
    c = CASES_ME[case]
    z = np.load(FIX_ME)
    g = lambda k: z[f"{case}/{k}"]
    obs = jnp.asarray(g("obs"))
    T1, B, d = obs.shape
    T = T1 - 1
    Adim = g("act").shape[-1]
    perms = jnp.asarray(g("perms"))
    E, n = perms.shape
    params = M.init_params(jax.random.PRNGKey(0), d, Adim, pi_width=8, v_width=g("var12").shape[0])
    for i, p in enumerate(SEP_PATHS):
        _set(params, p, jnp.asarray(g(f"var{i}")))
    cfg = A.LaneCfg(obs_norm=jnp.asarray(True), obs_clip=jnp.float32(10.0), tanh=jnp.asarray(c["tanh"]),
                    softplus=jnp.asarray(c["softplus"]), init_std=jnp.float32(c["init_std"]), min_std=jnp.float32(1e-3),
                    ppo_eps=jnp.float32(c["eps"]), vclip_eps=jnp.float32(c["vclip"]), gae_lambda=jnp.float32(c["lam"]),
                    gamma=jnp.float32(0.99), lr=jnp.float32(1e-2), lr_horizon_steps=jnp.float32(g("lr_decay_steps")),
                    lr_end=jnp.float32(0.0), adam_b1=jnp.float32(0.9), adam_b2=jnp.float32(0.999),
                    adam_eps=jnp.float32(1e-7), grad_clip=jnp.float32(0.5), ent_coef=jnp.float32(0.0),
                    value_norm=jnp.asarray(True), adv_norm=jnp.asarray(c["adv_norm"]), recompute=jnp.asarray(True),
                    shared=jnp.asarray(False), baseline_cost=jnp.float32(1.0), reward_norm=jnp.asarray(False),
                    logp_presquash=jnp.asarray(True))
    st = A.Static(num_envs=B, unroll=T, epochs=E, minibatch=n // 4)
    shift = M.std_shift(cfg.init_std, cfg.min_std, cfg.softplus)
    lb = jnp.asarray(g("logits_b"))
    batch = A.Batch(obs=obs, act=jnp.asarray(g("act"))[:-1], mu_b=lb[:-1, :, :Adim],
                    std_b=M.std_of(lb[:-1, :, Adim:], shift, cfg.min_std, cfg.softplus),
                    v_b=jnp.asarray(g("base_b"))[:-1], rew=jnp.asarray(g("rew"))[1:], done=jnp.asarray(g("done"))[1:])
    ls = A.LaneState(params=params, opt=A.adam_init(params),
                     obs_ms=D.meanstd_update(D.meanstd_init(d), jnp.asarray(g("prior_obs"))),
                     val_ms=D.meanstd_update(D.meanstd_init(1), jnp.asarray(g("prior_targets"))[:, None]),
                     ret_ms=D.meanstd_init(1), ret_acc=jnp.zeros((B,)), env=None, obs=None,
                     it=jnp.zeros((), jnp.int32), bad=jnp.zeros((), bool))
    ls2, aux = A.update(ls, batch, cfg, st, jax.random.PRNGKey(0), perms=perms)
    assert int(ls2.opt.t) == int(g("opt_iterations")) == 12
    m, s = D.meanstd_get(ls2.obs_ms)
    _close(m, g("obs_mean"), "obs mean")
    _close(s, g("obs_std"), "obs std")
    m, s = D.meanstd_get(ls2.val_ms)
    _close(m[0], g("popart_mean"), "popart mean after 3 recomputes")
    _close(s[0], g("popart_std"), "popart std after 3 recomputes")
    for i, p in enumerate(SEP_PATHS):
        _close(_get(ls2.params, p), g(f"wend_{i}"), f"weights after 3 epochs {p}", rtol=2e-4)
    # mutation: the key-derived shuffle (any other permutation) must NOT reproduce the official weights
    ls3, _ = A.update(ls, batch, cfg, st, jax.random.PRNGKey(0))
    with pytest.raises(AssertionError):
        for i, p in enumerate(SEP_PATHS):
            _close(_get(ls3.params, p), g(f"wend_{i}"), "mutation", rtol=2e-4)
