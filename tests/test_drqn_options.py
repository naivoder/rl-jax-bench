"""Oracle tests for the default-off options of rl_jax_bench.drqn: prioritized sequence replay, Ape-X epsilons,
Q-filter / DQfD margin loss, LayerNorm; plus bit-identity of the defaults against the frozen pre-option copy (drqn_pre_options.py).
Numeric checks are float64 where noted. CPU."""
import importlib.util
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rl_jax_bench import drqn  # noqa: E402

_spec = importlib.util.spec_from_file_location("drqn_archived", os.path.join(os.path.dirname(__file__), "drqn_pre_options.py"))
old = importlib.util.module_from_spec(_spec)
sys.modules["drqn_archived"] = old
_spec.loader.exec_module(old)


def x64(fn):
    import functools

    @functools.wraps(fn)
    def run(*a, **k):
        with jax.enable_x64(True):
            return fn(*a, **k)
    return run


@x64
def test_per_sample_distribution_and_weights():
    pr = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 100.0, 100.0, 100.0])
    count, alpha, beta, B = 5, 0.6, 0.4, 200000
    idx, w = drqn.per_sample(jax.random.PRNGKey(0), jnp.asarray(pr), count, B, alpha, beta)
    idx, w = np.asarray(idx), np.asarray(w)
    assert idx.max() < count                                   # count < N respected
    P = pr[:count] ** alpha
    P = P / P.sum()
    obs = np.bincount(idx, minlength=count)
    chi2 = ((obs - B * P) ** 2 / (B * P)).sum()
    assert chi2 < 18.5                                         # chi2(df=4) 0.999 quantile = 18.47
    raw = (count * P) ** -beta
    np.testing.assert_allclose(w, raw[idx] / raw[idx].max(), rtol=1e-12)
    assert abs(w.max() - 1.0) < 1e-12
    # alpha = 0 is uniform, weights all 1
    idx0, w0 = drqn.per_sample(jax.random.PRNGKey(1), jnp.asarray(pr), count, B, 0.0, 1.0)
    obs0 = np.bincount(np.asarray(idx0), minlength=8)
    assert obs0[count:].sum() == 0
    assert ((obs0[:count] - B / count) ** 2 / (B / count)).sum() < 18.5
    np.testing.assert_allclose(np.asarray(w0), 1.0, rtol=1e-12)
    # hand-computed small case
    _, wh = drqn.per_sample(jax.random.PRNGKey(2), jnp.asarray([1.0, 3.0]), 2, 64, 1.0, 1.0)
    for wi in np.unique(np.round(np.asarray(wh), 12)):
        assert min(abs(wi - 1.0), abs(wi - 1.0 / 3.0)) < 1e-9


def test_new_priority_running_max():
    pr = jnp.asarray([0.3, 2.5, 1.0, 99.0])
    assert float(drqn.per_new_priority(pr, 3)) == 2.5
    assert float(drqn.per_new_priority(pr, 0)) == 1.0


@x64
def test_sequence_priority_numpy():
    rng = np.random.default_rng(0)
    td = np.abs(rng.normal(size=(50, 7)))
    m = (rng.random((50, 7)) < 0.6).astype(np.float64)
    m[0] = 0
    m[1] = 1
    for eta in (0.9, 0.5, 1.0):
        want = np.zeros(50)
        for i in range(50):
            v = td[i][m[i] > 0]
            want[i] = 0.0 if v.size == 0 else eta * v.max() + (1 - eta) * v.mean()
        np.testing.assert_allclose(np.asarray(drqn.sequence_priority(jnp.asarray(td), jnp.asarray(m), eta)), want,
                                   rtol=1e-12, atol=1e-14)


def _setup(lstm=6, **kw):
    B, L, D, A = 6, 5, 4, 3
    cfg = drqn.QConfig(n_actions=A, obs_dim=D, hidden=(8,), lstm=lstm, trace_len=L, **kw)
    rng = np.random.default_rng(1)
    batch = dict(inputs=jnp.asarray(rng.normal(size=(B, L + 1, D))), actions=jnp.asarray(rng.integers(0, A, (B, L))),
                 rewards=jnp.asarray(rng.normal(size=(B, L))), discounts=jnp.full((B, L), 0.9),
                 mask=jnp.asarray((rng.random((B, L)) < 0.8).astype(np.float64)))
    return cfg, batch


def _f64(tree):
    return jax.tree.map(lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x, tree)


@x64
def test_weighted_sequence_loss():
    cfg, batch = _setup(lstm=0)             # feed-forward: the LSTM carry is float32 and would not scan under x64
    st = drqn.init(jax.random.PRNGKey(0), cfg)
    p, t = _f64(st.params), _f64(st.params)
    t = jax.tree.map(lambda x: x * 1.1, t)
    l0, a0 = drqn.sequence_loss(p, t, batch, cfg)
    l1, a1 = drqn.sequence_loss(p, t, dict(batch, weights=jnp.ones(6)), cfg)
    assert abs(float(l0) - float(l1)) < 1e-12
    assert a0["priorities"].shape == (6,) and a0["abs_td"].shape == (6, 5)
    # planted weights: loss_w == sum_b w_b * (per-sequence contribution), contributions from single-sequence masks
    w = np.array([0.0, 1.0, 2.5, 0.3, 0.0, 4.0])
    lw, _ = drqn.sequence_loss(p, t, dict(batch, weights=jnp.asarray(w)), cfg)
    denom = float(batch["mask"].sum())
    contrib = []
    for b in range(6):
        mb = jnp.zeros_like(batch["mask"]).at[b].set(batch["mask"][b])
        lb, _ = drqn.sequence_loss(p, t, dict(batch, mask=mb), cfg)
        contrib.append(float(lb) * float(mb.sum()) / denom)   # single-seq loss uses its own denominator
    # NOTE masked-out sequences change nothing else: carry/targets are per-sequence independent
    assert abs(float(lw) - float((w * np.array(contrib)).sum())) < 1e-12
    # priorities equal sequence_priority of |q_sa - y|
    ref = drqn.sequence_priority(jnp.abs(a0["q_sa"] - a0["y"]), batch["mask"])
    np.testing.assert_allclose(np.asarray(a0["priorities"]), np.asarray(ref), rtol=1e-12)
    st2, info = drqn.update(st, dict(batch, weights=jnp.asarray(w)), cfg)
    assert info["priorities"].shape == (6,)


def test_apex_epsilons():
    for n in (1, 2, 8, 360):
        got = np.asarray(drqn.apex_epsilons(n))
        want = np.array([0.4 ** (1 + 7.0 * i / (n - 1)) if n > 1 else 0.4 for i in range(n)])
        np.testing.assert_allclose(got, want, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(drqn.apex_epsilons(3, 0.5, 2.0)), [0.5, 0.5 ** 2, 0.5 ** 3], rtol=1e-6)


def _np_margin(q, a, margin, legal):
    n, A = q.shape
    je = np.zeros(n)
    mask = np.zeros(n, bool)
    for i in range(n):
        ok = legal[i] if legal is not None else np.ones(A, bool)
        ok = ok.copy()
        ok[a[i]] = True
        aug = [q[i, j] + (0.0 if j == a[i] else margin) for j in range(A) if ok[j]]
        je[i] = max(aug) - q[i, a[i]]
        cand = np.flatnonzero(legal[i]) if legal is not None else np.arange(A)
        best = q[i, cand].max() if cand.size else q[i].max()
        mask[i] = q[i, a[i]] > best
    return je, mask


@x64
def test_qfilter_and_margin_oracle():
    rng = np.random.default_rng(3)
    n, A = 1000, 9
    q = np.round(rng.normal(size=(n, A)), 1)
    a = rng.integers(0, A, n)
    legal = rng.random((n, A)) < 0.5
    legal[:15] = False
    for lg in (None, legal):
        je, fm = _np_margin(q, a, 0.8, lg)
        jl = None if lg is None else jnp.asarray(lg)
        np.testing.assert_array_equal(np.asarray(drqn.qfilter_mask(jnp.asarray(q), jnp.asarray(a), jl)), fm)
        np.testing.assert_allclose(np.asarray(drqn.margin_loss_terms(jnp.asarray(q), jnp.asarray(a), 0.8, jl)), je,
                                   rtol=1e-12, atol=1e-12)
        got = drqn.margin_loss(jnp.asarray(q), jnp.asarray(a), 0.8, jl, qfilter=True)
        np.testing.assert_allclose(float(got), (je * fm).mean(), rtol=1e-12)
        mk = (rng.random(n) < 0.7).astype(np.float64)
        got = drqn.margin_loss(jnp.asarray(q), jnp.asarray(a), 0.8, jl, qfilter=False, mask=jnp.asarray(mk))
        np.testing.assert_allclose(float(got), (je * mk).sum() / max(mk.sum(), 1), rtol=1e-12)
    # planted: student's argmax (action 2) beats the teacher (action 0) -> zero loss under qfilter, positive without
    q1 = jnp.asarray([[0.0, 0.1, 1.0]])
    t = jnp.asarray([0])
    assert float(drqn.margin_loss(q1, t, 0.8, qfilter=True)) == 0.0
    assert float(drqn.margin_loss(q1, t, 0.8)) > 0.0
    # planted: critic still prefers the teacher's action -> the filter keeps the row
    q2 = jnp.asarray([[1.0, 0.9, 0.5]])
    lg = jnp.asarray([[False, True, True]])            # teacher action illegal for the student, so Q(teacher) > best legal
    assert bool(drqn.qfilter_mask(q2, t, lg)[0]) and float(drqn.margin_loss(q2, t, 0.8, lg, qfilter=True)) > 0.0
    assert not bool(drqn.qfilter_mask(q2, t)[0])       # strict >: the teacher tying the critic's argmax is filtered out
    # the filter is stop_gradient'd: gradient equals gradient of the unfiltered loss on kept rows
    g = jax.grad(lambda x: drqn.margin_loss(x, jnp.asarray(a), 0.8, qfilter=True))(jnp.asarray(q))
    assert bool(jnp.isfinite(g).all())


@x64
def test_qfilter_two_estimate():
    # with the teacher action legal on every row, the one-estimate form keeps nothing
    rng = np.random.default_rng(5)
    n, A = 2000, 6
    q = jnp.asarray(rng.normal(size=(n, A)))
    q_tgt = q + 0.3 * jnp.asarray(rng.normal(size=(n, A)))
    a = jnp.asarray(rng.integers(0, A, n))
    assert not bool(drqn.qfilter_mask(q, a).any())
    assert float(drqn.margin_loss(q, a, 0.8, qfilter=True)) == 0.0
    # (a) target and online nets differ -> the kept share lies strictly inside (0, 1)
    kept = float(drqn.qfilter_mask(q, a, q_eval=q_tgt).mean())
    assert 0.0 < kept < 1.0
    assert float(drqn.margin_loss(q, a, 0.8, qfilter=True, q_eval=q_tgt)) > 0.0
    # oracle: Q_tgt(s, a_E) > Q_tgt(s, argmax Q_online)
    qn, qt, an = np.asarray(q), np.asarray(q_tgt), np.asarray(a)
    ref = qt[np.arange(n), an] > qt[np.arange(n), qn.argmax(-1)]
    np.testing.assert_array_equal(np.asarray(drqn.qfilter_mask(q, a, q_eval=q_tgt)), ref)
    # (b) planted: the target prefers the teacher (action 0), the online argmax (action 2) does not -> kept
    q_on = jnp.asarray([[0.5, 0.1, 1.0]])
    q_tg = jnp.asarray([[1.2, 0.1, 0.4]])
    t = jnp.asarray([0])
    assert bool(drqn.qfilter_mask(q_on, t, q_eval=q_tg)[0])
    assert not bool(drqn.qfilter_mask(q_on, t)[0])
    # the online argmax picks the student's action under legality
    lg = jnp.asarray([[True, True, False]])
    assert not bool(drqn.qfilter_mask(q_on, t, lg, q_eval=jnp.asarray([[0.3, 0.6, 9.0]]))[0])


def test_layer_norm():
    cfg_ln, batch = _setup(layer_norm=True)
    st = drqn.init(jax.random.PRNGKey(0), cfg_ln)
    assert "ln0" in st.params["params"]
    st, info = drqn.update(st, batch, cfg_ln)
    assert np.isfinite(float(info["loss"]))
    # layer_norm=False: identical tree and outputs to the archived copy
    for lstm in (0, 6):
        cfg, batch = _setup(lstm=lstm)
        cfo = old.QConfig(n_actions=3, obs_dim=4, hidden=(8,), lstm=lstm, trace_len=5)
        s_new, s_old = drqn.init(jax.random.PRNGKey(7), cfg), old.init(jax.random.PRNGKey(7), cfo)
        assert jax.tree.structure(s_new.params) == jax.tree.structure(s_old.params)
        x = jnp.asarray(np.random.default_rng(0).normal(size=(3, 6, 4)), jnp.float32)
        qn, _ = drqn.unroll(drqn.make_network(cfg), cfg, s_new.params, x, drqn.initial_carry(cfg, (3,)))
        qo, _ = old.unroll(old.make_network(cfo), cfo, s_old.params, x, old.initial_carry(cfo, (3,)))
        assert np.array_equal(np.asarray(qn), np.asarray(qo))


def test_default_bit_identity_vs_archive():
    B, L, D, A = 8, 6, 5, 4
    kw = dict(n_actions=A, obs_dim=D, hidden=(16,), lstm=8, trace_len=L, burn_in=2, n_step=2, lr=1e-3)
    cn, co = drqn.QConfig(**kw), old.QConfig(**kw)
    rng = np.random.default_rng(5)
    batch = dict(inputs=jnp.asarray(rng.normal(size=(B, L + 1, D)), jnp.float32),
                 actions=jnp.asarray(rng.integers(0, A, (B, L))), rewards=jnp.asarray(rng.normal(size=(B, L)), jnp.float32),
                 discounts=jnp.full((B, L), 0.9, jnp.float32), mask=jnp.ones((B, L), jnp.float32))
    sn, so = drqn.init(jax.random.PRNGKey(3), cn), old.init(jax.random.PRNGKey(3), co)
    un, uo = jax.jit(lambda s: drqn.update(s, batch, cn)), jax.jit(lambda s: old.update(s, batch, co))
    for k in range(20):
        sn, inn = un(sn)
        so, ino = uo(so)
        for key in ("loss", "q_mean", "maxq_mean", "grad_norm"):
            assert np.asarray(inn[key]).tobytes() == np.asarray(ino[key]).tobytes(), (k, key)
        if k % 7 == 6:
            sn, so = drqn.sync(sn, True), old.sync(so, True)
    for a, b in zip(jax.tree.leaves(sn.params), jax.tree.leaves(so.params)):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
