"""Finite-difference gradient checks (float64) of every loss mode and of the recurrent unroll.

ADVISOR's weight w is a stop-gradient (official .detach()), so the analytic gradient w.r.t. the aux logits is the
gradient of the aux CE alone; we check that directly instead of finite-differencing through w."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from src import algorithm as A
from src import model as M


@pytest.fixture(autouse=True)
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def _batch(rng, T=5, B=3, Aa=7):
    return dict(action=jnp.asarray(rng.randint(Aa, size=(T, B))), logp=jnp.asarray(-rng.rand(T, B) * 2),
                adv_norm=jnp.asarray(rng.randn(T, B)), value=jnp.asarray(rng.randn(T, B)),
                ret=jnp.asarray(rng.randn(T, B)), label=jnp.asarray(rng.randint(Aa, size=(T, B))),
                label_valid=jnp.asarray(rng.rand(T, B) < 0.8))


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_loss_modes_fd(mode):
    rng = np.random.RandomState(mode)
    b = _batch(rng)
    lg, ax, v = (jnp.asarray(rng.randn(5, 3, 7)), jnp.asarray(rng.randn(5, 3, 7)), jnp.asarray(rng.randn(5, 3)))
    clip = 0.5   # wide clip so the random point is away from the clip kinks

    def L(lg_, v_, ax_=ax):
        return A.loss_from_terms(A.per_step_terms(lg_, ax_, v_, b, clip), jnp.int32(mode), 8.0, 0.0)[0]
    check_grads(L, (lg, v), order=1, modes=("rev",), atol=1e-6, rtol=1e-6)
    if mode != 3:
        check_grads(lambda a: L(lg, v, a), (ax,), order=1, modes=("rev",), atol=1e-6, rtol=1e-6)
    else:
        g = jax.grad(lambda a: L(lg, v, a))(ax)
        aux_ce = lambda a: -A.masked_mean(A.take(A.log_softmax(a), b["label"]), b["label_valid"].astype(jnp.float64))
        np.testing.assert_allclose(np.asarray(g), np.asarray(jax.grad(aux_ce)(ax)), atol=1e-12)
        check_grads(aux_ce, (ax,), order=1, modes=("rev",), atol=1e-6, rtol=1e-6)


def test_pdnet_unroll_fd():
    net = M.PDNet(hidden=8)
    B, T = 2, 6
    obs = jnp.asarray(np.random.RandomState(0).randint(4, size=(T, B)))
    start = jnp.zeros((T, B), bool).at[0].set(True).at[3, 1].set(True)
    params = net.init(jax.random.PRNGKey(0), obs[0], net.initial_carry(B), start[0])
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)

    def f(p):
        def body(c, x):
            lg, ax, v, c = net.apply(p, x[0], c, x[1])
            return c, (lg, ax, v)
        _, (lg, ax, v) = jax.lax.scan(body, jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), net.initial_carry(B)),
                                      (obs, start))
        return jnp.sum(jnp.sin(lg)) + jnp.sum(ax ** 2) * 0.1 + jnp.sum(v)
    check_grads(f, (params,), order=1, modes=("rev",), atol=1e-5, rtol=1e-5)


def test_gae_matches_allenact_loop():
    rng = np.random.RandomState(3)
    T, B = 9, 4
    r, v, d = rng.randn(T, B), rng.randn(T, B), rng.rand(T, B) < 0.3
    last = rng.randn(B)
    adv, ret = A.gae(jnp.asarray(r), jnp.asarray(v), jnp.asarray(d), jnp.asarray(last), 0.99, 0.95)
    # AllenAct RolloutStorage.compute_returns, masks[t+1] = 1 - done[t]
    vp = np.concatenate([v, last[None]])
    masks = np.concatenate([np.ones((1, B)), 1.0 - d])
    gae, rets = 0.0, np.zeros((T, B))
    for t in reversed(range(T)):
        delta = r[t] + 0.99 * vp[t + 1] * masks[t + 1] - vp[t]
        gae = delta + 0.99 * 0.95 * masks[t + 1] * gae
        rets[t] = gae + vp[t]
    np.testing.assert_allclose(np.asarray(ret), rets, atol=1e-12)
    np.testing.assert_allclose(np.asarray(adv), rets - v, atol=1e-12)


def test_advisor_imitation_warmup_is_main_plus_aux_ce():
    """Official AdvisorImitationStage (warm-up of DAgger->ADV, BC^tf=1->ADV): masked CE on the main head plus masked
    CE on the aux head, each sum(mask * -log p(a*)) / max(sum(mask), 1). Closed-form check on a random batch."""
    import numpy as np
    import jax.numpy as jnp
    from src import algorithm as A
    r = np.random.RandomState(3)
    T, B, K = 6, 4, 5
    lg, ax = jnp.asarray(r.randn(T, B, K)), jnp.asarray(r.randn(T, B, K))
    lab = jnp.asarray(r.randint(0, K, (T, B)))
    valid = jnp.asarray(r.rand(T, B) < 0.8)
    batch = dict(action=lab, logp=jnp.zeros((T, B)), adv_norm=jnp.zeros((T, B)), value=jnp.zeros((T, B)),
                 ret=jnp.zeros((T, B)), label=lab, label_valid=valid)
    terms = A.per_step_terms(lg, ax, jnp.zeros((T, B)), batch, 0.1)
    tot, _ = A.loss_from_terms(terms, A.MODE_ID["advisor_imitation"], 8.0, 0.0)
    lpm = np.take_along_axis(np.asarray(jax_log_softmax(lg)), np.asarray(lab)[..., None], -1)[..., 0]
    lpa = np.take_along_axis(np.asarray(jax_log_softmax(ax)), np.asarray(lab)[..., None], -1)[..., 0]
    m = np.asarray(valid, float)
    ref = -(m * lpm).sum() / max(m.sum(), 1) - (m * lpa).sum() / max(m.sum(), 1)
    assert abs(float(tot) - ref) < 1e-5 * max(1.0, abs(ref))


def jax_log_softmax(z):
    import jax
    return jax.nn.log_softmax(z, axis=-1)
