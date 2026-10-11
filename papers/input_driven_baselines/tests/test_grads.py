"""Finite-difference gradient checks (jax.test_util.check_grads) on toy inputs, in float64.

Covers the A2C actor loss, the critic loss, the differentiable inner SGD loop and the full
second-order MAML meta-objective; plus the cross-fitting independence property.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import src  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

try:
    from jax.experimental import enable_x64
except ImportError:  # pragma: no cover
    enable_x64 = None

from src import algorithm as alg  # noqa: E402
from src.model import Actor, Critic  # noqa: E402

TOL = dict(atol=1e-4, rtol=1e-4)


def _x64():
    if enable_x64 is not None:
        return enable_x64()
    jax.config.update("jax_enable_x64", True)
    import contextlib

    return contextlib.nullcontext()


def _toy(key, n=3, T=5, d=4, K=3):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    obs = jax.random.normal(k1, (n, T, d), jnp.float64)
    acts = jax.random.randint(k2, (n, T), 0, K)
    adv = jax.random.normal(k3, (n, T), jnp.float64)
    G = jax.random.normal(k4, (n, T), jnp.float64)
    return obs, acts, adv, G


def _f64(tree):
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), tree)


def test_actor_loss_grads():
    with _x64():
        obs, acts, adv, _ = _toy(jax.random.PRNGKey(0))
        m = Actor((6,), 3, "tanh")  # tanh: smooth, so finite differences are meaningful
        p = _f64(m.init(jax.random.PRNGKey(1), obs[0, 0]))
        f = lambda q: alg.actor_loss(m.apply, q, obs, acts, adv, 0.3, 3)[0]
        check_grads(f, (p,), order=2, modes=("rev",), **TOL)


def test_value_loss_grads_all_reductions():
    with _x64():
        obs, _, _, G = _toy(jax.random.PRNGKey(2))
        m = Critic((6,), "tanh")
        p = _f64(m.init(jax.random.PRNGKey(3), obs[0, 0]))
        for red in ("mean", "sum", "sum_time"):
            check_grads(lambda q: alg.value_loss(m.apply, q, obs, G, red), (p,), order=2, modes=("rev",), **TOL)


def test_maml_second_order_meta_grad():
    """grad of phi -> L_H2(adapt(phi, H1)) + L_H1(adapt(phi, H2)) through 3 inner SGD steps."""
    with _x64():
        obs, _, _, G = _toy(jax.random.PRNGKey(4), n=4)
        m = Critic((5,), "tanh")
        phi = _f64(m.init(jax.random.PRNGKey(5), obs[0, 0]))
        x1, x2, G1, G2 = obs[:2], obs[2:], G[:2], G[2:]

        def meta(p):
            q1 = alg.sgd_adapt(m.apply, p, x1, G1, 3, 0.05, "mean")
            q2 = alg.sgd_adapt(m.apply, p, x2, G2, 3, 0.05, "mean")
            return alg.value_loss(m.apply, q1, x2, G2) + alg.value_loss(m.apply, q2, x1, G1)

        check_grads(meta, (phi,), order=1, modes=("rev",), **TOL)
        # maml_crossfit(first_order=False) must return exactly this gradient
        _, _, g, _ = alg.maml_crossfit(m.apply, phi, x1, G1, x2, G2, steps=3, lr=0.05, inner_reduction="mean",
                                       first_order=False)
        g_ref = jax.grad(meta)(phi)
        for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(g_ref)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)
        # FOMAML = gradient of the outer loss evaluated at the adapted parameters
        _, _, gfo, _ = alg.maml_crossfit(m.apply, phi, x1, G1, x2, G2, steps=3, lr=0.05, inner_reduction="mean",
                                         first_order=True)
        q1 = alg.sgd_adapt(m.apply, phi, x1, G1, 3, 0.05, "mean")
        q2 = alg.sgd_adapt(m.apply, phi, x2, G2, 3, 0.05, "mean")
        gfo_ref = jax.tree_util.tree_map(lambda a, b: a + b, jax.grad(lambda q: alg.value_loss(m.apply, q, x2, G2))(q1),
                                         jax.grad(lambda q: alg.value_loss(m.apply, q, x1, G1))(q2))
        for a, b in zip(jax.tree_util.tree_leaves(gfo), jax.tree_util.tree_leaves(gfo_ref)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)


def test_crossfit_baseline_independent_of_own_half_returns():
    """: the H2 baselines are bit-identical when H2's own returns change (they come from
    phi^1, fitted on H1 only), and do change when H1's returns change."""
    obs, _, _, G = _toy(jax.random.PRNGKey(6), n=4)
    obs, G = obs.astype(jnp.float32), G.astype(jnp.float32)
    m = Critic((5,))
    phi = m.init(jax.random.PRNGKey(7), obs[0, 0])
    kw = dict(steps=5, lr=0.01, inner_reduction="mean", clip=10.0)
    b1, b2, _, _ = alg.maml_crossfit(m.apply, phi, obs[:2], G[:2], obs[2:], G[2:], **kw)
    b1x, b2x, _, _ = alg.maml_crossfit(m.apply, phi, obs[:2], G[:2], obs[2:], G[2:] + 100.0, **kw)
    assert np.array_equal(np.asarray(b2), np.asarray(b2x))
    assert not np.allclose(np.asarray(b1), np.asarray(b1x))


def test_discounted_returns():
    r = jnp.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(np.asarray(alg.discounted_returns(r, 0.5)), [[1 + 1 + 0.75, 2 + 1.5, 3.0]])


def test_per_rollout_pg_matches_autodiff_sum():
    obs, acts, adv, _ = _toy(jax.random.PRNGKey(8))
    obs, adv = obs.astype(jnp.float32), adv.astype(jnp.float32)
    m = Actor((6,), 3)
    p = m.init(jax.random.PRNGKey(9), obs[0, 0])
    vecs = alg.per_rollout_pg(m.apply, p, obs, acts, adv)
    tot = jax.grad(lambda q: sum(alg.pg_surrogate(m.apply, q, obs[i], acts[i], adv[i]) for i in range(obs.shape[0])))(p)
    flat = jnp.concatenate([x.reshape(-1) for x in jax.tree_util.tree_leaves(tot)])
    np.testing.assert_allclose(np.asarray(vecs.sum(0)), np.asarray(flat), rtol=1e-5, atol=1e-5)
