"""Smoke test (< 30 s on CPU): shapes and one step of every arm (BP, L2, S&P, CBP-SGD, CBP-Adam) on both problems."""
import math

import jax
import jax.numpy as jnp
import numpy as np

from src import algorithm as A
from src import data as D
from src import diagnostics as G
from src.model import ACT_ID, init_mlp


def _arms():
    return [A.make_hp(lr=0.01), A.make_hp(lr=0.01, wd=1e-3), A.make_hp(lr=0.01, wd=1e-3, sigma=1e-2),
            A.make_hp(lr=0.01, rho=0.1, mt=0)]


def test_scr_stream_and_step():
    spec = D.scr_spec("flip_one")
    X, Y, tgt = D.scr_stream(jax.random.PRNGKey(0), jax.random.PRNGKey(1), spec, 2)
    assert X.shape == (2 * spec.T, 20) and Y.shape == (2 * spec.T,)
    assert set(np.unique(np.asarray(X))) <= {0.0, 1.0}
    # exactly one flipping bit differs between the two periods, random bits vary
    assert int(jnp.sum(X[0, :15] != X[spec.T, :15])) == 1
    st = A.Static(n_in=20, widths=(5,), n_out=1, replace_mode="bernoulli")
    p = init_mlp(jax.random.PRNGKey(2), 20, (5,), 1, 5.0 / 3.0)
    hps = A.stack_hp([h._replace(act=jnp.asarray(ACT_ID["tanh"])) for h in _arms()])
    s = A.init_state(st, p)
    states = jax.tree.map(lambda a: jnp.stack([a] * 4), s)
    s2, info = jax.jit(jax.vmap(lambda s, h: A.learner_step(st, s, h, X[:1], Y[:1, None], jax.random.PRNGKey(3))))(
        states, hps)
    assert info["loss"].shape == (4,) and np.all(np.isfinite(np.asarray(info["loss"])))
    assert jax.tree.structure(s2) == jax.tree.structure(states)
    assert s2["params"]["params"]["layer_0"]["kernel"].shape == (4, 20, 5)


def test_mnist_shapes_and_cbp_adam_step():
    st = A.Static(n_in=784, widths=(16, 16, 16), n_out=10, loss="ce", opt="adam")
    p = init_mlp(jax.random.PRNGKey(0), 784, (16, 16, 16), 10, math.sqrt(2))
    s = A.init_state(st, p)
    x = jax.random.uniform(jax.random.PRNGKey(1), (1, 784))
    y = jnp.asarray([3])
    hp = A.make_hp(lr=1e-3, rho=0.05, mt=0)
    step = jax.jit(lambda s, k: A.learner_step(st, s, hp, x, y, k))
    for t in range(3):
        s, info = step(s, jax.random.PRNGKey(t))
    assert info["correct"].shape == () and info["n_rep"].shape == (3,)
    assert int(info["n_rep"].sum()) >= 1
    _, hs = A.loss_and_out(st, s["params"], jnp.tile(x, (50, 1)), jnp.zeros(50, jnp.int32), 0)[1]
    d = G.layer_diagnostics(hs)
    assert d["erank_per_layer"].shape == (3,) and 0.0 <= float(d["dead_pct"]) <= 100.0


def test_vmapped_runs_match_single():
    """vmap over arms gives the same result as running each arm alone (arms are independent)."""
    st = A.Static(n_in=20, widths=(5,), n_out=1)
    p = init_mlp(jax.random.PRNGKey(2), 20, (5,), 1, math.sqrt(2))
    x = jnp.ones((1, 20))
    y = jnp.ones((1, 1))
    hps = _arms()
    s = A.init_state(st, p)
    states = jax.tree.map(lambda a: jnp.stack([a] * len(hps)), s)
    out, _ = jax.vmap(lambda s, h: A.learner_step(st, s, h, x, y, jax.random.PRNGKey(0)))(states, A.stack_hp(hps))
    for i in (0, 3):
        one, _ = A.learner_step(st, s, hps[i], x, y, jax.random.PRNGKey(0))
        a = one["params"]["params"]["layer_0"]["kernel"]
        b = out["params"]["params"]["layer_0"]["kernel"][i]
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-6)
