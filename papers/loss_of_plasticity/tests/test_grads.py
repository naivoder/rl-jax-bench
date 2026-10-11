"""Gradient checks: finite differences (jax.test_util.check_grads) on a toy MLP for both losses, plus the analytic
identities d CE / d logits = softmax - onehot and d (yhat - y)^2 / d yhat = 2 (yhat - y) (official F.mse_loss)."""
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from src import algorithm as A
from src.model import ACT_ID, init_mlp

jax.config.update("jax_enable_x64", True)


def _setup(loss, act):
    st = A.Static(n_in=6, widths=(4, 3), n_out=3 if loss == "ce" else 1, loss=loss)
    p = init_mlp(jax.random.PRNGKey(0), 6, (4, 3), st.n_out, math.sqrt(2))
    p = jax.tree.map(lambda a: a.astype(jnp.float64), p)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 6), jnp.float64)
    y = jnp.asarray([2, 0]) if loss == "ce" else jnp.asarray([[0.7], [-1.2]], jnp.float64)
    return st, p, x, y


def test_check_grads_ce_and_mse():
    for loss in ("ce", "mse"):
        for act in ("tanh", "sigmoid", "swish", "elu"):  # smooth activations for finite differences
            st, p, x, y = _setup(loss, act)
            f = lambda p: A.loss_and_out(st, p, x, y, ACT_ID[act])[0]
            check_grads(f, (p,), order=1, modes=["rev"], atol=1e-6, rtol=1e-6)


def test_ce_logit_gradient_identity():
    z = jnp.asarray([0.3, -1.0, 2.0, 0.1], jnp.float64)
    y = 2
    g = jax.grad(lambda z: -jax.nn.log_softmax(z)[y])(z)
    assert np.allclose(np.asarray(g), np.asarray(jax.nn.softmax(z) - jax.nn.one_hot(y, 4)), atol=1e-12)


def test_mse_gradient_is_two_times_error():
    st, p, x, y = _setup("mse", "tanh")
    x, y = x[:1], y[:1]
    out = A.loss_and_out(st, p, x, y, ACT_ID["tanh"])[1][0]
    gb = jax.grad(lambda p: A.loss_and_out(st, p, x, y, ACT_ID["tanh"])[0])(p)["params"]["layer_2"]["bias"]
    assert np.allclose(np.asarray(gb), 2 * np.asarray(out - y)[0], atol=1e-12)


def test_sgd_l2_snp_update_rules():
    """L2: theta <- theta - lr (g + wd theta) on all params; S&P adds N(0, sigma^2) after the step (std reading)."""
    st, p, x, y = _setup("mse", "tanh")
    p = jax.tree.map(lambda a: a.astype(jnp.float32), p)
    x, y = x.astype(jnp.float32)[:1], y.astype(jnp.float32)[:1]
    g = jax.grad(lambda p: A.loss_and_out(st, p, x, y, ACT_ID["tanh"])[0])(p)
    hp = A.make_hp(lr=0.1, wd=0.01, act=ACT_ID["tanh"])
    s = A.init_state(st, p)
    s2, _ = A.learner_step(st, s, hp, x, y, jax.random.PRNGKey(0))
    exp = jax.tree.map(lambda a, b: a - 0.1 * (b + 0.01 * a), p, g)
    for a, b in zip(jax.tree.leaves(s2["params"]), jax.tree.leaves(exp)):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-6)
    # noise std: many perturbations of a zero tree
    z = {"w": jnp.zeros((200_000,))}
    n = A.perturb(z, jnp.asarray(1e-2), jax.random.PRNGKey(5))["w"]
    assert abs(float(n.std()) - 1e-2) < 1e-4
