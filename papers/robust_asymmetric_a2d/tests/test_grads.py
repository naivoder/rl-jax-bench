"""Gradient checks (float64): finite-difference check_grads of every differentiated objective on toy inputs, the
TRPO Fisher-vector product against finite differences of the KL gradient, and the A2D importance-weighted policy
gradient (Eq. 26) against its on-policy closed form."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

jax.config.update('jax_enable_x64', True)
from src import algorithm as AL, model as M  # noqa: E402

N = 32
rng = np.random.RandomState(0)
X = jnp.asarray(rng.randn(N, 6))
A = jnp.asarray(rng.randint(0, 4, N))
ADV = jnp.asarray(rng.randn(N) * 5)
pol = M.PolicyNet(kind='flat', out_gain=1.0)
P0 = pol.init(jax.random.PRNGKey(0), X[0])
P0 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), P0)
OLD = jnp.take_along_axis(pol.apply(P0, X), A[:, None], 1)[:, 0] + 0.1 * jnp.asarray(rng.randn(N))


def test_surrogate_grads():
    for ent, reg in [(10.0, True), (0.02, False), (0.0, True)]:
        check_grads(lambda p: AL.surrogate(pol.apply(p, X), A, OLD, ADV, ent, reg), (P0,), order=1, modes=['rev'],
                    atol=1e-5, rtol=1e-5)


def test_kl_loss_grads():
    e = pol.apply(jax.tree_util.tree_map(lambda x: x * 1.3, P0), X)
    check_grads(lambda p: AL.kl_loss(e, pol.apply(p, X)), (P0,), order=1, modes=['rev'], atol=1e-6, rtol=1e-6)


def test_value_loss_grads():
    old = jnp.asarray(rng.randn(N)); vs = old + jnp.asarray(rng.randn(N) * 0.5)
    nd = jnp.asarray(rng.rand(N) > 0.1)
    check_grads(lambda v: AL.value_loss_gae(v, old, ADV, nd, jnp.ones(N), 0.2), (vs,), order=1, modes=['rev'])
    check_grads(lambda v: AL.value_loss_returns(v, ADV, nd, jnp.ones(N)), (vs,), order=2, modes=['rev'])


def test_fisher_vector_product():
    from jax.flatten_util import ravel_pytree
    f0, unr = ravel_pytree(P0)
    p0 = jax.lax.stop_gradient(jnp.exp(pol.apply(P0, X)))
    kl = lambda f: jnp.mean(AL.kl_probs(p0, jnp.exp(pol.apply(unr(f), X))))
    v = jnp.asarray(rng.randn(f0.shape[0]))
    hvp = jax.jvp(jax.grad(kl), (f0,), (v,))[1]
    eps = 1e-5
    fd = (jax.grad(kl)(f0 + eps * v) - jax.grad(kl)(f0 - eps * v)) / (2 * eps)
    np.testing.assert_allclose(hvp, fd, rtol=1e-4, atol=1e-8)
    assert float(v @ hvp) > 0                    # Fisher is PSD


def test_a2d_is_gradient_matches_onpolicy_closed_form():
    """sum_a pi_psi(a|b) w(a) A(a) grad log pi_theta(a|s) == sum_a pi_theta(a|s) A(a) grad log pi_theta(a|s)."""
    z = jnp.asarray(rng.randn(4)); lpl = jax.nn.log_softmax(jnp.asarray(rng.randn(4))); Av = jnp.asarray(rng.randn(4))
    g_is = jax.grad(lambda z: jnp.sum(jnp.exp(lpl) * AL.importance_weight(jax.nn.log_softmax(z), lpl, 0.0) * Av))(z)
    pz = jax.nn.softmax(z)
    np.testing.assert_allclose(g_is, pz * (Av - jnp.sum(pz * Av)), atol=1e-10)
