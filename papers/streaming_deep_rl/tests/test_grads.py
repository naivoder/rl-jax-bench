"""Gradient checks (finite differences, jax.test_util.check_grads) on the differentiable pieces: parameter-free
LayerNorm (incl. its closed-form backward), the LN Q-network output, the stream-AC policy objective
log pi(A|S) + tau*sign(delta)*H (App. E) and the critic. ObGD itself has no gradient; its closed form is tested in
tests/test_toys.py (toy, exact) and against the official optimizer in tests/test_official_parity.py."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from src import algorithm as alg  # noqa: E402
from src import model as mdl  # noqa: E402



def x64(fn):
    """run a test in float64 without leaking jax_enable_x64 into other test modules"""
    def wrapped():
        with jax.enable_x64(True):
            fn()
    wrapped.__name__ = fn.__name__
    return wrapped


@x64
def test_layernorm_fd():
    x = jax.random.normal(jax.random.PRNGKey(0), (7,), jnp.float64) * 3 + 1
    check_grads(mdl.layer_norm, (x,), order=2, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)


@x64
def test_layernorm_closed_form_backward():
    """dL/da = (1/s)(g - mean(g) - a_hat * mean(g * a_hat))."""
    a = jax.random.normal(jax.random.PRNGKey(1), (7,), jnp.float64) * 2
    g = jax.random.normal(jax.random.PRNGKey(2), (7,), jnp.float64)
    _, vjp = jax.vjp(mdl.layer_norm, a)
    s = jnp.sqrt(jnp.var(a) + mdl.LN_EPS)
    ah = (a - a.mean()) / s
    closed = (g - g.mean() - ah * jnp.mean(g * ah)) / s
    np.testing.assert_allclose(vjp(g)[0], closed, atol=1e-12)


def _f64(t):
    return jax.tree.map(lambda x: x.astype(jnp.float64), t)


@x64
def test_qnet_fd():
    net = mdl.QNetMinAtar(3)
    p = _f64(mdl.init_params(jax.random.PRNGKey(3), mdl.template(net, (10, 10, 4)), mdl.INIT_DENSE0))
    x = jax.random.normal(jax.random.PRNGKey(4), (10, 10, 4), jnp.float64)
    check_grads(lambda pp: net.apply({"params": pp}, x, 1.0)[1], (p,), order=1, modes=("rev",), atol=1e-5, rtol=1e-5, eps=1e-6)  # small eps: LeakyReLU kinks


@x64
def test_policy_objective_fd():
    actor = mdl.GaussianActor(3)
    p = _f64(mdl.init_params(jax.random.PRNGKey(5), mdl.template(actor, (12,)), mdl.INIT_DENSE0))
    s = jax.random.normal(jax.random.PRNGKey(6), (12,), jnp.float64)
    a = jnp.array([0.3, -1.2, 0.8])

    def obj(pp):
        m, sd = actor.apply({"params": pp}, s, 1.0)
        return alg.gaussian_logp(a, m, sd) + 0.01 * -1.0 * alg.gaussian_entropy(sd)

    check_grads(obj, (p,), order=1, modes=("rev",), atol=1e-5, rtol=1e-5, eps=1e-6)  # small eps: LeakyReLU kinks


@x64
def test_gaussian_terms_match_scipy():
    from scipy import stats
    mu, sd, a = np.array([0.1, -0.5]), np.array([0.7, 1.9]), np.array([0.4, 2.0])
    np.testing.assert_allclose(alg.gaussian_logp(a, mu, sd), stats.norm(mu, sd).logpdf(a).sum(), rtol=1e-12)
    np.testing.assert_allclose(alg.gaussian_entropy(sd), stats.norm(mu, sd).entropy().sum(), rtol=1e-12)


@x64
def test_softplus_torch_threshold():
    x = jnp.array([-30.0, -1.0, 0.0, 19.9, 20.0, 20.1, 50.0], jnp.float64)
    want = np.where(np.asarray(x) > 20, np.asarray(x), np.log1p(np.exp(np.asarray(x))))
    np.testing.assert_allclose(mdl.softplus_torch(x), want, rtol=1e-12)
