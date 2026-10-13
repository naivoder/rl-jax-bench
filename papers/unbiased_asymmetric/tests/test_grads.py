"""Finite-difference gradient checks (float64, jax.test_util.check_grads):
* the custom-VJP sequence GRU (model.gru_scan) -- the only hand-written backward pass in the repo;
* the critic loss (semi-gradient: target values are constants) and the actor loss (policy + negentropy, critic
  values are constants) of every arm, w.r.t. all parameters, on a tiny network and a real Heaven-Hell batch.
The estimator's statistical correctness (unbiasedness of V(h), V(h,s); bias of V(s)) is toy check (src/toys.py)."""
from _helpers import REPO  # noqa: F401

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from algorithm import make_losses, make_rollout  # noqa: E402
from envs import make_env  # noqa: E402
from model import METHODS, NetSpec, _gru_gates, gru_scan, init_params  # noqa: E402

ENV = make_env("heavenhell_3")
TINY = NetSpec(ENV.kind, ENV.n_actions, ENV.n_obs, ENV.n_states, emb=3, hidden=4, head=(5,))


def _f64(t):
    return jax.tree.map(lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x, t)


def test_custom_gru_vjp_finite_difference_and_autodiff():
    with jax.enable_x64(True):
        k = jax.random.split(jax.random.PRNGKey(0), 5)
        T, H = 6, 3
        gi = jax.random.normal(k[0], (T, 3 * H), jnp.float64)
        w = 0.5 * jax.random.normal(k[1], (H, 3 * H), jnp.float64)
        b = jax.random.normal(k[2], (3 * H,), jnp.float64)
        h0 = jax.random.normal(k[3], (H,), jnp.float64)
        c = jax.random.normal(k[4], (T, H), jnp.float64)
        f = lambda gi, w, b, h0: jnp.sum(jnp.tanh(gru_scan(gi, w, b, h0)) * c)  # noqa: E731
        check_grads(f, (gi, w, b, h0), order=1, modes=["rev"], eps=1e-6, atol=1e-7, rtol=1e-7)

        def plain(gi, w, b, h0):
            def body(h, g):
                hn = _gru_gates(g, h @ w + b, h, H)[3]
                return hn, hn
            return jnp.sum(jnp.tanh(jax.lax.scan(body, h0, gi)[1]) * c)
        for a, bb in zip(jax.grad(f, (0, 1, 2, 3))(gi, w, b, h0), jax.grad(plain, (0, 1, 2, 3))(gi, w, b, h0)):
            assert np.allclose(a, bb, atol=1e-12)


@pytest.mark.parametrize("method", list(METHODS))
def test_losses_finite_difference(method):
    pa, pc = init_params(TINY, method, jax.random.PRNGKey(1))
    ep = jax.vmap(make_rollout(ENV, TINY, method, 12), (None, 0))(pa, jax.random.split(jax.random.PRNGKey(2), 2))
    with jax.enable_x64(True):
        pa, pc, ep = _f64(pa), _f64(pc), _f64(ep)
        pt = jax.tree.map(lambda x: x + 0.1, pc)
        _, closs, aloss = make_losses(TINY, method, 0.99)
        check_grads(lambda p: closs(p, pt, ep), (pc,), order=1, modes=["rev"], eps=1e-6, atol=1e-6, rtol=1e-6)
        check_grads(lambda p: aloss(p, pc, ep, 0.3)[0], (pa,), order=1, modes=["rev"], eps=1e-6, atol=1e-6,
                    rtol=1e-6)
        # semi-gradient: no gradient flows into the target critic / the critic inside the actor loss
        assert all(float(jnp.max(jnp.abs(g))) == 0.0 for g in jax.tree.leaves(jax.grad(lambda t: closs(pc, t, ep))(pt)))
        assert all(float(jnp.max(jnp.abs(g))) == 0.0
                   for g in jax.tree.leaves(jax.grad(lambda c: aloss(pa, c, ep, 0.3)[0])(pc)))
