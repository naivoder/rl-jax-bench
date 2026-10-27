"""Gradient check.  JSRL itself (the switch rule + curriculum) has NO gradient: it is a data-collection rule wrapped around the learner.
So the differentiable object is the TRAIN_POLICY loss it feeds (the earlier DQfD reproduction's PDD double-Q 1-step + n-step TD loss), and we check
jax.test_util.check_grads (finite differences, float64) of that loss w.r.t. the Q values on a toy batch, and that the TD target is
stop-gradient (no gradient flows to the target/online-argmax inputs).  The estimator-level correctness of the mechanism is tested
in tests/test_lock.py (closed forms of the combination lock)."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np
from jax.test_util import check_grads
from src import algorithm as AL

import pytest


@pytest.fixture(autouse=True)
def _x64():
    """float64 only inside this module's tests (dtype policy: float32 everywhere else)."""
    with jax.enable_x64(True):
        yield


def _batch(rng, B=6, A=4):
    f = lambda *s: jnp.asarray(rng.normal(size=s))
    return dict(act=jnp.asarray(rng.integers(0, A, B)), r1=f(B), term1=jnp.asarray(rng.random(B) < 0.3), Gn=f(B),
                disc_n=jnp.asarray(0.99 ** 10 * (rng.random(B) < 0.7)), is_demo=jnp.asarray(rng.random(B) < 0.5)), f


def test_td_loss_grad_wrt_q():
    rng = np.random.default_rng(0)
    batch, f = _batch(rng)
    hp = dict(nmax=0.0, margin=0.8, huber=0.0, lam_td=1.0, lam_n=1.0, lam_e=0.0, lam_ce=0.0)
    q_on1, q_tg1, q_onn, q_tgn = f(6, 4), f(6, 4), f(6, 4), f(6, 4)

    def loss(q_s):
        t = AL.dqfd_losses(q_s, q_on1, q_tg1, q_onn, q_tgn, batch, hp, 0.99)
        return jnp.mean(AL.total_per_sample(t, hp))

    check_grads(loss, (f(6, 4),), order=2, modes=["rev"], atol=1e-5, rtol=1e-5)


def test_targets_are_stop_gradient():
    rng = np.random.default_rng(1)
    batch, f = _batch(rng)
    hp = dict(nmax=0.0, margin=0.8, huber=0.0, lam_td=1.0, lam_n=1.0, lam_e=0.0, lam_ce=0.0)
    q_s = f(6, 4)

    def loss(q_on1, q_tg1, q_onn, q_tgn):
        t = AL.dqfd_losses(q_s, q_on1, q_tg1, q_onn, q_tgn, batch, hp, 0.99)
        return jnp.mean(AL.total_per_sample(t, hp))

    # the loss DOES depend on the target values numerically, but dqfd_losses wraps y in stop_gradient -> zero gradient to them
    g = jax.grad(loss, argnums=(0, 1, 2, 3))(f(6, 4), f(6, 4), f(6, 4), f(6, 4))
    assert all(float(jnp.abs(x).max()) == 0.0 for x in g)
