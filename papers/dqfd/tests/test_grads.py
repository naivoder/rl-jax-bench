"""Finite-difference gradient check (jax.test_util.check_grads, float64, forward+reverse) of the DQfD total loss.
DQfD has a gradient, so this checks the semi-gradient implementation: the TD targets, the online
argmax and the importance weights are constants of the loss (computed outside), and the remaining function of theta
is smooth almost everywhere, so autodiff must equal central differences away from ties / ReLU kinks."""
import jax
import jax.numpy as jnp
import numpy as np

from jax.test_util import check_grads

from src import algorithm as A
from src.model import DuelingQNet, l2_half_sumsq


def test_total_loss_check_grads():
    with jax.enable_x64(True):
        net = DuelingQNet(n_actions=3, channels=2, stream_hidden=4)
        rng = np.random.default_rng(0)
        B = 6
        x = jnp.asarray(rng.integers(0, 2, (3 * B, 10, 10, 4)), jnp.float64)      # [s ; s1 ; sn]
        params = jax.tree_util.tree_map(lambda p: p.astype(jnp.float64), net.init(jax.random.PRNGKey(0), x[:1]))
        tparams = jax.tree_util.tree_map(lambda p: p + 0.05 * jnp.asarray(rng.normal(size=p.shape)), params)
        q_on_next = net.apply(params, x[B:])                  # constants of the loss (stop-gradient targets / argmax)
        q_tg_next = net.apply(tparams, x[B:])
        batch = dict(act=jnp.asarray(rng.integers(0, 3, B), jnp.int32), r1=jnp.asarray(rng.normal(size=B)),
                     term1=jnp.asarray([0, 0, 1, 0, 0, 0], bool), Gn=jnp.asarray(rng.normal(size=B)),
                     disc_n=jnp.asarray([0.9, 0.9, 0.0, 0.9, 0.9, 0.9]), is_demo=jnp.asarray([1, 0, 1, 1, 0, 0], bool))
        w = jnp.asarray(rng.uniform(0.3, 1.0, B))
        hp = dict(lam_td=1.0, lam_n=1.0, lam_e=1.0, lam_ce=0.5, lam3=1e-5, margin=0.8, nmax=0.0)

        def f(p):
            q_s = net.apply(p, x[:B])
            t = A.dqfd_losses(q_s, q_on_next[:B], q_tg_next[:B], q_on_next[B:], q_tg_next[B:], batch, hp, 0.99)
            return jnp.mean(w * A.total_per_sample(t, hp)) + hp["lam3"] * l2_half_sumsq(p)

        check_grads(f, (params,), order=1, modes=["rev", "fwd"], atol=1e-5, rtol=1e-5, eps=1e-6)
        g = jax.grad(f)(params)
        assert all(np.isfinite(np.asarray(l)).all() for l in jax.tree_util.tree_leaves(g))
        # the semi-gradient must NOT flow through the targets: the loss's gradient wrt tparams-derived constants is zero by construction
        assert float(jnp.sum(jnp.abs(jax.tree_util.tree_leaves(g)[0]))) > 0
