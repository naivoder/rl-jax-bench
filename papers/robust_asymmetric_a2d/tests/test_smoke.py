"""Smoke test (< 30 s on CPU): env/obs shapes, network shapes, and one iteration of every method on tiny settings."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src import data as D, model as M, train as T


def test_env_and_obs_shapes():
    for name, sd, pd in [('FL', 34, 25), ('TD', 30, 30), ('TD1', 54, 54)]:
        spec = D.get_spec(name)
        st = D.reset(spec, jax.random.PRNGKey(0))
        st, r, d = D.step(spec, st, 1)
        lat = D.latent(spec, st)
        assert D.state_vec(spec, lat).shape == (sd,) and D.partial_vec(spec, lat).shape == (pd,)
        assert float(r) == -2.0 and not bool(d)
    spec = D.get_spec('TD')
    img = D.image(spec, jnp.asarray(D.image_table('TD')), D.latent(spec, D.reset(spec, jax.random.PRNGKey(1))),
                  jax.random.PRNGKey(2))
    assert img.shape == (42, 42, 3) and float(img.min()) >= 0 and float(img.max()) <= 1


def test_network_shapes():
    x = jnp.zeros((3, 42, 42, 3))
    pol, val = M.make_policy('image'), M.make_value('image')
    lp = pol.apply(pol.init(jax.random.PRNGKey(0), x), x)
    assert lp.shape == (3, 4) and np.allclose(np.exp(lp).sum(-1), 1, atol=1e-5)
    assert val.apply(val.init(jax.random.PRNGKey(0), x), x).shape == (3,)


TINY = dict(iters=1, batch=64, buffer=128, val_epochs=1, q_epochs=1, n_mb=4, il_mb=16)


@pytest.mark.parametrize('arm,env', [('a2d_compact', 'FL'), ('rl_mdp', 'TD'), ('a2d_q', 'TD1')])
def test_one_iteration(arm, env):
    cfg = T.Config(**dict(T.ARMS[arm], env=env, **TINY))
    tr = T.Trainer(cfg)
    key = jax.random.PRNGKey(0)
    ts = tr.init(key)
    ts2, info = jax.jit(tr.iteration)(ts, jax.random.fold_in(key, 1))
    leaves0 = jax.tree_util.tree_leaves(ts.params)
    leaves1 = jax.tree_util.tree_leaves(ts2.params)
    assert all(np.all(np.isfinite(np.asarray(l))) for l in leaves1)
    assert any(not np.allclose(a, b) for a, b in zip(leaves0, leaves1)), 'one step must change some parameter'
    assert int(ts2.it) == 1
