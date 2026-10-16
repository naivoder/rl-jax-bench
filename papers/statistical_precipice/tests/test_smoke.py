"""Smoke test (< 30 s on CPU, JAX_PLATFORMS=cpu): shapes of every estimator, one bootstrap, every CI method, the
case-study procedures, and one vendored-PPO update in a float32 subprocess (the statistics enable x64)."""
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_estimator_shapes_and_one_bootstrap():
    import jax
    import jax.numpy as jnp
    from src import algorithm as A
    assert jax.config.jax_enable_x64
    k = A.branch_key(0, A.EXP_IDS["tests"])
    x = jax.random.uniform(k, (5, 7), jnp.float64)
    y = jax.random.uniform(jax.random.fold_in(k, 1), (4, 7), jnp.float64)
    assert A.aggregates4(x).shape == (4,) and A.aggregates4(x).dtype == jnp.float64
    assert A.score_profile(x, np.linspace(0, 1, 11)).shape == (11,)
    assert A.poi(x, y).shape == ()
    b, ci, d = A.interval_estimate(k, (x,), A.aggregates4, 300)
    assert b.shape == (4,) and ci.shape == (2, 4) and d.shape == (300, 4)
    jk = A.jackknife((x,), A.aggregates4)
    for m in ("percentile", "basic", "bc", "bca"):
        ci = A.ci_from_dist(d, b, m, jk=jk)
        assert ci.shape == (2, 4) and np.all(np.asarray(ci[0]) <= np.asarray(ci[1]) + 1e-12)
    r = A.rank_matrix(k, (x, y[:, :7], x + 0.1), 500)
    assert r.shape == (7, 3, 3) and np.allclose(np.asarray(r).sum(1), 1) and np.allclose(np.asarray(r).sum(2), 1)
    pool = jax.random.uniform(k, (20, 7), jnp.float64)
    assert A.sampling_distribution(k, pool, A.agg_iqm, 3, 50).shape == (50, 1)
    # lift / diff / trimmed-MSE / coverage are exercised by tests/test_grads.py and the gate scripts


PPO_STEP = r"""
import os, sys, time
sys.path.insert(0, %r)
import jax, jax.numpy as jnp
from src import train as T
from src.ppo8 import train as p8
assert not jax.config.jax_enable_x64
T.vendored_hashes()
base, e6 = T.load_e6_config(os.path.join(%r, 'src/configs/e6.yaml'),
                            ['num_steps=16', 'total_timesteps=128', 'seeds=[1,2]'])
cfg = dict(base, env_id='CartPole-v1', **p8.ARMS['shared'])
ppo = p8.PPO(cfg)
runner, consts = jax.jit(jax.vmap(ppo.init))(jnp.asarray([1, 2], jnp.int32))
new, m = jax.jit(jax.vmap(ppo.update))(runner, consts)
assert new.update.tolist() == [1, 1] and m['loss'].shape == (2,) and bool(jnp.all(jnp.isfinite(m['loss'])))
print('OK', float(m['loss'][0]))
"""


def test_one_ppo_update_float32_subprocess():
    t0 = time.time()
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    out = subprocess.run([sys.executable, "-c", PPO_STEP % (REPO, REPO)], env=env, capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK" in out.stdout
    assert time.time() - t0 < 30
