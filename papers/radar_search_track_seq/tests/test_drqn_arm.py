"""The descriptive DRQN arm (src/drqn_arm.py, rl_jax_bench.drqn core) runs end to end on a tiny config:
observation flatten/unflatten round-trips, transitions are stored, gradient steps start after learning_starts and the
target network syncs."""
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common  # noqa: F401,E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.configs.config import Config  # noqa: E402
from src import data, drqn_arm as DA  # noqa: E402


def test_flatten_roundtrip():
    cfg = Config().env
    s = data.reset(jax.random.PRNGKey(0), cfg)
    s, o, _ = data.step(s, jnp.array([3, 4], jnp.int32), cfg)
    o2 = DA.unflatten_obs(DA.flatten_obs(o), cfg)
    for a, b in zip(o, o2):
        np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))


def test_drqn_arm_runs():
    cfg = Config()
    cfg = replace(cfg, env=replace(cfg.env, T=20, n_sv=16, k_max=6))
    st = DA.DRQNSettings(batch=8, buffer=256, n_envs=2, target_period=5, learning_starts=16)
    net, init, run_chunk, n_iter = DA.make(cfg, st, "mhsa", 120)
    k = DA.lane_key(cfg, "drqn_mhsa", 0)
    carry = jax.jit(init)(k)
    carry, info = jax.jit(lambda c: run_chunk(c, 0, n_iter, k))(carry)
    qs, buf = carry[0], carry[1]
    assert int(buf.count) == 120 and int(qs.n_updates) == n_iter - 16 // 2 + 1
    assert np.all(np.isfinite(np.asarray(info["loss"])))
    assert not np.allclose(np.asarray(jax.tree.leaves(qs.params)[0]), np.asarray(jax.tree.leaves(qs.target_params)[0])) \
        or int(qs.n_updates) % 5 == 0
