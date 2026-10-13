"""Smoke test (< 30 s on CPU): shapes + one PPO update for every network/arm family.

Run: uv run python -m pytest tests/test_smoke.py -q
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.train import ARMS, CONFIG_DIR, PPO, load_config  # noqa: E402

TINY = ["num_envs=2", "num_steps=8", "num_minibatches=2", "update_epochs=1", "total_timesteps=32", "hidden=8",
        "lstm_hidden=8"]


@pytest.mark.parametrize("config,arm", [
    ("classic_control.yaml", "separate"),
    ("classic_control.yaml", "shared"),
    ("classic_control.yaml", "trunc_bootstrap"),
    ("pendulum.yaml", "default"),
    ("pendulum.yaml", "gymnasium029"),
    ("lstm_masked_cartpole.yaml", "lstm"),
])
def test_one_update(config, arm):
    cfg = load_config(os.path.join(CONFIG_DIR, config), TINY)
    cfg.update(ARMS[arm])
    ppo = PPO(cfg)
    runner, consts = ppo.init(jnp.int32(1))
    d = ppo.spec.obs_dim
    assert runner.obs.shape == (2, d)
    vs, obs, done, carry, ic, traj = jax.jit(ppo.rollout)(runner, consts)
    assert traj.obs.shape == (8, 2, d) and traj.reward.shape == (8, 2) and traj.logp.shape == (8, 2)
    if ppo.cont:
        assert traj.action.shape == (8, 2, 1)
    new, m = jax.jit(ppo.update)(runner, consts)
    assert int(new.update) == 1
    for k, v in m.items():
        assert np.all(np.isfinite(np.asarray(v))), k
    assert float(m["ratio_dev_first_mb"]) < 1e-5  # debug Q-C: rho == 1 at the first minibatch
    moved = jax.tree.map(lambda a, b: float(jnp.abs(a - b).max()), runner.params, new.params)
    assert max(jax.tree.leaves(moved)) > 0
    # determinism of the key tree: same seed -> same first rollout
    r2, c2 = ppo.init(jnp.int32(1))
    np.testing.assert_array_equal(np.asarray(jax.jit(ppo.rollout)(r2, c2)[-1].action), np.asarray(traj.action))


def test_lstm_zero_train_breaks_ratio():
    """W12 power check: training from a zero state instead of the saved one must break rho == 1."""
    cfg = load_config(os.path.join(CONFIG_DIR, "lstm_masked_cartpole.yaml"), TINY)
    ppo = PPO(cfg)
    runner, consts = ppo.init(jnp.int32(1))
    upd = jax.jit(ppo.update)
    for _ in range(2):  # after one update the saved carry is non-zero
        runner, m = upd(runner, consts)
    # a saved state far from zero (as after many steps of a long episode) makes the effect visible on a tiny net
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    runner = runner._replace(carry=(jax.random.normal(k1, runner.carry[0].shape),
                                    jax.random.normal(k2, runner.carry[1].shape)))
    _, m = upd(runner, consts)
    assert float(m["ratio_dev_first_mb"]) < 1e-5
    cfg_b = dict(cfg, lstm_train_from_zero=True)
    ppo_b = PPO(cfg_b)
    _, mb = jax.jit(ppo_b.update)(runner, consts)
    assert float(mb["ratio_dev_first_mb"]) > 1e-4
