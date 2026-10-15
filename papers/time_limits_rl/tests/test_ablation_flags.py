"""full-scale ablation switches (src/train.py): defaults leave the trainer bit-identical; each switch changes only what
it names.

* tau_obs_norm (Tonic MeanStd also z-scores the time feature): a time-UNAWARE lane (tau = 0) is unchanged exactly,
  a time-aware lane changes.
* adv_norm_scope = batch (Tonic: one normalisation per segment) and anneal_lr (Baselines ppo1 schedule='linear'):
  the first update with anneal_lr equals the constant-lr update (lr_0 = lr0); later ones differ.
Run: JAX_PLATFORMS=cpu uv run python -m pytest tests/test_ablation_flags.py -q
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.train import PPO, lanes_from_cfg, load_config  # noqa: E402

BASE = ["num_steps=64", "num_minibatches=2", "update_epochs=2", "total_timesteps=192", "seeds=[0]", "hidden=16"]


def run(extra):
    cfg = load_config(os.path.join(REPO, "src", "configs", "e4_inverted_pendulum.yaml"), BASE + extra)
    cfg["arms"] = {"ppo": {"time_mode": "none", "T": 20, "gamma": 1.0},
                   "ta": {"time_mode": "elapsed_pm1", "T": 20, "gamma": 1.0}}
    ppo = PPO(cfg)
    names, lane = lanes_from_cfg(cfg)
    r, k = jax.jit(jax.vmap(ppo.init))(lane)
    r, m = jax.jit(jax.vmap(lambda rr, kk, ll: ppo.run_chunk(rr, kk, ll, 3)))(r, k, lane)
    return names, np.asarray(m["loss"]), jax.tree.leaves(r.params)


@pytest.fixture(scope="module")
def base():
    return run([])


def test_tau_obs_norm_leaves_time_unaware_lane_unchanged(base):
    _, l0, p0 = base
    _, l1, p1 = run(["tau_obs_norm=true"])
    np.testing.assert_array_equal(l0[0], l1[0])                    # lane 0 = time-unaware
    assert all(np.array_equal(np.asarray(a)[0], np.asarray(b)[0]) for a, b in zip(p0, p1))
    assert not np.array_equal(l0[1], l1[1])                        # lane 1 = time-aware: changed


def test_anneal_first_update_equals_constant_then_differs(base):
    _, l0, _ = base
    _, l1, _ = run(["anneal_lr=true"])
    # equal in the trainer's float32 mode; under jax_enable_x64 (set by test_grads.py when run in the same session)
    # lr_at's float32 schedule value is cast to a float64 hyperparameter, so allow round-off
    np.testing.assert_allclose(l0[:, 0], l1[:, 0], rtol=1e-5, atol=1e-7)
    assert not np.array_equal(l0[:, 1:], l1[:, 1:])


def test_batch_adv_norm_changes_update(base):
    _, l0, _ = base
    _, l1, _ = run(["adv_norm_scope=batch"])
    assert np.all(np.isfinite(l1)) and not np.array_equal(l0, l1)
