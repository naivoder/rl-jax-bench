"""Verification block (planned-check): the per-lane lr_scale flag equals training with that learning rate, and the
CRN arms share their control's lane key (common random numbers)."""
import os
import sys
import tempfile
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common  # noqa: F401,E402
import jax  # noqa: E402

from src.configs.config import Config  # noqa: E402
from src import train as TR  # noqa: E402


def _cfg(lr=3e-4):
    cfg = Config()
    return replace(cfg, env=replace(cfg.env, T=40), ppo=replace(cfg.ppo, lr=lr, n_steps=32, n_envs=2, n_epochs=2,
                                                                minibatch=32, curve_scenarios=2),
                   pre=replace(cfg.pre, ae_steps=20, bc_episodes=1, bc_epochs=1))


def test_lr_scale_equals_learning_rate():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        _, la, _, _ = TR.run_family(_cfg(), ["flat_lr1e-3", "flat"], [0], d1, total_steps=64 * 3, curve_every=64,
                                    ckpt_every_it=0)
        _, lb, _, _ = TR.run_family(_cfg(1e-3), ["flat"], [0], d2, total_steps=64 * 3, curve_every=64, ckpt_every_it=0)
    for x, y in zip(jax.tree.leaves(la.params), jax.tree.leaves(lb.params)):
        x, y = np.asarray(x), np.asarray(y)
        assert np.allclose(x[0], y[0], rtol=1e-4, atol=1e-6)        # lane 0 = flat_lr1e-3 == lr 1e-3
    moved = [np.max(np.abs(np.asarray(x)[0] - np.asarray(x)[1])) for x in jax.tree.leaves(la.params)]
    assert max(moved) > 1e-4                                          # lane 1 (lr 3e-4) differs


def test_crn_arms_share_the_control_key():
    cfg = Config()
    for arm, ctl in (("mhsa_lr1e-4", "mhsa"), ("mhsa_lr1e-3", "mhsa"), ("mhsa_lit_crn", "mhsa"),
                     ("mhsa_unmasked", "mhsa"), ("flat_lr1e-4", "flat"), ("flat_lr1e-3", "flat"), ("flat_single", "flat")):
        assert np.array_equal(np.asarray(TR.lane_key(cfg, arm, 2)), np.asarray(TR.lane_key(cfg, ctl, 2)))
