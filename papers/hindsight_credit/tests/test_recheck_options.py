"""Recheck runner options (2026-10-11): defaults are the pre-recheck primary bit for bit; the residual h prior
reduces to the primary when the policy never moves; theta0 reproduces the -1.819 E1 start.

    uv run python -m pytest tests/test_recheck_options.py -q
"""
import os

import src  # noqa: F401  (bootstrap + x64 before jax)
import jax.numpy as jnp
import numpy as np
import pytest

from src.train import load_config, make_env, make_runner, method_lrs, run_keys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_config()


def _runner(exp, method, **kw):
    ec = CFG["experiments"][exp]
    env = make_env(ec, ec.get("env_kwargs", {}) if "sweep" not in ec else {**ec["env_kwargs"], **{k: v[0] for k, v in ec["sweep"].items()}})
    return env, make_runner(env, method, episodes=ec["episodes"], seg_len=ec["seg_len"], n_bins=ec["n_bins"],
                            bin_widen=CFG["hca"]["bin_widen"], gamma=CFG["gamma"], **kw)


@pytest.mark.parametrize("exp,method", [("E6", "state_hca"), ("E6", "return_hca"), ("E1", "state_hca")])
def test_defaults_reproduce_stored_seed0(exp, method):
    path = os.path.join(ROOT, "runs", "seed_0", "curves.npz")
    if not os.path.exists(path):
        pytest.skip("needs runs/seed_0/curves.npz: run `uv run python -m scripts.full tune` then "
                    "`uv run python -m scripts.full eval --seed 0` (about 1.5 min on a GPU)")
    z = np.load(path)
    _, run = _runner(exp, method)
    c = np.asarray(run(run_keys(0, exp, 0, 100), method_lrs(CFG, method))["curve"])
    np.testing.assert_array_equal(c.astype(np.float32), z[f"{exp}_s0_{method}"])  # stored as float32


@pytest.mark.parametrize("exp,method", [("E6", "state_hca"), ("E6", "return_hca"), ("E3", "state_hca")])
def test_residual_equals_logit_when_policy_frozen(exp, method):
    lrs = method_lrs(CFG, method).at[0].set(0.0)
    keys = run_keys(5, exp, 0, 8)
    _, a = _runner(exp, method)
    _, b = _runner(exp, method, h_param="residual")
    pa, pb = a(keys, lrs)["params"], b(keys, lrs)["params"]
    np.testing.assert_allclose(np.asarray(pa.psi), np.asarray(pb.psi), atol=1e-12)
    np.testing.assert_allclose(np.asarray(pa.phi), np.asarray(pb.phi), atol=1e-12)
    np.testing.assert_allclose(np.asarray(pa.V), np.asarray(pb.V), atol=1e-12)


def test_counts_and_ema_run_and_differ():
    keys = run_keys(5, "E6", 0, 8)
    lrs = method_lrs(CFG, "state_hca")
    _, p = _runner("E6", "state_hca")
    _, c = _runner("E6", "state_hca", h_est="counts")
    _, e = _runner("E6", "state_hca", h_est="ema")
    cp, cc, ce = (np.asarray(r(keys, lrs)["curve"]) for r in (p, c, e))
    assert np.all(np.isfinite(cc)) and np.all(np.isfinite(ce))
    assert not np.allclose(cp, cc) and not np.allclose(cp, ce)
    np.testing.assert_array_equal(cp[:, :2], cc[:, :2])  # first update: uniform h in every form -> same credit


def test_theta0_e1_start():
    env, _ = _runner("E1", "pg")
    th = np.zeros((env.n_obs, 2)); th[:5, 1] = 1.0
    _, run = _runner("E1", "pg", theta0=tuple(map(tuple, th)))
    c = np.asarray(run(run_keys(0, "E1", 0, 4), method_lrs(CFG, "pg", 0.1))["curve"])
    np.testing.assert_allclose(c[:, 0], -1.8190, atol=5e-4)
