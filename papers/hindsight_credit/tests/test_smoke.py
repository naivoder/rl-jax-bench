"""Smoke test (<30 s on CPU): env shapes, one episode, one update per arm, tiny vmapped runs.

    JAX_PLATFORMS=cpu uv run python -m pytest tests/test_smoke.py -q
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import src  # noqa: E402,F401  (bootstrap + x64 before jax)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import data as D  # noqa: E402
from src.algorithm import episode_grads  # noqa: E402
from src.eval import start_value  # noqa: E402
from src.model import init_params, policy  # noqa: E402
from src.train import load_config, make_fig3r_runner, make_runner, method_lrs  # noqa: E402

ENVS = {"shortcut": D.shortcut(), "delayed_effect": D.delayed_effect(5, 1.0), "bandit_hidden": D.ambiguous_bandit(0.1, 1.5, True)}


@pytest.mark.parametrize("name", list(ENVS))
def test_env_and_episode(name):
    env = ENVS[name]
    S = env.P.shape[0]
    assert env.P.shape == (S, 2, S)
    np.testing.assert_allclose(env.P[~env.terminal].sum(-1), 1.0)
    ea = D.env_arrays(env)
    pi = policy(jnp.zeros((env.n_obs, 2)))
    ep = D.sample_episode(ea, env.s0, env.L, lambda o: pi[o], jax.random.PRNGKey(0))
    assert ep.obs.shape == (env.L,)
    v = np.asarray(ep.valid)
    assert v[0] and np.all(np.diff(v.astype(int)) <= 0)               # valid is a prefix
    T = v.sum()
    assert env.terminal[int(ep.state[T - 1])] and not ep.decision[T - 1]  # ends on a terminal reward-only step
    assert np.all(np.asarray(ep.decision[: T - 1]))


@pytest.mark.parametrize("method", ["state_hca", "return_hca", "pg"])
def test_one_update_every_arm(method):
    env = ENVS["delayed_effect"]
    p = init_params(env.n_obs, 3)
    ea = D.env_arrays(env)
    pi = policy(p.theta)
    ep = D.sample_episode(ea, env.s0, env.L, lambda o: pi[o], jax.random.PRNGKey(1))
    g = episode_grads(method, p, ep, seg_len=3, gamma=1.0, centres=jnp.linspace(-3, 3, 3))
    for a, b in zip(g, p):
        assert a.shape == b.shape and np.all(np.isfinite(np.asarray(a)))
    np.testing.assert_allclose(np.asarray(g.theta).sum(-1), 0.0, atol=1e-12)  # softmax logit grads sum to 0
    # at zero init h = pi, so both HCA logit updates are exactly 0 (no credit before h is learned)


def test_tiny_runs_and_fig3r():
    cfg = load_config()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    env = ENVS["shortcut"]
    for m in ("state_hca", "return_hca", "pg"):
        run = make_runner(env, m, episodes=3, seg_len=None, n_bins=10, bin_widen=2.0, gamma=1.0)
        res = run(keys, method_lrs(cfg, m, 0.2))
        assert res["curve"].shape == (2, 4)
        v0 = float(start_value(D.env_arrays(env), jnp.zeros((env.n_obs, 2)), 0, env.L))
        np.testing.assert_allclose(np.asarray(res["curve"][:, 0]), v0)
        assert np.all(np.isfinite(np.asarray(res["curve"])))
    run, oracle = make_fig3r_runner(env, "state_hca", 0.5, rollouts=5, n_bins=10, bin_widen=2.0, gamma=1.0,
                                    immediate="rhat", hca_lrs=method_lrs(cfg, "state_hca"))
    out = run(keys)
    assert out["online"].shape == (2,) and abs(oracle - 1.7261888) < 1e-6


def test_sensitivity_options_run():
    """Running-range return bins, sliding n-step windows and the online-V Fig. 3-right PG baseline (sensitivities)."""
    cfg = load_config()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    env = ENVS["delayed_effect"]
    run = make_runner(env, "return_hca", episodes=5, seg_len=None, n_bins=3, bin_widen=2.0, gamma=1.0,
                      bin_mode="running")
    assert np.all(np.isfinite(np.asarray(run(keys, method_lrs(cfg, "return_hca"))["curve"])))
    for m in ("state_hca", "pg"):
        run = make_runner(env, m, episodes=5, seg_len=3, n_bins=3, bin_widen=2.0, gamma=1.0, seg_mode="sliding")
        assert np.all(np.isfinite(np.asarray(run(keys, method_lrs(cfg, m, 0.2))["curve"])))
    run, _ = make_fig3r_runner(ENVS["shortcut"], "pg", 0.5, rollouts=5, n_bins=10, bin_widen=2.0, gamma=1.0,
                               immediate="rhat", hca_lrs=method_lrs(cfg, "state_hca"), pg_baseline="online_v")
    assert np.all(np.isfinite(np.asarray(run(keys)["online"])))
