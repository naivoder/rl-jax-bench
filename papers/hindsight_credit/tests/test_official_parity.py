"""Parity with the official HCA snippet (github.com/hca-neurips2019/hca @ 3a564ad), from recorded fixtures.

    JAX_PLATFORMS=cpu uv run python -m pytest tests/test_official_parity.py -q

The fixtures were recorded by ``scripts/official_lockstep.py --fixtures``, which runs the UNMODIFIED official
``StateHCA.update`` / ``ReturnHCA.update`` in its own numpy venv ($OFFICIAL_ROOT);
none of the official code is copied here, only its outputs.

  official_grads.npz    per-episode (dlogits, dV, dlogits_h) for State and Return HCA on random tables, 8 sampled
                        episodes x 4 envs x gamma in {1, 0.9}. Tolerance 1e-10 (float64; observed <= 1e-14).
  official_lockstep.npz end-to-end learning: the official updates inside a separately written numpy env/sampler/
                        caller, fed the exact uniforms our sampler draws from the stored run keys. Per-episode
                        exact-value curves, returns and final policy/value tables. Tolerance 1e-9 (observed
                        <= 1e-12 over up to 500 episodes).
Rows of the terminal (reward-only) observations are excluded: the snippet treats the terminal step as an update
source, our code does not (A-TERMINAL); those rows are never read by the learner.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import src  # noqa: E402,F401  (bootstrap + x64 before jax)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import data as D  # noqa: E402
from src.algorithm import return_hca_grads, state_hca_grads  # noqa: E402
from src.model import Params  # noqa: E402
from src.train import load_config, make_env, make_runner, method_lrs, settings  # noqa: E402
from scripts.official_lockstep import LOCK_FIX, episode_draws  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
GRAD_ENVS = {"shortcut": lambda: D.shortcut(), "delayed_effect_n3_s2": lambda: D.delayed_effect(3, 2.0),
             "bandit_hidden": lambda: D.ambiguous_bandit(0.2, 0.5, True),
             "bandit_observed": lambda: D.ambiguous_bandit(0.1, 1.5, False)}


@pytest.fixture(scope="module")
def grads_fx():
    return np.load(os.path.join(FIX, "official_grads.npz"))


@pytest.fixture(scope="module")
def lock_fx():
    return np.load(os.path.join(FIX, "official_lockstep.npz"))


@pytest.mark.parametrize("name", list(GRAD_ENVS))
@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_per_episode_gradients_match_official(grads_fx, name, gamma):
    env = GRAD_ENVS[name]()
    tag = f"{name}_g{gamma}"
    g = lambda k: grads_fx[f"{tag}/{k}"]  # noqa: E731
    p = Params(jnp.asarray(g("theta")), jnp.asarray(g("psi")), jnp.asarray(g("phi")), jnp.asarray(g("V")),
               jnp.zeros_like(jnp.asarray(g("theta"))))
    eps = D.Episode(*(jnp.asarray(g(f"ep_{f}")) for f in D.Episode._fields))
    gs = jax.vmap(lambda e: state_hca_grads(p, e, None, gamma, "official"))(eps)
    gr = jax.vmap(lambda e: return_hca_grads(p, e, gamma, jnp.asarray(g("centres"))))(eps)
    dec = np.unique(env.obs[~env.terminal])
    for i in range(eps.obs.shape[0]):
        for kind, ours, hkey in (("state", gs, "psi"), ("return", gr, "phi")):
            np.testing.assert_allclose(np.asarray(ours.theta[i])[dec], g(f"off_ep{i}_{kind}_dlogits")[dec], atol=1e-10)
            np.testing.assert_allclose(np.asarray(ours.V[i])[dec], g(f"off_ep{i}_{kind}_dV")[dec], atol=1e-10)
            dh = np.transpose(np.asarray(getattr(ours, hkey)[i]), (2, 0, 1))[:, dec]
            np.testing.assert_allclose(dh, g(f"off_ep{i}_{kind}_dh")[:, dec], atol=1e-10)


def test_gradient_fixture_is_sensitive(grads_fx):
    """Mutant check: dropping the official j = i term (our paper-form credit without r-hat) must break parity."""
    env = GRAD_ENVS["shortcut"]()
    tag = "shortcut_g1.0"
    g = lambda k: grads_fx[f"{tag}/{k}"]  # noqa: E731
    p = Params(jnp.asarray(g("theta")), jnp.asarray(g("psi")), jnp.asarray(g("phi")), jnp.asarray(g("V")),
               jnp.zeros_like(jnp.asarray(g("theta"))))
    eps = D.Episode(*(jnp.asarray(g(f"ep_{f}")) for f in D.Episode._fields))
    gs = jax.vmap(lambda e: state_hca_grads(p, e, None, 1.0, "paper"))(eps)
    dec = np.unique(env.obs[~env.terminal])
    err = max(np.abs(np.asarray(gs.theta[i])[dec] - g(f"off_ep{i}_state_dlogits")[dec]).max()
              for i in range(eps.obs.shape[0]))
    assert err > 1e-3


@pytest.mark.parametrize("case", LOCK_FIX, ids=lambda c: f"{c[0]}_s{c[1]}_{c[2]}")
def test_lockstep_learning_matches_official(lock_fx, case):
    exp, si, method, E = case
    cfg = load_config()
    ec = cfg["experiments"][exp]
    env = make_env(ec, settings(ec)[si])
    tag = f"{exp}_s{si}_{method}"
    f = lambda k: lock_fx[f"{tag}/{k}"]  # noqa: E731
    keys = jnp.asarray(f("keys"), dtype=jnp.uint32)
    assert int(f("episodes")) == E
    # 1. our sampler's draws are the ones the official run consumed (recorded on GPU; CPU/GPU float64
    #    uniform/normal transforms can differ in the last ulp, hence 1e-14 rather than bit equality)
    for mine, theirs in zip(episode_draws(keys, E, env.L), ("u_env", "z", "u_act")):
        np.testing.assert_allclose(mine, f(theirs), rtol=0, atol=1e-14)
    # 2. same learning trajectory
    runner = make_runner(env, method, episodes=E, seg_len=None, n_bins=ec["n_bins"],
                         bin_widen=cfg["hca"]["bin_widen"], gamma=cfg["gamma"])
    res = runner(keys, method_lrs(cfg, method))
    dec = np.unique(env.obs[~env.terminal])
    np.testing.assert_allclose(np.asarray(res["curve"]), f("off_curve"), atol=1e-9)
    np.testing.assert_allclose(np.asarray(res["returns"]), f("off_returns"), atol=1e-9)
    np.testing.assert_allclose(np.asarray(res["params"].theta)[:, dec], f("off_theta")[:, dec], atol=1e-9)
    np.testing.assert_allclose(np.asarray(res["params"].V)[:, dec], f("off_V")[:, dec], atol=1e-9)
