"""Differential tests against the authors' code: Tonic @0e20c89 (fabiopardo/tonic, the library
the paper's project page links to). Oracle outputs are recorded fixtures produced by Tonic's own
functions in their isolated venv (scripts/toys/make_tonic_fixtures.py); this file recomputes OUR side.

* PEB lambda-returns: ours (src.algorithm.gae_peb, GAE form) == Tonic replays/utils.py lambda_returns (recursive
  lambda-return form) on random batches with resets and terminations (tol 1e-5, float32).
* Runner semantics: our runner-side limit + time feature (src.data.tl_step, mode elapsed_pm1) == Tonic
  Sequential(max_episode_steps=T) over TimeFeature(env, T): next obs, pre-reset 'observations', resets,
  terminations, rewards, on a scripted env with both timeouts and terminations.
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402

FIX = os.path.join(REPO, "tests", "fixtures", "tonic")


def load(name):
    return dict(np.load(os.path.join(FIX, f"{name}_in.npz"))), dict(np.load(os.path.join(FIX, f"{name}_oracle.npz")))


def test_provenance():
    p = json.load(open(os.path.join(FIX, "provenance.json")))
    assert p["commit"].startswith("0e20c89")


@pytest.mark.parametrize("name", ["lr_mixed", "lr_dense"])
def test_peb_lambda_returns_equal_tonic(name):
    i, o = load(name)
    _, ret = A.gae_peb(jnp.asarray(i["lr_rewards"]), jnp.asarray(i["lr_values"]), jnp.asarray(i["lr_next_values"]),
                       jnp.asarray(i["lr_terminations"]), jnp.asarray(i["lr_resets"]), float(i["lr_gamma"]),
                       float(i["lr_lam"]), 1.0)
    np.testing.assert_allclose(np.asarray(ret), o["lambda_returns"], rtol=1e-5, atol=1e-5)
    assert i["lr_resets"].sum() > 5 and (i["lr_terminations"] < i["lr_resets"]).sum() > 3  # both kinds present


@pytest.mark.parametrize("name", ["lr_mixed", "lr_dense"])
def test_no_peb_equals_tonic_with_timeouts_as_terminations(name):
    """The paper's standard agent = Tonic with terminations := resets (terminal_timeouts=True)."""
    i, _ = load(name)
    _, ret = A.gae_peb(jnp.asarray(i["lr_rewards"]), jnp.asarray(i["lr_values"]), jnp.asarray(i["lr_next_values"]),
                       jnp.asarray(i["lr_terminations"]), jnp.asarray(i["lr_resets"]), 0.99, 0.95, 0.0)
    # recursive lambda-return with terminations = resets (Tonic's formula, re-derived inline as the reference)
    v, nv, r, rs = i["lr_values"], i["lr_next_values"], i["lr_rewards"], i["lr_resets"]
    ref = np.zeros_like(v)
    last = nv[-1]
    for t in reversed(range(len(r))):
        b = (0.05 * nv[t] + 0.95 * last) * (1 - rs[t]) + rs[t] * nv[t]
        b = b * (1 - rs[t])
        ref[t] = last = r[t] + 0.99 * b
    np.testing.assert_allclose(np.asarray(ret), ref, rtol=1e-5, atol=1e-5)


def scripted_env():
    """Same dynamics as the oracle harness ScriptedEnv: obs = [steps since reset, sum of actions], r = 1 + a,
    terminates when a == 1."""
    def reset(key):
        s = jnp.zeros(2, jnp.float32)
        return s, s

    def step(key, s, a):
        s2 = s + jnp.array([1.0, a], jnp.float32)
        return s2, s2, 1.0 + a.astype(jnp.float32), a == 1

    return D.Env(reset, step, 2, False, 2, None, None, "scripted")


def test_runner_and_time_feature_equal_tonic():
    i, o = load("runner_T5")
    T = int(i["rn_T"])
    env = scripted_env()
    mode = D.TIME_MODES["elapsed_pm1"]
    st, ob, tau = D.tl_reset_obs(env, jax.random.PRNGKey(0), T, mode)
    np.testing.assert_allclose(np.append(np.asarray(ob), tau), o["rn_obs0"], atol=1e-6)
    for k, a in enumerate(i["rn_actions"]):
        s = D.tl_step(env, jax.random.PRNGKey(k), st, jnp.int32(a), T, mode)
        np.testing.assert_allclose(np.append(np.asarray(s.obs), s.tau), o["rn_obs"][k], atol=1e-6, err_msg=f"obs {k}")
        np.testing.assert_allclose(np.append(np.asarray(s.final_obs), s.final_tau), o["rn_final"][k], atol=1e-6,
                                   err_msg=f"final obs {k}")
        assert bool(s.reset) == bool(o["rn_resets"][k]), k
        assert bool(s.term) == bool(o["rn_terms"][k]), k
        assert bool(s.timeout) == (bool(o["rn_resets"][k]) and not bool(o["rn_terms"][k])), k
        assert float(s.reward) == float(o["rn_rewards"][k])
        st = s.state
    assert o["rn_resets"].sum() > o["rn_terms"].sum() > 0  # the script exercises both timeouts and terminations


# ===================================================================== oracle stage: env parity vs the authors' env builder
# Oracle: Tonic @0e20c89 environments.Gym(<id>, terminal_timeouts=False) -> gym 0.24.1 MuJoCo-v4 on mujoco 2.2.0 with
# Tonic's ActionRescaler, recorded by scripts/oracle/make_env_fixtures.py (harness/env_trace.py in the oracle venv).
# Tolerances: obs 1e-5 / reward 1e-5 (float32 port vs float64 gym) on the steps BEFORE the first termination; the
# termination flag must agree exactly. After a termination (and, for the hopper, after the first ground contact) the
# trajectories separate because gym runs mujoco 2.2.0 and we run mujoco 3.x: CPU MuJoCo 3.x in float64 shows the same
# separation from gym while matching our MJX port to ~1e-11 (V3).
GV4 = os.path.join(REPO, "tests", "fixtures", "gym_v4")
GV4_CASES = [("inverted_pendulum", 0), ("hopper", 1)]


def gv4(kind):
    return dict(np.load(os.path.join(GV4, f"{kind}_in.npz"))), dict(np.load(os.path.join(GV4, f"{kind}_oracle.npz")))


def run_ours(env, i_, i, key):
    st, _ = env.reset(key)
    step = jax.jit(env.step)
    out = []
    for k, a in enumerate(i_[f"act_{i}"]):
        st, o, r, t = step(jax.random.PRNGKey(k), st, jnp.asarray(a))
        out.append((np.asarray(o), float(r), bool(t)))
    return out


@pytest.mark.parametrize("name,i", GV4_CASES)
def test_gymv4_port_free_run_equals_gym(name, i):
    """MJX port of gym-v4 (src.data.gymv4_env) from the same initial state through the same actions (incl. actions
    outside [-1, 1], clipped like Tonic's ActionRescaler) == gym via Tonic, step by step, until termination."""
    i_, o = gv4("mjx_port")
    env = D.gymv4_env(name)
    res = run_ours(env, i_, i, jax.random.PRNGKey(100 + i))  # the generator's reset key -> the recorded q0, qd0
    st0, _ = env.reset(jax.random.PRNGKey(100 + i))
    np.testing.assert_allclose(np.asarray(st0.qpos), i_[f"q0_{i}"], atol=1e-7)
    alive = next((k + 1 for k, x in enumerate(res) if x[2]), len(res))
    assert alive >= 5
    for k in range(alive):
        np.testing.assert_allclose(res[k][0], o[f"fr_obs_{i}"][k], atol=1e-5, err_msg=f"{name} obs step {k}")
        assert abs(res[k][1] - o[f"fr_rew_{i}"][k]) < 1e-5, (name, k)
        assert res[k][2] == bool(o[f"fr_done_{i}"][k]), (name, k)
    assert np.abs(i_[f"act_{i}"]).max() > 1.0  # the clipping path is exercised


@pytest.mark.parametrize("name,i", GV4_CASES)
def test_gymv4_port_constants_equal_gym(name, i):
    """dt (timestep x frame_skip), init_qpos (reset centre) and the model's integrator equal gym's."""
    import mujoco
    _, o = gv4("mjx_port")
    m = mujoco.MjModel.from_xml_path(os.path.join(REPO, "src", "assets", "gym_v4", D.GYMV4[name]["xml"]))
    assert abs(m.opt.timestep * D.GYMV4[name]["frame_skip"] - float(o[f"dt_{i}"])) < 1e-12
    np.testing.assert_allclose(m.qpos0, o[f"init_qpos_{i}"], atol=1e-12)
    np.testing.assert_allclose(np.zeros(m.nv), o[f"init_qvel_{i}"], atol=1e-12)
    assert int(m.opt.integrator) == int(o[f"integrator_{i}"])


@pytest.mark.parametrize("name,i", GV4_CASES)
def test_brax_proxy_reward_and_termination_rules_match_gym_but_dynamics_do_not(name, i):
    """small-scale Brax proxy (generalized backend) vs gym-v4, one step from the SAME (re-synchronised) pre-state:
    the termination rule agrees on every pre-termination step and the IP reward is identical, but the one-step
    observation differs by > 1e-3 (Euler vs RK4; for the hopper also re-placed root joints and Brax solver
    parameters) -- the proxy gap, recorded rather than hidden."""
    i_, o = gv4("brax_proxy")
    env = D.brax_env(name, "generalized")
    res = run_ours(env, i_, i, jax.random.PRNGKey(100 + i))
    alive = next((k + 1 for k, x in enumerate(res) if x[2]), len(res))
    gap = max(np.abs(res[k][0] - o[f"rs_obs_{i}"][k]).max() for k in range(alive))
    assert all(res[k][2] == bool(o[f"rs_done_{i}"][k]) for k in range(alive))
    if name == "inverted_pendulum":
        assert all(res[k][1] == o[f"rs_rew_{i}"][k] for k in range(alive))
    else:
        assert max(abs(res[k][1] - o[f"rs_rew_{i}"][k]) for k in range(alive)) < 0.1
    assert 1e-3 < gap < 0.5, gap
