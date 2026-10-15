"""Recheck fixes ; every switch defaults to the original behavior.

* action_interface = raw: the paper-era Baselines + gym path. On the MJX IP port it must
  equal gym 0.24.1 InvertedPendulum-v4 driven by RAW actions (fixture recorded in the official-code venv,
  runs/recheck/official-oracle/scripts/p1_gym_ref.py), clamp at the ctrlrange, and charge the hopper control cost
  on the raw action.
* reward_scale_reset_on_done = false: Baselines VecNormalize @9fa8e1b never zeroes `ret`.
* stop_rule = ppo1_completed_steps: Baselines ppo1 iterates until the completed-episode steps reach the budget,
  with lrmult = 1 - timesteps_so_far / max_timesteps.
Run: JAX_PLATFORMS=cpu uv run python -m pytest tests/test_recheck_fixes.py -q
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import data as D  # noqa: E402
from src.ppo8 import data as P8D  # noqa: E402
from src.train import PPO, lanes_from_cfg, load_config  # noqa: E402

FIX = os.path.join(REPO, "tests", "fixtures", "gym_v4", "ip_v4_rawact_gym_ref.npz")


def _ip_one_step(env):
    d0, _ = env.reset(jax.random.PRNGKey(0))

    @jax.jit
    def one(q, v, a):
        d = d0.replace(qpos=q.astype(d0.qpos.dtype), qvel=v.astype(d0.qvel.dtype))
        _, o, r, term = env.step(jax.random.PRNGKey(0), d, a)
        return o, r, term
    return one


def test_gymv4_ip_raw_interface_matches_gym_raw_actions():
    z = np.load(FIX)
    env = D.gymv4_env("inverted_pendulum", "raw")
    np.testing.assert_array_equal(np.asarray(env.act_low), [-3.0])
    np.testing.assert_array_equal(np.asarray(env.act_high), [3.0])
    one = _ip_one_step(env)
    worst, n_clamped = 0.0, 0
    for i in range(3):
        A, O, Dn = z[f"A{i}"], z[f"O{i}"], z[f"D{i}"]
        prev = np.concatenate([z[f"q0{i}"], z[f"v0{i}"]])
        for t in range(min(len(A), 300)):
            o, r, term = one(jnp.asarray(prev[:2]), jnp.asarray(prev[2:]), jnp.asarray(A[t], jnp.float32))
            worst = max(worst, float(np.abs(np.asarray(o) - O[t]).max()))
            assert bool(term) == bool(Dn[t]) and float(r) == 1.0
            n_clamped += int(np.abs(A[t][0]) > 3.0)
            prev = O[t]
    assert worst < 1e-4, worst          # float32 MJX vs float64 gym, one-step resynchronised
    assert n_clamped > 0                # the fixture exercises MuJoCo's ctrlrange clamp


def test_gymv4_ip_raw_equals_rescaled_input_and_clamps():
    raw, res = D.gymv4_env("inverted_pendulum", "raw"), D.gymv4_env("inverted_pendulum")
    assert res.act_low == -1.0 and res.act_high == 1.0     # default interface unchanged
    o_raw, o_res = _ip_one_step(raw), _ip_one_step(res)
    q, v = jnp.asarray([0.01, -0.02]), jnp.asarray([0.1, 0.05])
    for a in (-2.5, -0.7, 0.3, 1.9):
        np.testing.assert_allclose(np.asarray(o_raw(q, v, jnp.asarray([a], jnp.float32))[0]),
                                   np.asarray(o_res(q, v, jnp.asarray([a / 3.0], jnp.float32))[0]), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(o_raw(q, v, jnp.asarray([6.0]))[0]),
                                  np.asarray(o_raw(q, v, jnp.asarray([3.0]))[0]))


def test_gymv4_hopper_raw_cost_on_raw_action():
    env = D.gymv4_env("hopper", "raw")
    d, _ = env.reset(jax.random.PRNGKey(1))
    step = jax.jit(lambda a: env.step(jax.random.PRNGKey(0), d, a))
    _, o1, r1, _ = step(jnp.ones(3, jnp.float32))
    _, o2, r2, _ = step(2.0 * jnp.ones(3, jnp.float32))
    np.testing.assert_array_equal(np.asarray(o1), np.asarray(o2))           # clamped to ctrlrange [-1, 1]
    np.testing.assert_allclose(float(r1) - float(r2), 1e-3 * 3 * (4.0 - 1.0), rtol=1e-3)  # gym: cost on raw a


def test_brax_ip_raw_interface():
    raw, res = D.brax_env("inverted_pendulum", action_interface="raw"), D.brax_env("inverted_pendulum")
    s0, _ = raw.reset(jax.random.PRNGKey(0))
    f_raw = jax.jit(lambda a: raw.step(None, s0, a)[1])
    f_res = jax.jit(lambda a: res.step(None, s0, a)[1])
    for a in (-2.0, 0.5):
        np.testing.assert_allclose(np.asarray(f_raw(jnp.asarray([a]))), np.asarray(f_res(jnp.asarray([a / 3.0]))),
                                   atol=1e-5)
    np.testing.assert_allclose(np.asarray(f_raw(jnp.asarray([7.0]))), np.asarray(f_raw(jnp.asarray([3.0]))),
                               atol=1e-6)


def test_reward_scaler_paper_era_never_resets():
    rms = P8D.rms_init(())
    ret = jnp.asarray([5.0])
    r, d = jnp.asarray([1.0]), jnp.asarray([True])
    a_ret, a_rms, a_rew = P8D.scale_reward_baselines(ret, rms, r, d, 0.99, 1e-8, 10.0)
    b_ret, b_rms, b_rew = D.scale_reward_baselines_2018(ret, rms, r, 0.99, 1e-8, 10.0)
    np.testing.assert_array_equal(np.asarray(a_ret), [0.0])                 # post-2018-09 VecNormalize
    np.testing.assert_allclose(np.asarray(b_ret), [5.0 * 0.99 + 1.0])       # 9fa8e1b: never reset
    np.testing.assert_array_equal(np.asarray(a_rew), np.asarray(b_rew))     # same step: same RMS and scaled reward
    np.testing.assert_array_equal(np.asarray(a_rms.var), np.asarray(b_rms.var))


BASE = ["num_steps=64", "num_minibatches=2", "update_epochs=2", "seeds=[0]", "hidden=16"]


def _cfg(extra, T=20, total=192):
    cfg = load_config(os.path.join(REPO, "src", "configs", "e4_inverted_pendulum.yaml"),
                      BASE + [f"total_timesteps={total}"] + extra)
    cfg["arms"] = {"ppo": {"time_mode": "none", "T": T, "gamma": 1.0},
                   "ta": {"time_mode": "remaining_pm1", "T": T, "gamma": 1.0}}
    return cfg


def _run(cfg, n=None):
    ppo = PPO(cfg)
    names, lane = lanes_from_cfg(cfg)
    r0, k = jax.jit(jax.vmap(ppo.init))(lane)
    r, m = jax.jit(jax.vmap(lambda rr, kk, ll: ppo.run_chunk(rr, kk, ll, n or ppo.U)))(r0, k, lane)
    return ppo, r0, r, jax.tree.map(np.asarray, m)


def test_vecnorm_noreset_changes_training_only_through_the_scaler():
    _, _, ra, ma = _run(_cfg([]), 3)
    _, _, rb, mb = _run(_cfg(["reward_scale_reset_on_done=false"]), 3)
    _, _, rc, mc = _run(_cfg(["reward_scale_reset_on_done=true"]), 3)
    np.testing.assert_array_equal(ma["loss"], mc["loss"])                   # explicit true == default
    assert not np.array_equal(ma["ret_scale"], mb["ret_scale"])             # episodes end within 192 steps


def test_ppo1_stop_rule_runs_until_completed_steps_reach_budget():
    total, T, M = 192, 20, 64
    ppo, r0, r, m = _run(_cfg(["stop_rule=ppo1_completed_steps", "anneal_lr=true", "anneal_clip=true"], T, total))
    assert ppo.U == -(-(total + T - 1) // M)
    act = m["active"] > 0.5                                                 # (lanes, U)
    ts = m["ts_done"]
    for i in range(act.shape[0]):
        n = int(act[i].sum())
        assert act[i, :n].all() and not act[i, n:].any() and n >= 1
        assert ts[i, n - 1] >= total                                        # stopped because the budget was met
        assert n == 1 or ts[i, n - 2] < total                               # ... and not before
        assert np.all(m["ep_cnt"][i, n:] == 0)
    # the first update uses lrmult = 1 (ts = 0): identical to the 'updates' rule's first update
    _, _, _, m0 = _run(_cfg(["anneal_lr=true", "anneal_clip=true"], T, total), 1)
    np.testing.assert_allclose(m["loss"][:, 0], m0["loss"][:, 0], rtol=1e-5, atol=1e-7)


def test_ppo1_numerics_init_normalizer_and_batch_statistics():
    cfg = _cfg(["ppo1_numerics=true", "tau_obs_norm=true", "reward_scaling=false", "obs_clip=5.0",
                "clip_vloss=false", "adv_norm_scope=batch"], 20, 128)
    ppo, r0, r, m = _run(cfg, 2)
    for net in ("actor", "critic"):
        k = np.asarray(r0.params["params"][net]["l0"]["kernel"])          # (lanes, d + 1, h); lane 0 = time-unaware
        np.testing.assert_array_equal(k[0, -1], 0.0)                       # ppo1 unaware net has no tau input
        np.testing.assert_allclose(np.sqrt((k ** 2).sum(1)), 1.0, rtol=1e-5)  # normc(1.0) columns
    head = np.asarray(r0.params["params"]["actor_head"]["kernel"])
    np.testing.assert_allclose(np.sqrt((head ** 2).sum(1)), 0.01, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(r0.obs_rms.var), 1.0)
    np.testing.assert_allclose(np.asarray(r0.obs_rms.count), 1e-2, rtol=1e-6)
    # one batch update per iteration: count = 1e-2 + 64 per update
    np.testing.assert_allclose(np.asarray(r.obs_rms.count), 1e-2 + 2 * 64, rtol=1e-6)
    assert np.all(np.asarray(m["ratio_dev_first_mb"]) < 1e-5)              # old log-probs recomputed with new stats
    assert np.all(np.isfinite(m["loss"]))
