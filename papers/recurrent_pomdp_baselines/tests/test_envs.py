"""Env differential tests vs the official code / gym sources (fixtures from scripts/official/dump_fixtures.py) and the
the README Pendulum identities.  gym computes in float64, we in float32: the one-step map is compared to 1e-5,
and multi-step trajectories (chaotic dynamics, float32 rounding amplifies) to a stated looser bound."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import common  # noqa: F401
from src.envs import Pendulum, SemiCircle, Wind, official_winds

Z = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "envs.npz"))


@pytest.mark.parametrize("ver", ["v0", "v1"])
def test_pendulum_one_step_map(ver):
    env = Pendulum(obs_dims=(0, 1, 2), version=ver)
    f = jax.jit(jax.vmap(env.physics))
    th, thd, r = f(jnp.asarray(Z[f"{ver}/th"], jnp.float32), jnp.asarray(Z[f"{ver}/thd"], jnp.float32),
                   jnp.asarray(Z[f"{ver}/u"], jnp.float32))
    assert np.max(np.abs(np.asarray(th) - Z[f"{ver}/nth"])) < 1e-5
    assert np.max(np.abs(np.asarray(thd) - Z[f"{ver}/nthd"])) < 1e-5
    assert np.max(np.abs(np.asarray(r) - Z[f"{ver}/r"]) / np.maximum(1, np.abs(Z[f"{ver}/r"]))) < 1e-5


@pytest.mark.parametrize("ver", ["v0", "v1"])
def test_pendulum_trajectories(ver):
    """10 official 200-step rollouts replayed through our env (agent action = torque / 2)."""
    env = Pendulum(obs_dims=(0, 1, 2), version=ver)
    errs = []
    for e in range(10):
        s0, acts = Z[f"{ver}/traj{e}/s0"], Z[f"{ver}/traj{e}/act"]
        st = {"th": jnp.float32(s0[0]), "thdot": jnp.float32(s0[1]), "t": jnp.int32(0)}

        def step(st, a):
            st, o, r, d, term = env.step(None, st, None, a / 2.0)
            return st, (o, r, d, term)
        _, (obs, rew, done, term) = jax.lax.scan(step, st, jnp.asarray(acts, jnp.float32))
        errs.append(max(np.max(np.abs(np.asarray(obs) - Z[f"{ver}/traj{e}/obs"][1:])),
                        np.max(np.abs(np.asarray(rew) - Z[f"{ver}/traj{e}/rew"]) / 16.3)))
        assert bool(done[-1]) and not bool(done[-2]) and float(jnp.max(term)) == 0.0  # time-limit, non-terminal
    # free-running float32 vs float64 over 200 chaotic steps: rounding is amplified on a few rollouts
    assert np.median(errs) < 1e-4 and max(errs) < 0.05, errs
    # re-synchronised every step from the official observation (theta = atan2(sin, cos); the dynamics and the cost
    # depend on theta only modulo 2 pi): every transition of every rollout must match to float32 precision
    for e in range(10):
        obs, acts, rews = Z[f"{ver}/traj{e}/obs"], Z[f"{ver}/traj{e}/act"], Z[f"{ver}/traj{e}/rew"]
        th = jnp.asarray(np.arctan2(obs[:-1, 1], obs[:-1, 0]), jnp.float32)
        nth, nthd, r = jax.vmap(env.physics)(th, jnp.asarray(obs[:-1, 2], jnp.float32),
                                             jnp.asarray(acts[:, 0], jnp.float32))
        o2 = np.stack([np.cos(np.asarray(nth)), np.sin(np.asarray(nth)), np.asarray(nthd)], -1)
        assert np.max(np.abs(o2 - obs[1:])) < 2e-5, e
        assert np.max(np.abs(np.asarray(r) - rews)) < 2e-5 * 16.3, e


def test_pendulum_identities():
    """the README: -P gives (cos, sin); theta-dot*dt = wrap(theta' - theta) and
    sin(theta) = ((thd' - thd)/0.05 - 3u)/15 whenever the speed clip does not bind (v0 order)."""
    env = Pendulum(obs_dims=(0, 1, 2), version="v0")
    key = jax.random.PRNGKey(0)
    st, o = env.reset(key, None)
    ths, thds, us = [float(st["th"])], [float(st["thdot"])], []
    for t in range(400):
        a = jax.random.uniform(jax.random.fold_in(key, t), (1,), minval=-1, maxval=1)
        st, o, r, d, _ = env.step(None, st, None, a)
        ths.append(float(st["th"])), thds.append(float(st["thdot"])), us.append(2.0 * float(a[0]))
        assert abs(float(o[0]) - np.cos(ths[-1])) < 1e-6 and abs(float(o[1]) - np.sin(ths[-1])) < 1e-6
    ths, thds, us = np.array(ths), np.array(thds), np.array(us)
    free = np.abs(thds[1:]) < 7.99
    dth = ths[1:] - ths[:-1]
    wrap = (dth + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(thds[1:][free] * 0.05 - wrap[free])) < 1e-5
    s_hat = ((thds[1:] - thds[:-1]) / 0.05 - 3 * us) / 15.0
    assert np.max(np.abs(s_hat[free] - np.sin(ths[:-1][free]))) < 2e-3   # float32 differencing amplifies by 1/(0.05*15)
    assert free.mean() > 0.5


def test_semicircle_vs_official():
    env = SemiCircle()
    nb, ngoal = 0, 0
    for e in range(4):
        goal = jnp.asarray(Z[f"pr{e}/goal"], jnp.float32)
        st = {"pos": jnp.asarray(Z[f"pr{e}/s0"], jnp.float32), "t": jnp.int32(0), "ep": jnp.int32(0)}
        obs0 = env._obs(st["pos"], goal, jnp.zeros((), bool))
        assert np.allclose(np.asarray(obs0), Z[f"pr{e}/obs"][0], atol=1e-6)
        for t in range(120):
            st, o, r, done, term = env.step(jax.random.PRNGKey(t), st, goal, jnp.asarray(Z[f"pr{e}/act"][t], jnp.float32))
            assert np.allclose(np.asarray(o), Z[f"pr{e}/obs"][t + 1], atol=1e-5), (e, t)
            assert float(r) == Z[f"pr{e}/rew"][t], (e, t)
            assert bool(done) == bool(Z[f"pr{e}/done"][t]), (e, t)
            if t == 59:  # inner boundary: the official env reset internally; inject its fresh start state
                nb += 1
                st = {**st, "pos": jnp.asarray(Z[f"pr{e}/state_after"][t], jnp.float32)}
                term = (jnp.linalg.norm(st["pos"] - goal) <= 0.2).astype(jnp.float32)
            assert np.allclose(np.asarray(st["pos"]), Z[f"pr{e}/state_after"][t], atol=1e-5), (e, t)
            assert float(term) == Z[f"pr{e}/term"][t], (e, t)
            ngoal += int(Z[f"pr{e}/rew"][t])
    assert nb == 4 and ngoal > 20   # the goal / terminal path was exercised


def test_wind_vs_official():
    w, perm = official_winds(60)
    assert np.array_equal(w, Z["wind/winds"].astype(np.float32)) and np.array_equal(perm, Z["wind/perm"])
    env = Wind()
    tasks = env.make_tasks(jax.random.PRNGKey(0))
    assert np.allclose(np.asarray(tasks["eval"]), Z["wind/winds"][Z["wind/perm"][-20:]], atol=1e-7)
    task = jnp.asarray(Z["wind/winds"][3], jnp.float32)
    st, o = env.reset(None, task)
    assert np.allclose(np.asarray(o), Z["wind/task3_obs"][0])
    for t in range(75):
        st, o, r, d, term = env.step(None, st, task, jnp.asarray(Z["wind/task3_act"][t], jnp.float32))
        assert np.allclose(np.asarray(o), Z["wind/task3_obs"][t + 1], atol=1e-5), t
        assert float(r) == Z["wind/task3_rew"][t], t
    # the official steering fixture never entered the 0.03 goal: check the reward/terminal path directly
    st = {"pos": jnp.array([0.0, 1.0 - 0.025], jnp.float32) - task, "t": jnp.int32(0)}
    st, o, r, d, term = env.step(None, st, task, jnp.zeros((2,), jnp.float32))   # action 0 -> move 0
    assert float(r) == 1.0 and float(term) == 1.0
