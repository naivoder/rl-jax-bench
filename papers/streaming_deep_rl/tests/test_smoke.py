"""Smoke (< 30 s on CPU): shapes + one step of every agent and the vmapped multi-arm program."""
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import agents as ag  # noqa: E402
from src import data as dt  # noqa: E402
from src import train as tr  # noqa: E402

T0 = [None]   # set by the first smoke test (collection of other modules must not count)


def test_q_agent_two_arms_two_steps():
    T0[0] = time.time()
    env = dt.minatar_env("Breakout")
    assert env.obs_shape == (10, 10, 4) and env.n_actions == 3
    agent = ag.QAgent(env, 1000, overshoot_diag=False)
    lane = tr.stack_lanes("q", [tr.lane_cfg("q", "stream_q"), tr.lane_cfg("q", "classic_q")])
    keys = jnp.stack([jax.random.PRNGKey(1)] * 2)
    c = jax.vmap(agent.init)(keys, lane)
    assert c.obs.shape == (2, 10, 10, 4)
    step = jax.jit(jax.vmap(agent.step, in_axes=(0, None, 0, 0)))
    for t in range(2):
        c, (ret, ln) = step(c, jnp.int32(t), lane, keys)
    leaves = jax.tree.leaves(c.params)
    assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
    assert float(c.diag.max_viol[0]) <= 1e-5   # P1 on the ObGD lane


def test_ac_and_td_learn_one_step():
    env = dt.Env(None, None, (11,), 3, False, "fake")
    agent = ag.ACAgent(env)
    lane = jax.tree.map(lambda x: x[0], tr.stack_lanes("ac", [tr.lane_cfg("ac", "stream_ac")]))
    c = None
    p_pi = agent.t_pi
    import src.model as mdl
    p_pi = mdl.init_params(jax.random.PRNGKey(0), agent.t_pi, 0)
    p_v = mdl.init_params(jax.random.PRNGKey(1), agent.t_v, 0)
    z = jax.tree.map(jnp.zeros_like, p_pi), jax.tree.map(jnp.zeros_like, p_v)
    s = jnp.ones(12)
    out = agent.learn(p_pi, p_v, z[0], z[1], ag.alg.adam_init(p_pi), ag.alg.adam_init(p_v), s,
                              jnp.zeros(3), jnp.float32(1.0), s * 0.5, jnp.asarray(False), lane)
    assert out[0]["mu"]["kernel"].shape == (128, 3)
    tda = ag.TDAgent(7)
    tl = jax.tree.map(lambda x: x[0], tr.stack_lanes("td", [tr.lane_cfg("td", "stream_td")]))
    tc = tda.init(jax.random.PRNGKey(0), tl, jnp.ones(7))
    tc, (pred, cum) = tda.step(tc, jnp.ones(7) * 2, jnp.float32(0.3), jnp.asarray(False), tl)
    assert np.isfinite(float(pred))


def test_under_30s():
    assert T0[0] is not None and time.time() - T0[0] < 30
