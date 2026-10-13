"""Smoke test (<30 s, CPU): shapes, one full jitted update for every arm (tiny widths), frozen finished runs."""
import time

from _helpers import REPO  # noqa: F401

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from algorithm import init_run_state, make_rollout, make_update, slice_episodes  # noqa: E402
from envs import heavenhell_probes, make_env  # noqa: E402
from model import METHODS, Actor, NetSpec, build, init_params  # noqa: E402

T0 = [None]   # set when the first smoke test starts (module import happens at collection time)
ENV = make_env("heavenhell_3")
TINY = NetSpec(ENV.kind, ENV.n_actions, ENV.n_obs, ENV.n_states, emb=8, hidden=16, head=(16,))
CFG = dict(gamma=0.99, cap=16, episodes_per_update=2, max_env_steps=10_000, target_period=100, n_log=20,
           alpha_pi=1e-3, alpha_v=1e-3, adam_eps=1e-4, lambda0=0.1, lambda_final_factor=0.1, negentropy_nsteps=2000,
           length_buckets=[8])


@pytest.fixture(autouse=True, scope="module")
def _clock():
    T0[0] = time.time()
    yield


def _rs(method, R=2, spec=TINY, probes=0):
    def one(k):
        pa, pc = init_params(spec, method, jax.random.fold_in(k, 0))
        return init_run_state(pa, pc, jax.random.fold_in(k, 1), CFG, probes)
    return jax.vmap(one)(jax.random.split(jax.random.PRNGKey(0), R))


def test_shapes_full_size():
    spec = NetSpec(ENV.kind, ENV.n_actions, ENV.n_obs, ENV.n_states)
    T = 7
    a = jnp.zeros(T, jnp.int32)
    o = jnp.full(T, ENV.n_obs, jnp.int32)
    for m in METHODS:
        actor, critic = build(spec, m)
        pa, pc = init_params(spec, m, jax.random.PRNGKey(1))
        lp = actor.apply({"params": pa}, a, o)
        assert lp.shape == (T, ENV.n_actions) and np.allclose(np.exp(lp).sum(-1), 1.0, atol=1e-5)
        assert critic.apply({"params": pc}, a, o, a).shape == (T,)


@pytest.mark.parametrize("method", ["a2c_asym_hs", "a2c_react_4"])   # all 5 arms: test_equivalence.py
def test_one_update(method):
    probes = {k: jnp.asarray(v) for k, v in heavenhell_probes("heavenhell_3").items()}
    upd = jax.jit(make_update(ENV, TINY, method, {**CFG, "length_buckets": []}, probes))
    rs = _rs(method, probes=4)
    rs2 = upd(rs)
    assert np.all(np.asarray(rs2.t_env) > 0) and np.all(np.asarray(rs2.upd) == 1)
    assert not np.any(np.asarray(rs2.bad))
    assert np.all(np.asarray(rs2.log_idx) == 1)          # first update logs at t = 0 (official Dispenser)
    changed = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), rs.pa, rs2.pa)
    assert any(jax.tree.leaves(changed))


def test_inactive_runs_are_frozen():
    m = "a2c"
    upd = jax.jit(make_update(ENV, TINY, m, {**CFG, "length_buckets": []}))
    rs = _rs(m)
    rs = rs._replace(t_env=jnp.array([CFG["max_env_steps"], 0], jnp.int32))
    rs2 = upd(rs)
    assert int(rs2.upd[0]) == 0 and int(rs2.upd[1]) == 1
    same = jax.tree.map(lambda a, b: bool(jnp.all(a[0] == b[0])), rs.pa, rs2.pa)
    assert all(jax.tree.leaves(same))


def test_budget():
    assert time.time() - T0[0] < 30, "smoke test must stay under 30 s (excluding import/collection)"
