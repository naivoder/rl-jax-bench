"""Equivalences: batched while-loop rollout == per-episode scan rollout; bucket-sliced learning == cap-padded
learning; incremental (rollout) features == sequence (loss) features, incl. react-k windows."""
import time

from _helpers import REPO  # noqa: F401

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from algorithm import init_run_state, make_rollout, make_update, slice_episodes  # noqa: E402
from envs import heavenhell_probes, make_env  # noqa: E402
from model import METHODS, Actor, NetSpec, build, init_params  # noqa: E402

ENV = make_env("heavenhell_3")
TINY = NetSpec(ENV.kind, ENV.n_actions, ENV.n_obs, ENV.n_states, emb=8, hidden=16, head=(16,))
CFG = dict(gamma=0.99, cap=16, episodes_per_update=2, max_env_steps=10_000, target_period=100, n_log=20,
           alpha_pi=1e-3, alpha_v=1e-3, adam_eps=1e-4, lambda0=0.1, lambda_final_factor=0.1, negentropy_nsteps=2000,
           length_buckets=[8])


def _rs(method, R=2, spec=TINY, probes=0):
    def one(k):
        pa, pc = init_params(spec, method, jax.random.fold_in(k, 0))
        return init_run_state(pa, pc, jax.random.fold_in(k, 1), CFG, probes)
    return jax.vmap(one)(jax.random.split(jax.random.PRNGKey(0), R))


def test_batched_rollout_equals_per_episode_scan_and_buckets_are_exact():
    m = "a2c_asym_hs"
    upd = make_update(ENV, TINY, m, CFG)
    rs = _rs(m)
    keys = jax.vmap(lambda k, u: jax.random.split(jax.random.fold_in(k, u), 2))(rs.key, rs.upd)
    eb = jax.jit(upd.rollout)(rs.pa, keys, jnp.array([True, True]))
    single = make_rollout(ENV, TINY, m, CFG["cap"])
    es = jax.vmap(lambda p, k: jax.vmap(single, (None, 0))(p, k))(rs.pa, keys)
    for f in ("actions", "rewards", "mask", "length", "trunc"):
        assert np.array_equal(np.asarray(getattr(eb, f)), np.asarray(getattr(es, f))), f
    valid = np.concatenate([np.asarray(es.mask), np.zeros(es.mask.shape[:-1] + (1,), bool)], -1)
    valid |= np.arange(CFG["cap"] + 1) == np.asarray(es.length)[..., None]
    assert np.array_equal(np.asarray(eb.states)[valid], np.asarray(es.states)[valid])
    assert np.array_equal(np.asarray(eb.obs)[valid], np.asarray(es.obs)[valid])
    # learning on the bucket-sliced batch == learning on the cap-padded batch
    L = int(np.asarray(eb.length).max())
    full = jax.jit(jax.vmap(upd.learn))(rs, eb)
    sl = jax.jit(jax.vmap(upd.learn))(rs, slice_episodes(eb, L))
    for a, b in zip(jax.tree.leaves(full.pa) + jax.tree.leaves(full.pc), jax.tree.leaves(sl.pa) + jax.tree.leaves(sl.pc)):
        assert np.allclose(a, b, rtol=1e-5, atol=1e-7)



def test_step_features_equal_sequence_features():
    """The rollout's incremental GRU (and react-k buffer) gives the same policy as the sequence pass of the loss."""
    for m in ("a2c", "a2c_react_2", "a2c_react_4"):
        actor, _ = build(TINY, m)
        pa, _ = init_params(TINY, m, jax.random.PRNGKey(3))
        rng = np.random.default_rng(0)
        a = jnp.asarray(rng.integers(0, ENV.n_actions, 9), jnp.int32)
        o = jnp.asarray(np.r_[ENV.n_obs, rng.integers(0, ENV.n_obs, 8)], jnp.int32)
        seq = actor.apply({"params": pa}, a, o)
        c = actor.apply({"params": pa}, method=Actor.init_carry)
        prev = jnp.int32(-1)
        for t in range(9):
            c, lp = actor.apply({"params": pa}, c, prev, o[t], method=Actor.step)
            assert np.allclose(lp, seq[t], atol=1e-5), (m, t)
            prev = a[t]


@pytest.mark.parametrize("method", list(METHODS))
def test_one_update_every_arm(method):
    probes = {k: jnp.asarray(v) for k, v in heavenhell_probes("heavenhell_3").items()}
    upd = jax.jit(make_update(ENV, TINY, method, {**CFG, "length_buckets": []}, probes))
    rs = _rs(method, probes=4)
    rs2 = upd(rs)
    assert np.all(np.asarray(rs2.t_env) > 0) and np.all(np.asarray(rs2.upd) == 1)
    assert not np.any(np.asarray(rs2.bad))
    assert np.all(np.asarray(rs2.log_idx) == 1)          # first update logs at t = 0 (official Dispenser)
    changed = jax.tree.map(lambda a, b: bool(jnp.any(a != b)), rs.pa, rs2.pa)
    assert any(jax.tree.leaves(changed))


def test_td_lambda_diagnostic_matches_official_matrix_form():
    """Diagnostic estimator: official q_estimators.tdlambda_q_estimator is
    Q = triu((g*lam)^(j-i)) @ (r + g (1 - lam) V_next), V_next = roll(V, -1) with the last entry 0."""
    from algorithm import Episode, td_targets
    rng = np.random.default_rng(0)
    L, C, g, lam = 9, 12, 0.99, 0.7
    r = rng.normal(size=L)
    v = rng.normal(size=L)
    vn = np.r_[v[1:], 0.0]
    idx = np.arange(L)
    D = np.triu((g * lam) ** (idx[None] - idx[:, None]).astype(float))
    ref = D @ (r + g * (1 - lam) * vn)
    ep = Episode(states=jnp.zeros(C + 1, jnp.int32), obs=jnp.zeros(C + 1, jnp.int32), actions=jnp.zeros(C + 1, jnp.int32),
                 rewards=jnp.asarray(np.r_[r, np.zeros(C - L)], jnp.float32), mask=jnp.asarray(np.arange(C) < L),
                 trunc=jnp.bool_(False), length=jnp.int32(L))
    vv = jnp.asarray(np.r_[v, np.zeros(C + 1 - L)], jnp.float32)
    q = np.asarray(td_targets(ep, vv, g, "tdlambda", lam))[:L]
    assert np.allclose(q, ref, atol=1e-5)
    q0 = np.asarray(td_targets(ep, vv, g, "td0"))[:L]
    assert np.allclose(q0, r + g * vn, atol=1e-6)


@pytest.mark.parametrize("env_name", ["car_flag", "cleaner"])
def test_bucket_slicing_vector_envs(env_name):
    """Vector-feature envs: bucket-sliced learning == cap-padded learning, and a full jitted update with length
    buckets runs (regression: slice_episodes once sliced the trailing FEATURE axis of states/obs)."""
    env = make_env(env_name)
    spec = NetSpec(env.kind, env.n_actions, env.n_obs, env.n_states, emb=8, hidden=16, head=(16,))
    for m in ("a2c_asym_hs", "a2c_asym_s", "a2c_react_2"):
        upd = make_update(env, spec, m, CFG)
        rs = _rs(m, spec=spec)
        keys = jax.vmap(lambda k, u: jax.random.split(jax.random.fold_in(k, u), 2))(rs.key, rs.upd)
        eb = jax.jit(upd.rollout)(rs.pa, keys, jnp.array([True, True]))
        L = 8
        eb = eb._replace(length=jnp.minimum(eb.length, L), mask=eb.mask & (jnp.arange(CFG["cap"]) < L))
        sl = slice_episodes(eb, L)
        assert sl.obs.shape[-2:] == (L + 1, eb.obs.shape[-1]) and sl.states.shape[-1] == eb.states.shape[-1]
        full = jax.jit(jax.vmap(upd.learn))(rs, eb)
        cut = jax.jit(jax.vmap(upd.learn))(rs, sl)
        for a, b in zip(jax.tree.leaves(full.pa) + jax.tree.leaves(full.pc),
                        jax.tree.leaves(cut.pa) + jax.tree.leaves(cut.pc)):
            assert np.allclose(a, b, rtol=1e-5, atol=1e-7)
        rs2 = jax.jit(upd)(rs)
        assert not np.any(np.asarray(rs2.bad)) and np.all(np.asarray(rs2.upd) == 1)
