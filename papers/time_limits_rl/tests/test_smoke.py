"""Smoke tests (< 30 s on CPU): shapes + one PPO update with paired TA / no-TA / PEB lanes, the zero-time-column
equivalence, the rollout contract V(s'_t) == V(s_{t+1}) off resets, and one short run of each tabular learner.

Run: JAX_PLATFORMS=cpu uv run python -m pytest tests/test_smoke.py -q
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

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src import model as M  # noqa: E402
from src.train import PPO, lanes_from_cfg, load_config  # noqa: E402


@pytest.fixture(scope="module")
def small():
    cfg = load_config(os.path.join(REPO, "src", "configs", "e3_queue_of_cars.yaml"),
                      ["num_steps=64", "num_minibatches=4", "update_epochs=2", "total_timesteps=128", "probe_T=4",
                       "seeds=[0, 1000]", "hidden=16"])
    cfg["arms"] = {"ppo": {"time_mode": "none", "T": 4}, "ta": {"time_mode": "elapsed_pm1", "T": 4},
                   "peb": {"time_mode": "none", "T": 4, "peb": 1}}
    ppo = PPO(cfg)
    names, lane = lanes_from_cfg(cfg)
    r, k = jax.jit(jax.vmap(ppo.init))(lane)
    return cfg, ppo, names, lane, r, k


def test_paired_init_and_one_update(small):
    cfg, ppo, names, lane, r, k = small
    L = len(names)
    assert L == 6
    for leaf in jax.tree.leaves(r.params):  # every arm of a seed starts from identical parameters
        x = np.asarray(leaf)
        for i, (_, s) in enumerate(names):
            assert np.array_equal(x[i], x[names.index(("ppo", s))])
    r2, m = jax.jit(jax.vmap(lambda rr, kk, ll: ppo.run_chunk(rr, kk, ll, 1)))(r, k, lane)
    assert int(r2.u[0]) == 1
    assert m["p_danger"].shape == (L, 1, 4, 9)
    assert m["ep_ret_steps"].shape == (L, 1, 64)
    for v in jax.tree.leaves(r2.params):
        assert np.all(np.isfinite(np.asarray(v)))
    assert float(np.max(m["ratio_dev_first_mb"])) < 1e-5
    # time-unaware lanes: the time column's first-layer weights are EXACTLY unchanged (zero input -> zero gradient)
    for net in ("actor", "critic"):
        w0 = np.asarray(r.params["params"][net]["l0"]["kernel"])
        w1 = np.asarray(r2.params["params"][net]["l0"]["kernel"])
        for i, (arm, _) in enumerate(names):
            same = np.array_equal(w0[i, -1], w1[i, -1])
            assert same == (arm != "ta"), (arm, net)
    # the time-unaware policy probe is constant in t; the TA one is not
    pd = np.asarray(m["p_danger"])[:, 0]
    i_ppo, i_ta = names.index(("ppo", 0)), names.index(("ta", 0))
    assert np.allclose(pd[i_ppo], pd[i_ppo][0:1]) and not np.allclose(pd[i_ta], pd[i_ta][0:1])


def test_zero_time_column_equals_no_column():
    net_a, net_b = M.build(2, True, 16), M.build(2, True, 16)
    p = net_a.init(jax.random.PRNGKey(0), jnp.zeros((1, 6)))
    x = jax.random.normal(jax.random.PRNGKey(1), (7, 5))
    pb = jax.tree.map(lambda z: z, p)
    for net in ("actor", "critic"):
        pb["params"][net]["l0"]["kernel"] = p["params"][net]["l0"]["kernel"][:-1]
    (ma, sa), va = net_a.apply(p, M.net_input(x, jnp.zeros(7)))
    (mb, sb), vb = net_b.apply(pb, x)
    np.testing.assert_allclose(np.asarray(ma), np.asarray(mb), atol=1e-6)
    np.testing.assert_allclose(np.asarray(va), np.asarray(vb), atol=1e-6)


def test_rollout_contract_and_flags(small):
    """V(s'_t) == V(s_{t+1}) whenever transition t did not end an episode (what makes the GAE form equal Tonic's
    lambda-return form); timeouts happen exactly at t = T; PEB lanes see the same trajectory as no-PEB at step 0."""
    cfg, ppo, names, lane, r, k = small
    tj = jax.jit(jax.vmap(lambda rr, kk, ll: ppo.rollout(rr, kk, ll)[1]))(r, k, lane)
    v, vf, rs, tm, to = (np.asarray(x) for x in (tj.v, tj.v_fin, tj.reset, tj.term, tj.timeout))
    keep = rs[:, :-1] == 0
    np.testing.assert_allclose(vf[:, :-1][keep], v[:, 1:][keep], atol=1e-6)
    assert np.all(rs == np.maximum(tm, to)) and np.all(tm * to == 0)
    assert to.sum() > 0 and tm.sum() > 0
    # same seed, same params, same keys: the no-PEB and PEB lanes generate identical first rollouts
    i, j = names.index(("ppo", 0)), names.index(("peb", 0))
    np.testing.assert_array_equal(np.asarray(tj.a)[i], np.asarray(tj.a)[j])


def test_tabular_learners_run():
    tg = D.two_goal()
    NX, R = jnp.asarray(tg.NX), jnp.asarray(tg.R, jnp.float32)
    TN = jnp.asarray(tg.is_terminal[tg.NX], jnp.float32)
    out = jax.jit(jax.vmap(lambda a: A.tabular_q(jax.random.PRNGKey(0), NX, R, TN, jnp.asarray(tg.start, jnp.float32),
                                                 agent=a, T=3, gamma=0.99, eps=1.0, n_episodes=200, lr_power=0.7)))(
        jnp.arange(3))
    assert out.Q.shape == (3, 4, 27, 5) and np.all(np.isfinite(np.asarray(out.Q)))
    np.testing.assert_array_equal(np.asarray(out.returns)[0], np.asarray(out.returns)[2])  # same behavior, CRN
    dg = D.difficult_grid()
    o = jax.jit(lambda p: A.buffer_q(jax.random.PRNGKey(0), jnp.asarray(dg.NX), D.DGRID_GOAL, D.DGRID_START, peb=p,
                                     B=40, total_steps=1000, bin_steps=500))(1.0)
    assert o.bin_cnt.shape == (2,) and float(o.bin_cnt.sum()) >= 4
    th, _ = A.reinforce_jump(jax.random.PRNGKey(0), T=2, gamma=1.0, n_episodes=50, lr=0.05)
    assert np.isfinite(float(th))
