"""Smoke test (< 30 s, JAX_PLATFORMS=cpu): env shapes, one jitted pulse of every arm's lane step, one update per arm,
one baseline-eval CPI, policy iteration on a tiny transition set, and the stream identity across methods."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import algorithm as alg  # noqa: E402
import baselines as bl  # noqa: E402
import data as D  # noqa: E402
import train as TR  # noqa: E402
from model import ARMS, arm_config  # noqa: E402

CFG = TR.load_cfg(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "default.yaml"))


def test_env_shapes_and_step():
    p = D.env_params("mc_p04", "eq3")
    s = D.env_reset(jax.random.PRNGKey(0), p)
    o = D.obs_of(s, p)
    assert o.shape == (D.OBS_DIM,) and o.dtype == jnp.float32
    s2, r, (c, m, sz, th) = D.env_step(s, jnp.int32(11), jnp.int32(0), p.T_train, jax.random.PRNGKey(1), p)
    assert r.shape == () and 0.0 <= float(r) <= 1.0 and float(sz) == 3.0
    assert int(D.full_state_index(s2, p)) < D.N_POS * D.N_VEL * D.N_THETA


def test_one_step_every_arm():
    lanes, ids = TR.build_lanes(dict(CFG, eval_cpis=70), ["mc_p04"], ["eq3"], [0])
    lane = jax.tree.map(lambda x: x[0], lanes)
    for arm in ARMS:                     # every arm's config builds; the two code paths (MLP / LSTM) are stepped
        arm_config(arm, CFG)
    for arm in ("dqn", "ddrqn"):
        q = arm_config(arm, CFG)
        c = TR.init_carry(lane, q)
        step = jax.jit(lambda c, t: TR.lane_step(c, t, lane.params.T_train, lane, q, 500 * D.CPI, True))
        c2, out, fp, th = step(c, jnp.int32(0))
        assert out.shape == (8,) and int(c2.wr.k) == (1 % q.trace_len)
        batch = alg.buffer_sample(c2.buf, jax.random.PRNGKey(0), q.batch_size)
        assert batch["inputs"].shape == (q.batch_size, q.trace_len + 1, D.OBS_DIM)
        st, info = jax.jit(lambda s, b: alg.update(s, b, q))(c2.q, batch)
        assert np.isfinite(float(info["loss"])) and int(st.n_updates) == 1


def test_baseline_eval_and_pi():
    cfg = dict(CFG, eval_cpis=1, train_cpis=1)
    lanes, ids = TR.build_lanes(cfg, ["mc_p04"], ["eq3"], [0, 1])
    tab = jnp.asarray(np.tile(D.ASTAR_ACTION, (2, D.N_SEG, 1)).astype(np.int32))
    full = jnp.zeros((2, D.N_POS * D.N_VEL * D.N_THETA), jnp.int32)
    o1, fp1, th1 = TR.make_policy_eval(1, "table")(lanes, tab, full)
    o2, fp2, th2 = TR.make_policy_eval(1, "random")(lanes, tab, full)
    assert o1.shape == (2, 1, 8)
    np.testing.assert_array_equal(np.asarray(th1), np.asarray(th2))          # same eval stream for every policy
    np.testing.assert_array_equal(np.asarray(fp1), np.asarray(fp2))
    s, a, r, s2, _ = [np.asarray(x[0, :500]) for x in TR.collect_training(1)(lanes)]
    out = bl.policy_iteration(s, a, r, s2)
    assert out["table"].shape == (D.N_POS * D.N_VEL * D.N_THETA,) and out["iters"] >= 1
