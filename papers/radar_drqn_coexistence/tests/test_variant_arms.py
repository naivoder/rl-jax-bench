"""full-scale audit arms (T-A sliding-window DRQN, T-B hard_sigmoid gates): one jitted step + update each, the window loss
against a step-by-step act_q recomputation, and the hard_sigmoid gate formula. CPU, a few seconds."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import algorithm as alg  # noqa: E402
import data as D  # noqa: E402
import train as TR  # noqa: E402
from model import arm_config  # noqa: E402

CFG = TR.load_cfg(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "default.yaml"))


def test_defaults_unchanged():
    q = arm_config("ddrqn", CFG)
    assert q.recurrent_activation == "sigmoid" and q.buffer_traces == 200
    w = arm_config("ddrqn_win", CFG)
    assert w.buffer_traces == 2000 and w.trace_len == 10
    assert arm_config("ddrqn_hs", CFG).recurrent_activation == "hard_sigmoid"


def test_hard_sigmoid_gate():
    x = jnp.linspace(-4, 4, 17)
    np.testing.assert_allclose(alg.gate_activation("hard_sigmoid")(x), np.clip(0.2 * np.asarray(x) + 0.5, 0, 1), atol=1e-7)
    q = arm_config("ddrqn_hs", CFG)
    st = alg.init(jax.random.PRNGKey(0), q)
    s0 = alg.init(jax.random.PRNGKey(0), arm_config("ddrqn", CFG))
    for a, b in zip(jax.tree.leaves(st.params), jax.tree.leaves(s0.params)):
        np.testing.assert_array_equal(a, b)            # same parameters; only the gate nonlinearity differs


def test_window_step_and_loss():
    lanes, _ = TR.build_lanes(dict(CFG, eval_cpis=70), ["mc_p04"], ["eq3"], [0])
    lane = jax.tree.map(lambda x: x[0], lanes)
    q = arm_config("ddrqn_win", CFG)
    c = TR.init_carry(lane, q, True)
    step = jax.jit(lambda c, t: TR.lane_step_window(c, t, lane.params.T_train, lane, q, 500 * D.CPI, True))
    for t in range(40):
        c, out, _, _ = step(c, jnp.int32(t))
    assert int(c.buf.count) == 40 and float(out[7]) == 1.0          # learning started at 32 windows
    np.testing.assert_array_equal(np.asarray(c.buf.obs[39, :10]), np.asarray(c.ac))   # stored window = actor window
    batch = alg.buffer_gather(c.buf, jnp.arange(3, 6))
    loss, _ = TR.window_loss(c.q.params, c.q.target_params, batch, q)
    net = alg.make_network(q)

    def last_q(params, win):           # step-by-step LSTM from a zero state
        h = alg.initial_carry(q, (win.shape[0],))
        for k in range(win.shape[1]):
            qq, h = alg.act_q(net, q, params, win[:, k], h)
        return qq

    inp = batch["inputs"]
    q_cur = last_q(c.q.params, inp[:, :10])
    a_on = jnp.argmax(last_q(c.q.params, inp[:, 1:]), -1)
    v = jnp.take_along_axis(last_q(c.q.target_params, inp[:, 1:]), a_on[:, None], -1)[:, 0]
    y = batch["rewards"][:, 9] + 0.9 * v
    want = jnp.mean((jnp.take_along_axis(q_cur, batch["actions"][:, 9:10], -1)[:, 0] - y) ** 2)
    np.testing.assert_allclose(float(loss), float(want), rtol=1e-5)
    st, info = jax.jit(lambda s, b: TR.window_update(s, b, q))(c.q, batch)
    assert np.isfinite(float(info["loss"])) and int(st.n_updates) == int(c.q.n_updates) + 1


def test_review_arms():
    """review audit arms: lr override only changes lr; the no-reset arm resets the actor carry once per CPI; the
    shipped arms keep the  act-time reset period (= trace length)."""
    from model import act_reset_period
    cfg = TR.load_cfg(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "full.yaml"))
    a, b = arm_config("ddrqn", cfg), arm_config("ddrqn_lr1e4", cfg)
    assert b.lr == 1e-4 and a.lr == cfg["lr"] and a.__dict__ | {"lr": 1e-4} == b.__dict__
    assert arm_config("dqn_lr1e4", cfg).__dict__ == arm_config("dqn", cfg).__dict__ | {"lr": 1e-4}
    assert arm_config("ddrqn_noreset", cfg) == a
    assert act_reset_period("ddrqn") == 10 and act_reset_period("dqn") == 1 and act_reset_period("ddrqn_noreset") == D.CPI
    # the actor carry is NOT zeroed mid-CPI under the no-reset period, and is under the shipped one
    qcfg = a
    lanes, _ = TR.build_lanes(dict(cfg, seed_root=3, eval_cpis=10), ["mc_p04"], ["eq3"], [0])
    lane = jax.tree.map(lambda x: x[0], lanes)
    c = TR.init_carry(lane, qcfg)
    for rp, expect_zero in ((10, True), (D.CPI, False)):
        c1 = c._replace(ac=jax.tree.map(lambda x: x + 1.0, c.ac))
        c2, *_ = TR.lane_step(c1, jnp.int32(20), lane.params.T_train, lane, qcfg, 1000, True, rp)
        # after a reset at t = 20 the carry depends only on obs_20; compare with a zero-carry step
        c3, *_ = TR.lane_step(c, jnp.int32(20), lane.params.T_train, lane, qcfg, 1000, True, rp)
        same = all(np.allclose(x, y) for x, y in zip(jax.tree.leaves(c2.ac), jax.tree.leaves(c3.ac)))
        assert same == expect_zero
