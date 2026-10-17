"""Gradient and estimator correctness (toy checks items 1, 4, 5 and the DRQN burn-in equivalences). CPU, float64 where needed.

* test_exact_mdp            value iteration reproduces the exact Table D1 values (below) to 1e-6 (both readings, 5 values of p),
                            the Q*/V* worked example (R_eq3, p = 0.4) and V* = 10 on FH.
* test_targets_by_hand      Eq. 11 and Eq. 14 on a constructed batch whose online and target argmaxes differ.
* test_check_grads_*        finite-difference check_grads of the TD loss (MLP head; LSTM trace loss with and without
                            burn-in), float64.
* test_burnin_*             b = 0 with a stored zero state equals the zero-state primary; stored state + burn-in equals
                            a full-history unroll at matching weights.
* test_writer_alignment     the trace writer produces aligned, non-overlapping traces with the act-time resets.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import algorithm as alg  # noqa: E402
import baselines as bl  # noqa: E402
import data as D  # noqa: E402

D1 = {  # (reading, p): (J*, pi*(OFF), pi*(ON), J_SAA, J_rand, J_constB)   the README
    ("eq3", 0.1): (0.92083, 4, 11, 0.92083, 0.33287, 0.70833), ("eq3", 0.2): (0.84167, 4, 11, 0.84167, 0.33287, 0.70833),
    ("eq3", 0.4): (0.70833, 11, 11, 0.68333, 0.33287, 0.70833), ("eq3", 0.6): (0.70833, 11, 11, 0.52500, 0.33287, 0.70833),
    ("eq3", 0.8): (0.84167, 11, 4, 0.36667, 0.33287, 0.70833), ("lin", 0.1): (0.93333, 4, 11, 0.93333, 0.43333, 0.83333),
    ("lin", 0.2): (0.86667, 4, 11, 0.86667, 0.43333, 0.83333), ("lin", 0.4): (0.83333, 11, 11, 0.73333, 0.43333, 0.83333),
    ("lin", 0.6): (0.83333, 11, 11, 0.60000, 0.43333, 0.83333), ("lin", 0.8): (0.86667, 11, 4, 0.46667, 0.43333, 0.83333),
}


def test_exact_mdp():
    for (rd, p), (J, a0, a1, Js, Jr, JB) in D1.items():
        R = D.reward_table(rd)
        c = D.markov_chain(p)
        sol = bl.exact_solve(c, R)
        assert tuple(sol["pi"]) == (a0, a1), (rd, p, sol["pi"])
        for got, want in ((bl.stationary_J(c, R, sol["pi"]), J), (bl.stationary_J(c, R, bl.saa_policy(c)), Js),
                          (bl.stationary_J(c, R, None), Jr), (bl.stationary_J(c, R, np.array([11, 11])), JB)):
            assert abs(got - want) < 6e-6, (rd, p, got, want)       # D1 is printed to 5 decimals
    sol = bl.exact_solve(D.markov_chain(0.4), D.reward_table("eq3"))
    np.testing.assert_allclose(sol["V"], [7.01220, 7.15447], atol=6e-6)
    np.testing.assert_allclose(sol["Q"][0, [11, 4]], [7.01220, 6.96220], atol=6e-6)
    # the README prints Q*(ON, F) = 6.47114; by hand it is rbar(ON, F) + C(ON) = p + 6.38780 = 6.78780
    # (erratum dated 2026-10-16 in the README; V*, the gaps and the second-best action 00011 are unaffected)
    np.testing.assert_allclose(sol["Q"][1, [11, 4, 13]], [7.15447, 6.78780, 6.99892], atol=6e-6)
    np.testing.assert_allclose(sol["gap"], [0.05, 0.15556], atol=6e-6)
    for rd in D.READINGS:
        f = bl.exact_solve(D.fh_chain(), D.reward_table(rd))
        np.testing.assert_allclose(f["V"], 10.0, atol=1e-9)
        assert abs(bl.stationary_J(D.fh_chain(), D.reward_table(rd), f["pi"]) - 1.0) < 1e-12
    t2 = bl.stationary_table2(D.fh_chain(), bl.saa_policy(D.fh_chain()))       # Table II FH-SAA row
    np.testing.assert_allclose([t2["coll_pct"], t2["missed_pct"], t2["bw_mhz"], t2["adapt_pct"]], [60, 40, 64, 100],
                               atol=1e-9)


def test_targets_by_hand():
    qn_on = jnp.array([[1.0, 3.0, 2.0], [0.5, 0.1, 0.2]])
    qn_tg = jnp.array([[5.0, 4.0, 6.0], [1.0, 2.0, 3.0]])
    r, d = jnp.array([1.0, -1.0]), jnp.array([0.9, 0.0])
    np.testing.assert_allclose(alg.td_target(qn_on, qn_tg, r, d, double=False), [1 + 0.9 * 6, -1.0])   # Eq. 11
    np.testing.assert_allclose(alg.td_target(qn_on, qn_tg, r, d, double=True), [1 + 0.9 * 4, -1.0])    # Eq. 14
    # through sequence_loss with a hand-made unroll: q(params) = params["w"][None] broadcast (online), target fixed
    on = {"w": jnp.array([[0.0, 2.0, 1.0], [1.0, 3.0, 2.0]])}
    tg = {"w": jnp.array([[0.0, 0.0, 0.0], [5.0, 4.0, 6.0]])}

    def unroll(p, inputs, carry0, resets=None):
        return jnp.broadcast_to(p["w"][None], (inputs.shape[0], 2, 3)), carry0

    batch = dict(inputs=jnp.zeros((1, 2, 1)), actions=jnp.array([[0]]), rewards=jnp.array([[1.0]]),
                 discounts=jnp.array([[0.9]]))
    for double, y in ((False, 1 + 0.9 * 6), (True, 1 + 0.9 * 4)):
        cfg = alg.QConfig(n_actions=3, obs_dim=1, double=double)
        loss, aux = alg.sequence_loss(on, tg, batch, cfg, unroll)
        np.testing.assert_allclose(float(aux["y"][0, 0]), y, rtol=1e-6)
        np.testing.assert_allclose(float(loss), (0.0 - y) ** 2, rtol=1e-6)   # float32


def _rand_batch(key, B, L, D_, A):
    k = jax.random.split(key, 4)
    return dict(inputs=jax.random.normal(k[0], (B, L + 1, D_)), actions=jax.random.randint(k[1], (B, L), 0, A),
                rewards=jax.random.normal(k[2], (B, L)), discounts=jnp.full((B, L), 0.9),
                resets=jnp.zeros((B, L + 1), bool))


def _perturb(params, key, s=0.3):
    leaves, td = jax.tree.flatten(params)
    ks = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(td, [x + s * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, ks)])


def test_check_grads_mlp_head():
    with jax.enable_x64(True):
        for double in (False, True):
            cfg = alg.QConfig(n_actions=4, obs_dim=3, hidden=(8, 6), lstm=0, double=double, trace_len=1)
            st = alg.init(jax.random.PRNGKey(0), cfg)
            p = jax.tree.map(lambda x: x.astype(jnp.float64), st.params)
            tgt = _perturb(p, jax.random.PRNGKey(1))
            b = jax.tree.map(lambda x: x.astype(jnp.float64) if x.dtype == jnp.float32 else x,
                             _rand_batch(jax.random.PRNGKey(2), 5, 1, 3, 4))
            check_grads(lambda q: alg.sequence_loss(q, tgt, b, cfg)[0], (p,), order=1, modes=["rev"], eps=1e-6,
                        atol=1e-6, rtol=1e-6)


def test_check_grads_lstm_trace_loss():
    """b = 0: finite differences of the whole trace loss (BPTT through 4 steps from a nonzero carry0).
    b > 0: R2D2 stops the gradient at the burned-in state, so the analytic gradient must equal the gradient of the
    b = 0 loss on the remaining steps started from that (constant) state -- asserted exactly, and that function is
    finite-difference checked."""
    with jax.enable_x64(True):
        f64 = lambda t: jax.tree.map(lambda x: x.astype(jnp.float64) if x.dtype == jnp.float32 else x, t)
        b = f64(_rand_batch(jax.random.PRNGKey(5), 3, 4, 3, 4))
        b["carry0"] = (0.5 * jnp.ones((3, 5)), -0.3 * jnp.ones((3, 5)))
        cfg0 = alg.QConfig(n_actions=4, obs_dim=3, hidden=(8,), lstm=5, double=True, trace_len=4, burn_in=0)
        p = f64(alg.init(jax.random.PRNGKey(3), cfg0).params)
        tgt = _perturb(p, jax.random.PRNGKey(4))
        check_grads(lambda q: alg.sequence_loss(q, tgt, b, cfg0)[0], (p,), order=1, modes=["rev"], eps=1e-6,
                    atol=1e-6, rtol=1e-6)
        cfg2 = alg.QConfig(n_actions=4, obs_dim=3, hidden=(8,), lstm=5, double=True, trace_len=4, burn_in=2)
        net = alg.make_network(cfg2)
        _, c_on = alg.unroll(net, cfg2, p, b["inputs"][:, :2], b["carry0"])
        _, c_tg = alg.unroll(net, cfg2, tgt, b["inputs"][:, :2], b["carry0"])

        def sliced(q):
            # online from the online burned-in state, target from the target burned-in state (as sequence_loss does)
            q_on, _ = alg.unroll(net, cfg2, q, b["inputs"][:, 2:], c_on)
            q_tg, _ = alg.unroll(net, cfg2, tgt, b["inputs"][:, 2:], c_tg)
            q_sa = jnp.take_along_axis(q_on[:, :-1], b["actions"][:, 2:, None], -1)[..., 0]
            y = jax.lax.stop_gradient(alg.td_target(q_on[:, 1:], q_tg[:, 1:], b["rewards"][:, 2:], b["discounts"][:, 2:], True))
            return ((q_sa - y) ** 2).mean()

        g_burn = jax.grad(lambda q: alg.sequence_loss(q, tgt, b, cfg2)[0])(p)
        g_sl = jax.grad(sliced)(p)
        jax.tree.map(lambda a, c: np.testing.assert_allclose(np.asarray(a), np.asarray(c), rtol=1e-10, atol=1e-12),
                     g_burn, g_sl)
        check_grads(sliced, (p,), order=1, modes=["rev"], eps=1e-6, atol=1e-6, rtol=1e-6)


def _setup_rec(L=6, b=0):
    cfg = alg.QConfig(n_actions=4, obs_dim=3, hidden=(8,), lstm=5, double=True, trace_len=L, burn_in=b)
    st = alg.init(jax.random.PRNGKey(7), cfg)
    return cfg, st.params, _perturb(st.params, jax.random.PRNGKey(8))


def test_burnin_zero_equals_primary():
    cfg, p, t = _setup_rec()
    b = _rand_batch(jax.random.PRNGKey(9), 4, 6, 3, 4)
    l0, a0 = alg.sequence_loss(p, t, b, cfg)
    b2 = dict(b, carry0=alg.initial_carry(cfg, (4,)))
    l1, a1 = alg.sequence_loss(p, t, b2, cfg)
    np.testing.assert_array_equal(np.asarray(a0["y"]), np.asarray(a1["y"]))
    assert float(l0) == float(l1)


def test_burnin_stored_state_equals_full_history():
    """Trace from t0 with the actor's stored carry and burn-in b reproduces the full-history unroll's Q and targets."""
    Tn, t0, L, b = 40, 17, 10, 4
    cfg, p, t = _setup_rec(L, b)
    net = alg.make_network(cfg)
    xs = jax.random.normal(jax.random.PRNGKey(10), (1, Tn, 3))
    acts = jax.random.randint(jax.random.PRNGKey(11), (1, Tn), 0, 4)
    rews = jax.random.normal(jax.random.PRNGKey(12), (1, Tn))
    q_full, _ = alg.unroll(net, cfg, p, xs, alg.initial_carry(cfg, (1,)))
    qt_full, _ = alg.unroll(net, cfg, t, xs, alg.initial_carry(cfg, (1,)))
    # the actor's carry before consuming x[t0], by stepping act_q (online) -- this is what store_carry records
    carry = alg.initial_carry(cfg, (1,))
    carry_tgt = alg.initial_carry(cfg, (1,))
    for k in range(t0):
        _, carry = alg.act_q(net, cfg, p, xs[:, k], carry)
        _, carry_tgt = alg.act_q(net, cfg, t, xs[:, k], carry_tgt)
    batch = dict(inputs=xs[:, t0:t0 + L + 1], actions=acts[:, t0:t0 + L], rewards=rews[:, t0:t0 + L],
                 discounts=jnp.full((1, L), 0.9), carry0=carry)
    _, aux = alg.sequence_loss(p, t, batch, cfg)
    ks = np.arange(t0 + b, t0 + L)
    want_q = np.asarray(q_full[0, ks, acts[0, ks]])
    np.testing.assert_allclose(np.asarray(aux["q_sa"][0]), want_q, rtol=1e-5, atol=1e-6)
    # R2D2 unrolls the target net from the same stored (online-actor) state; the full-history target unroll differs
    # only through that initial state, whose effect after b burn-in steps is checked against the exact recomputation:
    a_on = np.asarray(q_full[0, ks + 1].argmax(-1))
    _, c_tg = alg.unroll(net, cfg, t, xs[:, t0:t0 + b], carry)
    qt_tr, _ = alg.unroll(net, cfg, t, xs[:, t0 + b:t0 + L + 1], c_tg)
    want_y = np.asarray(rews[0, ks]) + 0.9 * np.asarray(qt_tr[0, 1:])[np.arange(L - b), a_on]
    np.testing.assert_allclose(np.asarray(aux["y"][0]), want_y, rtol=1e-5, atol=1e-6)
    # with the TARGET carry stored for the target unroll the trace equals the full-history targets exactly
    want_y_full = np.asarray(rews[0, ks]) + 0.9 * np.asarray(qt_full[0, ks + 1])[np.arange(L - b), a_on]
    _, c_tg2 = alg.unroll(net, cfg, t, xs[:, t0:t0 + b], carry_tgt)
    qt_tr2, _ = alg.unroll(net, cfg, t, xs[:, t0 + b:t0 + L + 1], c_tg2)
    np.testing.assert_allclose(np.asarray(qt_tr2[0, 1:])[np.arange(L - b), a_on] * 0.9 + np.asarray(rews[0, ks]),
                               want_y_full, rtol=1e-5, atol=1e-6)


def test_writer_alignment():
    cfg = alg.QConfig(n_actions=3, obs_dim=2, lstm=4, trace_len=4, buffer_traces=5, store_carry=True)
    w, buf = alg.writer_init(cfg), alg.buffer_init(cfg)
    obs = jnp.arange(30, dtype=jnp.float32).reshape(15, 2)
    for t in range(14):
        w, buf, pushed = alg.writer_step(w, buf, obs[t], t % 3, float(t), 0.9, obs[t + 1], t % 4 == 0, (t + 1) % 4 == 0,
                                         (jnp.full(4, t, jnp.float32), jnp.full(4, -t, jnp.float32)))
        assert bool(pushed) == ((t + 1) % 4 == 0)
    assert int(buf.count) == 3 and int(buf.ptr) == 3
    for j in range(3):
        np.testing.assert_array_equal(np.asarray(buf.obs[j]), np.asarray(obs[4 * j:4 * j + 5]))
        np.testing.assert_array_equal(np.asarray(buf.rew[j]), np.arange(4 * j, 4 * j + 4, dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(buf.resets[j]), [True, False, False, False, True])
        np.testing.assert_array_equal(np.asarray(buf.carry0[0][j]), np.full(4, 4 * j))
