"""Official-code parity: differential tests of our DQN/DRQN core against fixtures recorded from oracle code.

No author code exists for Thornton et al. 2020 (see the README); every oracle here is a labelled second-best.
Fixtures were recorded by scripts/official/{cleanrl_harness.py, pomdp_harness.py, parity_fixtures.py}
(copied to tests/fixtures/, sha256 in tests/fixtures/SHA256SUMS):
  cleanrl_{max,double}.npz  CleanRL dqn_jax.py @ fe8d8a0 (QNetwork / update / linear_schedule run verbatim) on
                            gymnax CartPole-v1: 249 updates, every batch, loss, q_pred, param snapshots, eps, syncs.
  pomdp_seq_target.npz      pomdp-baselines @ e7c19c3 Critic_RNN + SACD.critic_loss reduced to the double-DQN
                            sequence target (alpha = 0, twin heads tied, greedy stub actor), float64.
  cleanrl_env_traj.npz      CleanRL dqn_jax.py's own make_env in gymnasium 0.29.1 SyncVectorEnv (its own venv), 1200 steps of a
                            fixed action stream: next/real_next obs, rewards, terminations, truncations, logged episodic returns.
  pomdp_lstm.npz            the nn.LSTM that pomdp-baselines Critic_RNN builds (torch 1.13.1, float64), whole sequence and
                            stepwise from a nonzero (h, c).
Run on CPU: JAX_PLATFORMS=cpu pytest tests/test_official_parity.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import algorithm as alg  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _cleanrl_params(z, prefix):
    m = {"Dense_0": "fc0", "Dense_1": "fc1", "Dense_2": "q"}
    p = {}
    for k in z.files:
        if k.startswith(prefix + "/"):
            _, _, layer, leaf = k.split("/")       # prefix/params/Dense_i/kernel
            p.setdefault(m[layer], {})[leaf] = jnp.asarray(z[k])
    return {"params": p}


@pytest.mark.parametrize("variant", ["max", "double"])
def test_cleanrl_dqn_differential(variant):
    z = np.load(os.path.join(FIX, f"cleanrl_{variant}.npz"))
    total, ls, tf, tnf, bsz, _, lr, gamma, s_e, e_e, frac = z["config"]
    cfg = alg.QConfig(n_actions=2, obs_dim=4, hidden=(120, 84), lstm=0, double=(variant == "double"), gamma=float(gamma),
                      lr=float(lr), adam_eps=1e-8, trace_len=1, batch_size=int(bsz))
    # epsilon schedule (all 3000 steps)
    eps = np.asarray(alg.linear_epsilon(jnp.arange(int(total)), float(s_e), float(e_e), float(frac) * float(total)))
    np.testing.assert_allclose(eps, z["eps"], rtol=1e-6, atol=1e-7)
    # target-update cadence: CleanRL syncs at env steps % 500 == 0 after learning_starts; with train_frequency 10 that
    # is "after every 50th gradient step" here -- our sync_due(n, 50) must select exactly the same updates
    upd_step = z["upd_step"]
    sync_after = {int(np.searchsorted(upd_step, s, side="right")) for s in z["sync_step"]}   # n_updates at each sync
    ours = {n for n in range(1, len(upd_step) + 1) if bool(alg.sync_due(alg.QState(None, None, None, jnp.int32(n)), 50))}
    assert sync_after == ours, (sync_after, ours)
    params = _cleanrl_params(z, "init")
    st = alg.QState(params, jax.tree.map(jnp.copy, params), alg.make_optimizer(cfg).init(params), jnp.zeros((), jnp.int32))
    upd = jax.jit(lambda s, b: alg.update(s, b, cfg))
    net = alg.make_network(cfg)
    act = jax.jit(lambda p, o: alg.greedy(alg.act_q(net, cfg, p, o, ())[0]))
    gsteps, gobs, gact = z["greedy_step"], z["greedy_obs"], z["greedy_act"]
    gi, n_greedy_checked = 0, 0
    for i in range(len(z["loss"])):
        # CleanRL acts before it updates within a step, so every greedy action at env step <= upd_step[i] was
        # taken with the params in force before update i
        while gi < len(gsteps) and gsteps[gi] <= upd_step[i]:
            assert int(act(st.params, jnp.asarray(gobs[gi]))) == int(gact[gi])
            n_greedy_checked += 1
            gi += 1
        batch = dict(inputs=jnp.stack([z["b_obs"][i], z["b_nobs"][i]], 1), actions=jnp.asarray(z["b_act"][i])[:, None],
                     rewards=jnp.asarray(z["b_rew"][i])[:, None],
                     discounts=(float(gamma) * (1.0 - jnp.asarray(z["b_done"][i])))[:, None])
        st, info = upd(st, batch)
        np.testing.assert_allclose(float(info["loss"]), z["loss"][i], rtol=2e-4, atol=1e-6)
        st = alg.sync(st, alg.sync_due(st, 50))
        if (i + 1) in set(z["snap_idx"].tolist()):
            ref = _cleanrl_params(z, f"snap{i + 1}")
            jax.tree.map(lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-3, atol=2e-5),
                         st.params, ref)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-3, atol=2e-5),
                 st.params, _cleanrl_params(z, "final"))
    assert n_greedy_checked > 100


def _pomdp_unroll(P, inputs, carry0, resets=None):
    """JAX port of pomdp-baselines Critic_RNN.forward (discrete, LSTM, head qf1), batch-major. Test-only."""
    relu = jax.nn.relu

    def lin(x, n):
        return x @ P[n + ".weight"].T + P[n + ".bias"]

    x = jnp.concatenate([relu(lin(inputs["a"], "action_embedder.fc")), relu(lin(inputs["r"], "reward_embedder.fc")),
                         relu(lin(inputs["o"], "observ_embedder.fc"))], -1)

    def cell(carry, xt):
        c, h = carry
        zz = xt @ P["rnn.weight_ih_l0"].T + P["rnn.bias_ih_l0"] + h @ P["rnn.weight_hh_l0"].T + P["rnn.bias_hh_l0"]
        i, f, g, o = jnp.split(zz, 4, -1)        # torch gate order (i, f, g, o)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return (c, h), h

    carry, hs = jax.lax.scan(cell, carry0, jnp.swapaxes(x, 0, 1))
    hs = jnp.swapaxes(hs, 0, 1)
    j = jnp.concatenate([hs, relu(lin(inputs["o"], "current_shortcut_embedder.fc"))], -1)
    q = relu(lin(j, "qf1.fc0"))
    q = relu(lin(q, "qf1.fc1"))
    return lin(q, "qf1.last_fc"), carry


def test_pomdp_sequence_target_differential():
    z = np.load(os.path.join(FIX, "pomdp_seq_target.npz"))
    with jax.enable_x64(True):
        on = {k[len("online/"):]: jnp.asarray(z[k]) for k in z.files if k.startswith("online/")}
        tg = {k[len("target/"):]: jnp.asarray(z[k]) for k in z.files if k.startswith("target/")}
        tm = lambda a: jnp.swapaxes(jnp.asarray(a), 0, 1)       # time-major -> batch-major
        B, H = z["observs"].shape[1], on["rnn.weight_hh_l0"].shape[1]
        gamma = float(z["gamma"])
        batch = dict(inputs=dict(a=tm(z["actions"]), r=tm(z["rewards"]), o=tm(z["observs"])),
                     actions=tm(z["act_idx"]).astype(jnp.int32), rewards=tm(z["rewards"][1:, :, 0]),
                     discounts=gamma * (1.0 - tm(z["dones"][1:, :, 0])), mask=tm(z["masks"][:, :, 0]),
                     carry0=(jnp.zeros((B, H)), jnp.zeros((B, H))))
        cfg = alg.QConfig(n_actions=4, obs_dim=3, lstm=H, double=True, gamma=gamma, trace_len=z["act_idx"].shape[0])
        (loss, aux), g = jax.value_and_grad(alg.sequence_loss, has_aux=True)(on, tg, batch, cfg, _pomdp_unroll)
        m = tm(z["masks"][:, :, 0])
        np.testing.assert_allclose(np.asarray(aux["q_sa"] * m), tm(z["q1_pred"][:, :, 0]), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(aux["y"] * m), tm(z["q_target"][:, :, 0]), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(float(loss), float(z["qf1_loss"]), rtol=1e-10)
        n = 0
        for k in z.files:
            if k.startswith("grad/"):
                name = k[len("grad/"):]
                np.testing.assert_allclose(np.asarray(g[name]), z[k], rtol=1e-8, atol=1e-11, err_msg=name)
                n += 1
        assert n >= 16
        # the online and target greedy actions differ somewhere, so the double-Q split is actually exercised
        q_on, _ = _pomdp_unroll(on, batch["inputs"], batch["carry0"])
        q_tg, _ = _pomdp_unroll(tg, batch["inputs"], batch["carry0"])
        assert bool(jnp.any(q_on.argmax(-1) != q_tg.argmax(-1)))


def test_cleanrl_env_and_episodic_return_parity():
    """gymnax CartPole-v1 (ours, used by scripts/cartpole_reference.py) vs gymnasium 0.29.1 CartPole-v1 through CleanRL's
    make_env + SyncVectorEnv, on the identical action stream and identical reset states (the reset observation IS the
    state for CartPole). Checks the one-step dynamics from every recorded state, reward, termination vs truncation,
    the stored next observation that the TD target sees, and the episodic-return metric computed our way vs the value
    CleanRL logs. gymnasium integrates in float64 and emits float32; gymnax runs float32, so a free-running float32
    trajectory drifts (1.1e-4 at step 121 of the 500-step episode, measured; it is then amplified without bound by the
    unstable plant) -- the dynamics check therefore re-seeds the state from the recorded observation at every step
    (tolerance 2e-6), and the free-running float32 trajectory is only bounded over the first 100 steps of each episode."""
    import gymnax
    from gymnax.environments.classic_control.cartpole import EnvState

    z = np.load(os.path.join(FIX, "cleanrl_env_traj.npz"))
    env, ep = gymnax.make("CartPole-v1")
    step = jax.jit(lambda s, a: env.step_env(jax.random.PRNGKey(0), s, a, ep))

    def state_from(o, time):
        o = jnp.asarray(o, jnp.float32)
        return EnvState(x=o[0], x_dot=o[1], theta=o[2], theta_dot=o[3], time=jnp.int32(time))

    k, elapsed, ret, rets = 0, 0, 0.0, []
    free = state_from(z["reset_obs"][0], 0)
    free_err = 0.0
    np.testing.assert_array_equal(z["obs"][0], z["reset_obs"][0])
    for t in range(len(z["act"])):
        a = int(z["act"][t])
        o2, s2, r, term, _ = step(state_from(z["obs"][t], elapsed), a)          # one-step map from the recorded state
        trunc = bool(env.is_truncated(s2, ep))
        np.testing.assert_allclose(np.asarray(o2), z["next_obs"][t], rtol=0, atol=2e-6, err_msg=f"t={t}")
        assert float(r) == float(z["rew"][t])
        assert bool(term) == bool(z["term"][t]), t
        assert trunc == bool(z["trunc"][t]), t
        if not bool(z["term"][t]):
            # what our learner stores as the bootstrap observation (info['final_observation'] = o2) equals CleanRL's
            # real_next_obs; on termination CleanRL stores the reset obs, which the (1 - done) factor discards
            np.testing.assert_allclose(np.asarray(o2), z["real_next_obs"][t], rtol=0, atol=2e-6)
        fo, free, *_ = step(free, a)
        if elapsed < 100:                                                    # free-running float32 drift, first 100 steps
            free_err = max(free_err, float(np.abs(np.asarray(fo) - z["next_obs"][t]).max()))
        ret += float(r)
        elapsed += 1
        if bool(term) or trunc:
            rets.append(ret)
            ret, elapsed, k = 0.0, 0, k + 1
            np.testing.assert_array_equal(z["obs"][t + 1] if t + 1 < len(z["obs"]) else z["reset_obs"][k], z["reset_obs"][k])
            free = state_from(z["reset_obs"][k], 0)
    np.testing.assert_array_equal(np.asarray(rets), z["ep_ret"])
    assert int(z["trunc"].sum()) >= 1 and int(z["term"].sum()) >= 5      # both episode ends are exercised
    assert free_err < 1e-4, free_err          # float32 vs float64 drift bound over the first 100 steps of each episode


def test_pomdp_lstm_cell_and_act_with_carry_parity():
    """Our Keras-convention LSTMCell (the cell inside QNetwork / rl_jax_bench.drqn) vs the torch nn.LSTM that
    pomdp-baselines' Critic_RNN uses, float64, from a nonzero carry: whole-sequence unroll, stepwise act-with-carry and
    the final (h, c). Weight map: kernel = W_ih^T, bias = b_ih + b_hh, recurrent = W_hh^T (both use gate order i, f, g, o)."""
    z = np.load(os.path.join(FIX, "pomdp_lstm.npz"))
    with jax.enable_x64(True):
        H = z["h0"].shape[-1]
        cell = alg.LSTMCell(H)
        p = {"params": {"kernel": {"kernel": jnp.asarray(z["rnn/weight_ih_l0"].T),
                                   "bias": jnp.asarray(z["rnn/bias_ih_l0"] + z["rnn/bias_hh_l0"])},
                        "recurrent": {"kernel": jnp.asarray(z["rnn/weight_hh_l0"].T)}}}
        ref_shapes = jax.tree.map(jnp.shape, cell.init(jax.random.PRNGKey(0), (jnp.zeros((1, H)), jnp.zeros((1, H))),
                                                       jnp.zeros((1, z["x"].shape[-1]))))
        assert jax.tree.map(jnp.shape, p) == ref_shapes
        carry0 = (jnp.asarray(z["c0"]), jnp.asarray(z["h0"]))      # ours is (c, h)

        def body(c, x):
            c2, h = cell.apply(p, c, x)
            return c2, h

        (cT, hT), hs = jax.lax.scan(body, carry0, jnp.asarray(z["x"]))
        np.testing.assert_allclose(np.asarray(hs), z["out_seq"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(hs), z["out_step"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(hT), z["hT"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(cT), z["cT"], rtol=1e-10, atol=1e-12)
        # act-with-carry through the full QNetwork path equals the sequence unroll (same params, stepwise vs scan)
        cfg = alg.QConfig(n_actions=3, obs_dim=5, hidden=(7,), lstm=H, double=True, trace_len=6)
        net = alg.make_network(cfg)
        params = alg.init(jax.random.PRNGKey(3), cfg).params
        params = jax.tree.map(lambda a: a.astype(jnp.float64), params)
        xs = jax.random.normal(jax.random.PRNGKey(4), (2, 6, 5), jnp.float64)
        c = (jnp.asarray(z["c0"][:2]), jnp.asarray(z["h0"][:2]))
        resets = jnp.zeros((2, 6), bool).at[1, 3].set(True)          # an act-time reset mid-trace in row 1
        q_seq, cT2 = alg.unroll(net, cfg, params, xs, c, resets)
        q_steps = []
        for t in range(6):
            q, c = alg.act_q(net, cfg, params, xs[:, t], c, resets[:, t])
            q_steps.append(q)
        np.testing.assert_allclose(np.asarray(jnp.stack(q_steps, 1)), np.asarray(q_seq), rtol=1e-10, atol=1e-12)
        jax.tree.map(lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12), c, cT2)
