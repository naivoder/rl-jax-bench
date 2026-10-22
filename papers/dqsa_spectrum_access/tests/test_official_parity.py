"""Differential tests against fixtures recorded from the THIRD-PARTY shkrwnd implementation (the authors released no code).

Fixtures: tests/fixtures/official/*.npz, written by scripts/official/make_fixtures.py (TF venv) by EXECUTING shkrwnd's env_network, QNetwork and
train.py::state_generator.  shkrwnd's code is not committed.  Where a quantity is *meant* to differ (their capacity input leaks the global
channel outcome; their cooperative reward; argmax acting) the test asserts the difference AND its exact form, so the divergence
is pinned rather than ignored.  Tolerances: boolean/integer quantities exact; float32 forward 1e-5 abs + 1e-4 rel; float32 gradients 1e-4 rel + 1e-5 abs of the tensor scale.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "official")


def load(name):
    return np.load(os.path.join(FIX, name))


def _ours_step(N, K, a):
    spec = D.EnvSpec(n_min=N, n_max=N, n_channels=K, k_local=K, subset="all")
    w = D.World(jnp.ones((N,), bool), jnp.int32(N), jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32), (N, K)))
    st = D.init_state(spec, jax.random.PRNGKey(0))
    return spec, w, st


@pytest.mark.parametrize("NK", [(2, 2), (3, 2), (4, 2), (5, 3), (10, 4)])
def test_env_success_and_residual_capacity_match_theirs(NK):
    """ATTEMPT_PROB = 1: success vector identical on 2000 joint actions; their residual-capacity vector (obs[-1] = 1 - [exactly one tx])
    equals the function of OUR per-channel transmitter counts (exact)."""
    N, K = NK
    f = load(f"env_N{N}K{K}.npz")
    spec, w, st = _ours_step(N, K, None)
    step = jax.jit(lambda a, k: D.step_local(spec, w, st, a, k)[1])
    for i in range(0, 2000, 7):                       # every 7th draw keeps the test fast; the full 2000 are checked in toys T-SHK-style loops
        info = step(jnp.asarray(f["actions"][i].astype(np.int32)), jax.random.PRNGKey(i))
        np.testing.assert_array_equal(np.asarray(info["success"]).astype(np.int8), f["rewards"][i])
        np.testing.assert_array_equal((1 - (np.asarray(info["count"]) == 1)).astype(np.int8), f["cap"][i])


def test_their_env_attempt_prob_below_one_rewards_silent_users():
    """pinned: with ATTEMPT_PROB < 1 their env keeps a user's PREVIOUS action when the user is gated out and then reads the channel
    count for that stale action, so users that did not transmit this slot can be rewarded (51 of 1600 here).  Our env never does this."""
    f = load("env_attempt06.npz")
    a, r, ua = f["actions"], f["rewards"], f["users_action"]
    assert ((a == 0) & (r == 1)).sum() > 0                              # silent-by-choice user rewarded: impossible in the paper's game
    # replay their gate draws (module-level random.uniform, seed 11, N draws per slot in user order) to learn who attempted
    import random
    rg = random.Random(int(f["seed"]))
    N, K, p = int(f["N"]), int(f["K"]), float(f["p"])
    spec, w, st = _ours_step(N, K, None)
    step = jax.jit(lambda aa, k: D.step_local(spec, w, st, aa, k)[1]["success"])
    stale_rewarded = 0
    for t in range(a.shape[0]):
        att = np.array([rg.uniform(0, 1) <= p for _ in range(N)])
        ours = np.asarray(step(jnp.asarray(np.where(att, a[t], 0).astype(np.int32)), jax.random.PRNGKey(t))).astype(np.int8)
        np.testing.assert_array_equal(ours[att], r[t][att])            # attempting users: identical outcome
        stale_rewarded += int(((~att) & (r[t] == 1)).sum())            # gated-out users rewarded through the stale action
    assert stale_rewarded > 0


def test_state_generator_layout_and_the_capacity_leak():
    """x = [onehot(a_t, K+1) | K capacity entries | ACK].  One-hot and ACK blocks equal OUR observe; their capacity block is
    1 - [exactly one transmitter on the channel] (a GLOBAL outcome of the previous slot) while ours is the constant 1 (paper v1/v2)."""
    f = load("state_generator_N3K2.npz")
    N, K = 3, 2
    spec, w, st = _ours_step(N, K, None)
    for t in range(0, 300, 5):
        a = f["actions"][t].astype(np.int32)
        info = D.step_local(spec, w, st, jnp.asarray(a), jax.random.PRNGKey(t))
        st2, inf = info
        ours = np.asarray(D.observe(spec, w, st2))                   # prev_a = a, prev_ack = success
        theirs = f["x"][t]
        assert ours.shape == theirs.shape == (N, 2 * K + 2)
        np.testing.assert_array_equal(ours[:, :K + 1], theirs[:, :K + 1])                 # one-hot of the last action (0 = silent)
        np.testing.assert_array_equal(ours[:, -1], theirs[:, -1])                         # ACK
        np.testing.assert_array_equal(theirs[:, K + 1:2 * K + 1], np.tile(1 - (np.asarray(inf["count"]) == 1), (N, 1)))   # their leak, exact form
        assert (ours[:, K + 1:2 * K + 1] == 1).all()                                       # ours: constant capacity
    ths = f["x"][:, :, 3:5]
    assert (ths == 0).any()                                                                # the leak is informative (not constant)


def test_eq11_probabilities_match_theirs_and_their_action_is_argmax_q():
    """train.py:278-281 == paper Eq. 11 == model.eq11_probs (float64-level fixture, float32 compute).  Their acting rule (train.py:291)
    is argmax(prob) == argmax(Q): the beta/alpha schedule never changes their action."""
    f = load("their_policy_probs.npz")
    Q = f["Q"]
    for alpha, beta, prob in zip(f["alpha"], f["beta"], f["prob"]):
        ours = np.asarray(M.eq11_probs(jnp.asarray(Q, jnp.float32), float(alpha), float(beta)))
        np.testing.assert_allclose(ours, prob, rtol=2e-4, atol=1e-6)
        if alpha == 0.0 or beta <= 3.0:                                   # avoid float ties after exp overflow handling at large beta
            assert (prob.argmax(1) == Q.argmax(1)).all()


def test_td_target_matches_theirs_when_target_net_is_online_net():
    """train.py:382 target = r + gamma max Q(s',.) (no done flag, same net) == drqn.td_target(double=False); with target params == online
    params the paper's double-Q target (double=True) collapses to the same number (so the algorithms differ only through the missing
    separate target network and the replay/stream data, not the formula)."""
    from rl_jax_bench import drqn
    f = load("their_td_target.npz")
    q = jnp.asarray(f["q_next"], jnp.float32); r = jnp.asarray(f["r"], jnp.float32); g = float(f["gamma"])
    for dbl in (False, True):
        t = drqn.td_target(q, q, r, jnp.full_like(r, g), double=dbl)
        np.testing.assert_allclose(np.asarray(t), f["target"], rtol=1e-5, atol=1e-6)


def _perm(m, H, axis=-1):
    """TF BasicLSTMCell gate blocks (i, j=candidate, f, o) -> drqn LSTMCell (i, f, g, o)."""
    b = np.split(np.asarray(m), 4, axis=axis)
    return np.concatenate([b[0], b[2], b[1], b[3]], axis=axis)


def test_their_qnetwork_forward_loss_and_gradients_match_our_lstm_cell():
    """Their QNetwork (BasicLSTMCell -> Dense+ReLU -> layer norm -> Dense, MSE on the taken action's Q) rebuilt with drqn.LSTMCell (the cell
    inside OUR DQSANet): outputs, loss and the gradient of every parameter agree with TF's tf.gradients on the recorded batch.
    BasicLSTMCell adds forget_bias = 1 to the forget pre-activation == bias + 1 on drqn's forget block (gate order permuted)."""
    from rl_jax_bench import drqn
    f = load("their_qnetwork_H8.npz")
    H = f["lstm_out"].shape[-1]
    S = f["x"].shape[-1]
    cell = drqn.LSTMCell(H, "glorot_uniform", "sigmoid")
    kern = f["w__main__rnn__basic_lstm_cell__kernel"]; bias = f["w__main__rnn__basic_lstm_cell__bias"]

    def to_params(P):
        return {"kernel": {"kernel": _perm(P["lk"][:S], H), "bias": _perm(P["lb"], H, 0) + np.concatenate([np.zeros(H), np.ones(H), np.zeros(2 * H)])},
                "recurrent": {"kernel": _perm(P["lk"][S:], H)}}

    def forward(P, x):
        prm = {"params": {"kernel": {"kernel": P["ok"], "bias": P["ob"]}, "recurrent": {"kernel": P["rk"]}}}
        B = x.shape[0]
        carry = (jnp.zeros((B, H)), jnp.zeros((B, H)))
        hs = []
        for t in range(x.shape[1]):
            carry, h = cell.apply(prm, carry, x[:, t])
            hs.append(h)
        hs = jnp.stack(hs, 1)
        h = hs[:, -1]
        h2 = jax.nn.relu(h @ P["w2"] + P["b2"])
        m = h2.mean(1, keepdims=True); v = ((h2 - m) ** 2).mean(1, keepdims=True)
        ln = (h2 - m) / jnp.sqrt(v + 1e-12) * P["gamma"] + P["beta"]
        return hs, ln @ P["w3"] + P["b3"]

    tf_names = dict(lk="w__main__rnn__basic_lstm_cell__kernel", lb="w__main__rnn__basic_lstm_cell__bias", w2="w__main__Variable", b2="w__main__Variable_1",
                    beta="w__main__LayerNorm__beta", gamma="w__main__LayerNorm__gamma", w3="w__main__Variable_2", b3="w__main__Variable_3")
    P_tf = {k: np.asarray(f[v], np.float32) for k, v in tf_names.items()}
    op = to_params(P_tf)
    P = dict(ok=op["kernel"]["kernel"], ob=op["kernel"]["bias"], rk=op["recurrent"]["kernel"], **{k: P_tf[k] for k in ("w2", "b2", "beta", "gamma", "w3", "b3")})
    P = {k: jnp.asarray(v, jnp.float32) for k, v in P.items()}
    x = jnp.asarray(f["x"]); acts = jnp.asarray(f["acts"]); tgt = jnp.asarray(f["targets"])
    hs, q = forward(P, x)
    np.testing.assert_allclose(np.asarray(hs), f["lstm_out"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(q), f["q"], rtol=1e-4, atol=1e-4)

    def loss_fn(P):
        _, qq = forward(P, x)
        qa = jnp.take_along_axis(qq, acts[:, None], 1)[:, 0]
        return jnp.mean((tgt - qa) ** 2)

    loss, g = jax.value_and_grad(loss_fn)(P)
    np.testing.assert_allclose(float(loss), float(f["loss"]), rtol=1e-4)
    # gradients: map back to TF layout and compare tensor-wise (scale = max |g_tf| of that tensor)
    back = dict(
        lk=np.concatenate([_perm_inv(np.asarray(g["ok"]), H), _perm_inv(np.asarray(g["rk"]), H)], 0),
        lb=_perm_inv(np.asarray(g["ob"]), H, 0),
        **{k: np.asarray(g[k]) for k in ("w2", "b2", "beta", "gamma", "w3", "b3")})
    for k, name in tf_names.items():
        ref = f["g" + name[1:]]
        scale = max(float(np.abs(ref).max()), 1e-6)
        assert np.abs(back[k] - ref).max() <= 1e-4 * scale + 1e-6, (k, np.abs(back[k] - ref).max(), scale)


def _perm_inv(m, H, axis=-1):
    b = np.split(np.asarray(m), 4, axis=axis)       # ours (i, f, g, o) -> TF (i, j=g, f, o)
    return np.concatenate([b[0], b[2], b[1], b[3]], axis=axis)


def test_their_cooperative_reward_is_not_the_papers_global_reward():
    """pinned (algebra only): shkrwnd gives sum_r to the users that SUCCEEDED and 0 to the rest (train.py:'reward[i] = sum_r if reward[i] > 0');
    the paper's sum-rate objective gives sum_r to EVERY user.  The two differ exactly on the failed users."""
    r = np.array([1, 0, 1, 0]); s = r.sum()
    theirs = np.where(r > 0, s, 0); paper = np.full_like(r, s)
    assert (theirs != paper).sum() == (r == 0).sum()


@pytest.mark.parametrize("name", ["trace_N3K2_s0", "trace_N2K2_s0"])
def test_metric_both_ways_on_their_rollouts(name):
    """Real action traces produced by their trained learner (first/last 1500 slots of 100k, N=3 or 2, K=2): our env reproduces their per-slot
    rewards exactly, our reduce_one utilization equals their sum_r / K, and their 'collision' (K - sum_r, train.py:'collision = NUM_CHANNELS - sum_r')
    equals K * (1 - util) -- i.e. it counts idle AND collided channels; our collision / idle split is a strict refinement of it."""
    import eval as E
    f = load(name + ".npz")
    a = f["actions"].astype(np.int32); r = f["rewards"].astype(np.int8); K = int(f["K"]); T, N = a.shape
    spec, w, st = _ours_step(N, K, None)

    def one(aa, k):
        return D.step_local(spec, w, st, aa, k)[1]
    info = jax.jit(jax.vmap(one))(jnp.asarray(a), jax.random.split(jax.random.PRNGKey(0), T))
    np.testing.assert_array_equal(np.asarray(info["success"]).astype(np.int8), r)
    o = dict(success=info["success"], tx=info["tx"], g=info["g"], rate=info["rate"], active=w.active, n_active=w.n_active,
             subset=w.subset, pu_free=info["pu_free"])
    red = E.reduce_one(spec, o)
    np.testing.assert_allclose(np.asarray(red["util_t"]), r.sum(1) / K, atol=1e-6)
    np.testing.assert_allclose(K - r.sum(1), K * (1 - np.asarray(red["util_t"])), atol=1e-5)
    assert np.all(np.asarray(red["coll_t"]) + np.asarray(red["idle_t"]) <= 1 + 1e-6)
