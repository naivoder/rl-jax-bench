"""Gradient correctness.

1. The hand-written BPTT (custom VJP, model.rnn_seq) equals autodiff of the plain scan (model.rnn_scan) and passes
   jax.test_util.check_grads (finite differences, float32 tolerances) for LSTM and GRU.
2. Finite-difference check of the full recurrent critic loss (TD3 and SAC) w.r.t. encoder + head parameters.
3. the README: the masked loss (M) and its gradient are exactly invariant to padded steps (mask 0) with
   arbitrary content appended after the valid ones; the TD target / twin-min recomputed by hand matches.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from src import common  # noqa: F401
from src.algorithm import Agent
from src.model import Nets, rnn_scan, rnn_seq, torch_orthogonal
from tests.oracle_map import ORACLE_SIZES


def _rnn_inputs(cell, H=6, I=4, T=5, B=2, seed=0):
    G = 4 if cell == "lstm" else 3
    k = jax.random.split(jax.random.PRNGKey(seed), 6)
    return (torch_orthogonal(k[0], (G * H, I)), torch_orthogonal(k[1], (G * H, H)),
            0.1 * jax.random.normal(k[2], (G * H,)), 0.1 * jax.random.normal(k[3], (G * H,)),
            jax.random.normal(k[4], (T, B, I))), jax.random.normal(k[5], (T, B, H))


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_fast_bptt_matches_autodiff(cell):
    args, ct = _rnn_inputs(cell)
    f1 = lambda *a: jnp.sum(rnn_seq(cell, *a, 1) * ct)  # noqa: E731
    f2 = lambda *a: jnp.sum(rnn_scan(cell, *a)[0] * ct)  # noqa: E731
    assert float(f1(*args)) == float(f2(*args))
    g1 = jax.grad(f1, argnums=range(5))(*args)
    g2 = jax.grad(f2, argnums=range(5))(*args)
    for a, b in zip(g1, g2):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=5e-6)


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_fast_bptt_check_grads(cell):
    args, ct = _rnn_inputs(cell, seed=1)
    f = lambda w_hh, x: jnp.sum(jnp.sin(rnn_seq(cell, args[0], w_hh, args[2], args[3], x, 1)) * ct)  # noqa: E731
    check_grads(f, (args[1], args[4]), order=1, modes=("rev",), atol=2e-2, rtol=2e-2, eps=1e-2)


def _agent(algo, arch="separate"):
    cfg = dict(ORACLE_SIZES, algo=algo, cell="lstm", arch=arch)
    nets = Nets(cfg, obs_dim=1, act_dim=1, discrete=False)
    ag = Agent(cfg, nets)
    return ag, ag.init(jax.random.PRNGKey(3))


def _batch(T=6, B=3, seed=0, pad=0):
    r = np.random.RandomState(seed)
    b = {"obs": r.normal(size=(T + 1, B, 1)), "act": np.concatenate([np.zeros((1, B, 1)), r.uniform(-1, 1, (T, B, 1))]),
         "rew": np.concatenate([np.zeros((1, B, 1)), r.normal(size=(T, B, 1))]),
         "term": np.concatenate([np.zeros((1, B, 1)), (r.uniform(size=(T, B, 1)) < .2) * 1.0]),
         "mask": np.ones((T, B, 1))}
    b["mask"][4:, 1] = 0
    if pad:
        for k in ("obs", "act", "rew", "term"):
            b[k] = np.concatenate([b[k], 100 * r.normal(size=(pad,) + b[k].shape[1:])])
        b["mask"] = np.concatenate([b["mask"], np.zeros((pad, B, 1))])
    noise = {"tgt": r.normal(size=(T + 1 + pad, B, 1)), "act": r.normal(size=(T + 1 + pad, B, 1))}
    f = lambda d: {k: jnp.asarray(v, jnp.float32) for k, v in d.items()}  # noqa: E731
    return f(b), f(noise)


@pytest.mark.parametrize("algo", ["td3", "sac"])
def test_critic_loss_finite_differences(algo):
    ag, L = _agent(algo)
    P = L["params"]
    b, noise = _batch()
    ax = P["actor_t"] if algo == "td3" else P["actor"]
    f = jax.jit(lambda c: ag.sep_critic_loss(c, ax, P["critic_t"], b, noise, jnp.float32(0.5))[0])
    g = jax.grad(f)(P["critic"])
    rng = np.random.RandomState(0)
    leaves, tdef = jax.tree_util.tree_flatten(P["critic"])
    gl = jax.tree_util.tree_leaves(g)
    checked = 0
    for li in range(len(leaves)):
        for _ in range(2):
            idx = tuple(rng.randint(s) for s in leaves[li].shape)
            eps = 3e-3

            def at(d):
                ls = list(leaves)
                ls[li] = ls[li].at[idx].add(d)
                return float(f(jax.tree_util.tree_unflatten(tdef, ls)))
            fd = (at(eps) - at(-eps)) / (2 * eps)
            assert abs(fd - float(gl[li][idx])) <= 2e-2 * abs(fd) + 2e-3, (li, idx, fd, float(gl[li][idx]))
            checked += 1
    assert checked >= 2 * len(leaves)


@pytest.mark.parametrize("algo,arch", [("td3", "separate"), ("sac", "separate"), ("td3", "shared"), ("sac", "shared")])
def test_masked_loss_padding_invariance(algo, arch):
    """(M): appending padded steps (mask 0, arbitrary content, after every sequence's end) changes nothing."""
    ag, L = _agent(algo, arch)
    b0, n0 = _batch()
    b1, n1 = _batch(pad=5)
    n1 = {k: v.at[: n0[k].shape[0]].set(n0[k]) for k, v in n1.items()}
    coef = {"leak": jnp.float32(1), "enc": jnp.float32(1)}
    l0, i0 = ag.update(L, b0, n0, coef)
    l1, i1 = ag.update(L, b1, n1, coef)
    for k in ("qf1_loss", "qf2_loss", "policy_loss"):
        np.testing.assert_allclose(float(i0[k]), float(i1[k]), rtol=2e-6, atol=1e-7)
    for a, c in zip(jax.tree_util.tree_leaves(l0["params"]), jax.tree_util.tree_leaves(l1["params"])):
        np.testing.assert_allclose(np.asarray(a), np.asarray(c), rtol=1e-5, atol=1e-7)


def test_td_target_by_hand():
    """TD3 target y = r + gamma (1-d) min(Q1', Q2')(h'_{t+1}, smoothed a'), recomputed outside the loss."""
    ag, L = _agent("td3")
    P = L["params"]
    n = ag.nets
    b, noise = _batch()
    o, a, r, d, m = b["obs"], b["act"], b["rew"], b["term"], b["mask"]
    hqt, _ = n.encode(P["critic_t"]["enc"], a, r, o)
    hpt, _ = n.encode(P["actor_t"]["enc"], a, r, o)
    a2 = jnp.tanh(n.pi_out(P["actor_t"]["pi"], jnp.concatenate([hpt, n.short_o_out(P["actor_t"]["short"], o)], -1)))
    a2 = np.clip(np.asarray(a2) + np.clip(0.2 * np.asarray(noise["tgt"]), -0.5, 0.5), -1, 1)
    q1t, q2t = ag._critic_qs(P["critic_t"], hqt, o, jnp.asarray(a2))
    y = (np.asarray(r) + 0.9 * (1 - np.asarray(d)) * np.minimum(np.asarray(q1t), np.asarray(q2t)))[1:]
    hq, _ = n.encode(P["critic"]["enc"], a, r, o)
    q1, q2 = ag._critic_qs(P["critic"], hq[:-1], o[:-1], a[1:])
    mm = np.asarray(m)
    want = (np.sum(mm * (np.asarray(q1) - y) ** 2) + np.sum(mm * (np.asarray(q2) - y) ** 2)) / mm.sum()
    got = ag.sep_critic_loss(P["critic"], P["actor_t"], P["critic_t"], b, noise, jnp.float32(0))[0]
    np.testing.assert_allclose(float(got), want, rtol=1e-5)
