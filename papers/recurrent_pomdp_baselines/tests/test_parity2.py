"""Parity tests (warm-up, replay buffer).  Fixture tests/fixtures/parity2.npz, recorded by
scripts/official/dump_parity2.py from the UNMODIFIED official code
(@ e7c19c3; no official code is imported here).

1. Warm-up (README).  The official ``Learner.collect_rollouts(.., random_actions=True)`` stores
   ``train_env.action_space.sample()`` RAW: Pendulum U[-2, 2] (about half outside the agent's [-1, 1]), Semi-Circle
   U[-0.1, 0.1]^2.  (a) Our ``env.warmup_action`` draws from the same distribution (two-sample KS, per dimension).
   (b) The executed transition: our ``env.step`` from the official pre-step state with the official stored action
   reproduces the official post-step state, observation and reward (so the wrapper's clip-then-rescale of an
   out-of-range stored action is what we execute too).  RNG streams differ (numpy vs JAX), so (a) is distributional.
2. Replay buffer.  The official RAMEfficient_SeqReplayBuffer (the App. A.1 buffer; ``data.py`` claims equivalence
   with it) at every valid start equals ``data.gather_sequences`` (mask exact; data exact where mask = 1).  The
   vanilla SeqReplayBuffer (the buffer the public configs actually instantiate, ``buffer_type`` unset) is covered by
   test_official_parity.py.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from src import common  # noqa: F401
from src import data as D
from src.envs import Pendulum, SemiCircle

Z = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "parity2.npz"))


def test_warmup_action_distribution():
    for kind, env, lo, hi in (("pendulum", Pendulum(obs_dims=(2,)), -2.0, 2.0), ("semi", SemiCircle(), -0.1, 0.1)):
        off = Z[f"warm/{kind}/act"]
        ours = np.asarray(jax.vmap(env.warmup_action)(jax.random.split(jax.random.PRNGKey(0), 4000)))
        assert off.min() >= lo and off.max() <= hi and ours.min() >= lo and ours.max() <= hi
        for d in range(off.shape[1]):
            p = stats.ks_2samp(off[:, d], ours[:, d]).pvalue
            assert p > 1e-3, (kind, d, p)
    a = Z["warm/pendulum/act"]
    assert 0.4 < np.mean(np.abs(a) > 1.0) < 0.6          # half the stored warm-up torques differ from the executed one


def test_warmup_executed_transition_pendulum():
    env = Pendulum(obs_dims=(2,))
    pre, post, act = Z["warm/pendulum/pre"], Z["warm/pendulum/post"], Z["warm/pendulum/act"]
    st = {"th": jnp.asarray(pre[:, 0], jnp.float32), "thdot": jnp.asarray(pre[:, 1], jnp.float32),
          "t": jnp.zeros(len(pre), jnp.int32)}
    st2, obs, r, _, _ = jax.vmap(lambda s, a: env.step(None, s, None, a))(st, jnp.asarray(act, jnp.float32))
    assert np.allclose(np.asarray(st2["th"]), post[:, 0], atol=2e-5)
    assert np.allclose(np.asarray(st2["thdot"]), post[:, 1], atol=2e-5)
    assert np.allclose(np.asarray(obs), Z["warm/pendulum/obs2"], atol=2e-5)
    assert np.allclose(np.asarray(r), Z["warm/pendulum/rew"][:, 0], rtol=1e-5, atol=1e-4)
    # the official buffer stores the RAW action (|a| up to 2) while the wrapper executed clip(a, -1, 1) * 2
    assert np.abs(act).max() > 1.5


def test_warmup_executed_transition_semicircle():
    env = SemiCircle()
    pre, post, goal, act = Z["warm/semi/pre"], Z["warm/semi/post"], Z["warm/semi/goal"], Z["warm/semi/act"]
    n = len(act)
    t_in = np.arange(n) % env.inner_H                     # inner-episode step index (episodes of 60, 2 per task)
    keep = t_in < env.inner_H - 1                         # skip the boundary step (random internal reset)
    st = {"pos": jnp.asarray(pre[:, :2], jnp.float32), "t": jnp.asarray(t_in, jnp.int32),
          "ep": jnp.asarray((np.arange(n) % env.H) // env.inner_H, jnp.int32)}
    st2, obs, r, _, _ = jax.vmap(lambda s, g, a: env.step(jax.random.PRNGKey(0), s, g, a))(
        st, jnp.asarray(goal, jnp.float32), jnp.asarray(act, jnp.float32))
    assert np.allclose(np.asarray(st2["pos"])[keep], post[keep, :2], atol=1e-6)
    assert np.allclose(np.asarray(obs)[keep], Z["warm/semi/obs2"][keep], atol=1e-6)
    assert np.array_equal(np.asarray(r), Z["warm/semi/rew"][:, 0])
    # executed move is at most 0.01 per axis (stored U[-0.1, 0.1] -> clip -> * 0.1)
    step_len = np.abs(post[keep, :2] - pre[keep, :2])
    assert step_len.max() <= 0.01 + 1e-9


def _rbuf():
    lens = [int(x) for x in Z["rbuf/lens"]]
    rng = np.random.RandomState(3)                        # same generator as dump_parity2.dump_rbuf
    H = max(lens)
    buf = D.init_buffer(len(lens), H, 2, 1)
    for e, n in enumerate(lens):
        obs = rng.normal(size=(n + 1, 2)).astype(np.float32)
        act = rng.uniform(-2, 2, (n, 1)).astype(np.float32)
        rew = rng.normal(size=(n, 1)).astype(np.float32)
        term = np.zeros((n, 1)); term[-1] = float(e % 2)
        pad = lambda x, m: np.concatenate([x, np.zeros((m - x.shape[0],) + x.shape[1:])])  # noqa: E731
        ep = {"obs": pad(obs, H + 1)[None], "act": pad(act, H)[None], "rew": pad(rew, H)[None],
              "term": pad(term, H)[None], "length": np.array([n])}
        buf = D.add_episodes(buf, {k: jnp.asarray(v) for k, v in ep.items()})
    return buf, lens


def test_ram_efficient_buffer_parity():
    buf, lens = _rbuf()
    L = int(Z["rbuf/L"])
    off = np.cumsum([0] + [n + 1 for n in lens])          # the RAM-efficient buffer stores n + 1 slots per episode
    vs = Z["rbuf/valid_starts"]
    ours_n = np.maximum(1, np.asarray(buf["length"]) - L + 1)
    for e in range(len(lens)):
        seg = vs[off[e]:off[e + 1]]
        assert np.sum(seg > 0) == ours_n[e] and np.all(seg[: ours_n[e]] > 0) and np.isclose(seg.sum(), ours_n[e])
    g = Z["rbuf/starts"]
    ep = np.searchsorted(off, g, side="right") - 1
    st = g - off[ep]
    ours = D.gather_sequences(buf, jnp.asarray(ep), jnp.asarray(st), L)
    b = {k[len("rbuf/batch/"):]: Z[k] for k in Z.files if k.startswith("rbuf/batch/")}
    B = len(g)
    ref = {"obs": np.concatenate([b["obs"][:1], b["obs2"]]),
           "act": np.concatenate([np.zeros((1, B, 1)), b["act"]]),
           "rew": np.concatenate([np.zeros((1, B, 1)), b["rew"]]),
           "term": np.concatenate([np.zeros((1, B, 1)), b["term"]]), "mask": b["mask"]}
    m = ref["mask"]
    assert np.array_equal(np.asarray(ours["mask"]), m)
    m1 = np.concatenate([np.ones((1, B, 1)), m])
    for k in ("act", "rew", "term", "obs"):
        assert np.allclose(np.asarray(ours[k]) * m1, ref[k] * m1, atol=1e-6), k
