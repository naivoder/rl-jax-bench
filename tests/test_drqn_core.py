"""Oracle tests for rl_jax_bench.drqn: masked greedy act, masked double-Q / max target,
n-step targets inside a trace, the SMDP discount, and their wiring into sequence_loss (with burn-in and padding masks).

Each feature is compared against a brute-force numpy reference written from the definition, not from the JAX code:
* masked greedy / masked targets: 1000 random (Q, mask) draws with forced ties and all-illegal rows; equal to 1e-6 and
  never an illegal action when a legal one exists;
* n-step: float64, 1e-12, random terminals (discount 0), interior and tail mask zeros, n in {1, 2, 3, 5, L + 3};
* sequence_loss: LSTM, burn-in 2, n = 3, legal masks; aux["y"] equals the numpy target computed from independent
  unrolls; n_step = 1 with an all-legal mask is bit-identical to the plain path (no "legal" key).
CPU, < 30 s.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rl_jax_bench import drqn as alg  # noqa: E402


def _np_masked_argmax(q, legal):
    out = np.empty(q.shape[0], np.int64)
    for i in range(q.shape[0]):
        idx = np.flatnonzero(legal[i])
        if idx.size == 0:
            out[i] = int(np.argmax(q[i]))
        else:
            best = q[i, idx].max()
            out[i] = int(idx[np.flatnonzero(q[i, idx] == best)[0]])
    return out


def _draws(rng, n=1000, A=16):
    q = np.round(rng.normal(size=(n, A)), 1).astype(np.float32)        # rounding forces ties
    legal = rng.random((n, A)) < rng.uniform(0.05, 0.9, size=(n, 1))
    legal[:20] = False                                                  # all-illegal rows: fallback branch
    return q, legal


def test_masked_greedy_oracle():
    rng = np.random.default_rng(0)
    q, legal = _draws(rng)
    got = np.asarray(alg.masked_greedy(jnp.asarray(q), jnp.asarray(legal)))
    want = _np_masked_argmax(q, legal)
    assert np.array_equal(got, want)
    has = legal.any(1)
    assert legal[np.arange(len(got))[has], got[has]].all()             # never illegal when a legal action exists
    assert np.array_equal(np.asarray(alg.masked_greedy(jnp.asarray(q), None)), np.argmax(q, 1))


def test_masked_targets_oracle():
    rng = np.random.default_rng(1)
    q_on, legal = _draws(rng)
    q_tg = rng.normal(size=q_on.shape).astype(np.float32)
    r = rng.normal(size=q_on.shape[0]).astype(np.float32)
    d = (0.9 * (rng.random(q_on.shape[0]) > 0.1)).astype(np.float32)
    a = _np_masked_argmax(q_on, legal)
    want_double = r + d * q_tg[np.arange(len(a)), a]
    vmax = np.array([q_tg[i, legal[i]].max() if legal[i].any() else q_tg[i].max() for i in range(len(r))])
    want_max = r + d * vmax
    args = (jnp.asarray(q_on), jnp.asarray(q_tg), jnp.asarray(r), jnp.asarray(d))
    np.testing.assert_allclose(np.asarray(alg.td_target(*args, True, jnp.asarray(legal))), want_double, atol=1e-6)
    np.testing.assert_allclose(np.asarray(alg.td_target(*args, False, jnp.asarray(legal))), want_max, atol=1e-6)
    # legal=None is the unmasked Eq. 11 / 14
    a0 = np.argmax(q_on, 1)
    np.testing.assert_allclose(np.asarray(alg.td_target(*args, True)), r + d * q_tg[np.arange(len(a0)), a0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(alg.td_target(*args, False)), r + d * q_tg.max(1), atol=1e-6)


def _np_n_step(r, d, v, n, mask):
    B, L = r.shape
    y = np.zeros((B, L))
    for b in range(B):
        for t in range(L):
            g, acc, k = 1.0, 0.0, 0
            while True:
                acc += g * r[b, t + k]
                last = (k + 1 >= n) or (t + k + 1 >= L) or (mask[b, t + k + 1] == 0)
                if last:
                    acc += g * d[b, t + k] * v[b, t + k]
                    break
                g *= d[b, t + k]
                k += 1
            y[b, t] = acc
    return y


def test_n_step_oracle_float64():
    rng = np.random.default_rng(2)
    B, L = 7, 9
    with jax.enable_x64(True):
        for n in (1, 2, 3, 5, L + 3):
            r = rng.normal(size=(B, L))
            d = 0.97 * (rng.random((B, L)) > 0.15)                         # terminals
            v = rng.normal(size=(B, L))
            mask = (rng.random((B, L)) > 0.1).astype(np.float64)
            mask[0, 6:] = 0.0                                              # padded tail
            got = np.asarray(alg.n_step_target(jnp.asarray(r), jnp.asarray(d), jnp.asarray(v), n, jnp.asarray(mask)))
            assert got.dtype == np.float64
            np.testing.assert_allclose(got, _np_n_step(r, d, v, n, mask), rtol=0, atol=1e-12)
        got1 = np.asarray(alg.n_step_target(jnp.asarray(r), jnp.asarray(d), jnp.asarray(v), 1))
        np.testing.assert_allclose(got1, r + d * v, rtol=0, atol=1e-15)


def test_smdp_discount():
    with jax.enable_x64(True):
        dt = np.array([0.5, 1.0, 2.0, 7.25])
        term = np.array([0.0, 0.0, 1.0, 0.0])
        got = np.asarray(alg.smdp_discount(jnp.asarray(dt), 0.99, tau=2.0, terminal=jnp.asarray(term)))
        np.testing.assert_allclose(got, 0.99 ** (dt / 2.0) * (1 - term), rtol=0, atol=1e-15)
    assert abs(float(alg.smdp_discount(jnp.int32(3), 0.9, tau=3.0)) - 0.9) < 1e-7


def _batch(rng, B, L, D, A):
    return dict(
        inputs=jnp.asarray(rng.normal(size=(B, L + 1, D)).astype(np.float32)),
        actions=jnp.asarray(rng.integers(0, A, size=(B, L)).astype(np.int32)),
        rewards=jnp.asarray(rng.normal(size=(B, L)).astype(np.float32)),
        discounts=jnp.asarray((0.95 * (rng.random((B, L)) > 0.1)).astype(np.float32)),
        resets=jnp.asarray(rng.random((B, L + 1)) < 0.1),
        mask=jnp.asarray(np.concatenate([np.ones((B, L - 2)), (rng.random((B, 2)) > 0.5)], 1).astype(np.float32)),
    )


def test_sequence_loss_n_step_legal_burnin():
    rng = np.random.default_rng(3)
    B, L, D, A, b, n = 5, 8, 6, 7, 2, 3
    cfg = alg.QConfig(n_actions=A, obs_dim=D, hidden=(16,), lstm=8, double=True, trace_len=L, burn_in=b, n_step=n)
    st = alg.init(jax.random.PRNGKey(0), cfg)
    tgt = jax.tree.map(lambda x: x + 0.05 * jnp.sin(jnp.arange(x.size, dtype=x.dtype).reshape(x.shape)), st.params)
    batch = _batch(rng, B, L, D, A)
    legal = rng.random((B, L + 1, A)) < 0.6
    batch["legal"] = jnp.asarray(legal)
    _, aux = alg.sequence_loss(st.params, tgt, batch, cfg)

    # independent reference: burn-in carries, unroll both nets, numpy masked double-Q bootstrap + n-step
    net = alg.make_network(cfg)
    c0 = alg.initial_carry(cfg, (B,))
    _, c_on = alg.unroll(net, cfg, st.params, batch["inputs"][:, :b], c0, batch["resets"][:, :b])
    _, c_tg = alg.unroll(net, cfg, tgt, batch["inputs"][:, :b], c0, batch["resets"][:, :b])
    q_on, _ = alg.unroll(net, cfg, st.params, batch["inputs"][:, b:], c_on, batch["resets"][:, b:])
    q_tg, _ = alg.unroll(net, cfg, tgt, batch["inputs"][:, b:], c_tg, batch["resets"][:, b:])
    q_on, q_tg = np.asarray(q_on, np.float64), np.asarray(q_tg, np.float64)
    Lb = L - b
    v = np.zeros((B, Lb))
    for i in range(B):
        a = _np_masked_argmax(q_on[i, 1:].astype(np.float32), legal[i, b + 1:])
        v[i] = q_tg[i, 1:][np.arange(Lb), a]
    want = _np_n_step(np.asarray(batch["rewards"][:, b:], np.float64), np.asarray(batch["discounts"][:, b:], np.float64),
                      v, n, np.asarray(batch["mask"][:, b:]))
    np.testing.assert_allclose(np.asarray(aux["y"]), want, atol=1e-5)

    # n_step = 1 and an all-legal mask reproduce the earlier loss bit for bit
    cfg1 = alg.QConfig(n_actions=A, obs_dim=D, hidden=(16,), lstm=8, double=True, trace_len=L, burn_in=b)
    base = {k: v_ for k, v_ in batch.items() if k != "legal"}
    l0, a0 = alg.sequence_loss(st.params, tgt, base, cfg1)
    l1, a1 = alg.sequence_loss(st.params, tgt, dict(base, legal=jnp.ones((B, L + 1, A), bool)), cfg1)
    assert float(l0) == float(l1) and np.array_equal(np.asarray(a0["y"]), np.asarray(a1["y"]))
    # and the masked loss still differentiates (target is stop-gradient)
    g = jax.grad(lambda p: alg.sequence_loss(p, tgt, batch, cfg)[0])(st.params)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(g))
