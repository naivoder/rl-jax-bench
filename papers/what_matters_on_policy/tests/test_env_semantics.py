"""Environment semantics on the REAL brax-mjx env: time-limit cut is terminal (C22 = False), ended
envs restart from the reset pool, the learner never sees the terminal obs, finished returns are the episode sums,
and GAE masks the bootstrap at every done (terminal OR time limit). CPU, ~1-2 min (mjx compile).

  JAX_PLATFORMS=cpu uv run python -m pytest tests/test_env_semantics.py -q
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402


def test_brax_hopper_time_limit_terminal_and_reset_pool():
    env, spec = D.make_env("hopper")
    N, L, steps = 8, 25, 60                      # short time limit so that cuts happen inside the test
    es = D.env_reset(env, jax.random.split(jax.random.PRNGKey(0), N))
    pool = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(1), N))
    acts = jax.random.uniform(jax.random.PRNGKey(2), (steps, N, spec.act_dim), minval=-1.0, maxval=1.0)
    step = jax.jit(lambda s, a: D.env_step(env, s, a, pool, L))
    raw = jax.jit(jax.vmap(env.step))
    acc = np.zeros(N)
    n_cut = n_term = 0
    for i in range(steps):
        s2_raw = raw(es.bstate, acts[i])          # what brax alone does (no auto-reset wrapper)
        es2, nobs, r, done, fin = step(es, acts[i])
        r, done, fin = np.asarray(r), np.asarray(done), np.asarray(fin)
        np.testing.assert_array_equal(r, np.asarray(s2_raw.reward))
        acc += r
        term = np.asarray(s2_raw.done) > 0.5
        cut = (np.asarray(es.t) + 1) >= L
        np.testing.assert_array_equal(done, term | cut)           # time-limit cut IS a done (C22 = False)
        for j in range(N):
            if done[j]:
                np.testing.assert_array_equal(np.asarray(nobs[j]), np.asarray(pool.obs[j]))   # restart from the pool
                assert int(es2.t[j]) == 0 and float(es2.ep_ret[j]) == 0.0
                np.testing.assert_allclose(fin[j], acc[j], rtol=1e-5, atol=1e-4)            # finished return
                acc[j] = 0.0
                n_cut += int(cut[j] and not term[j])
                n_term += int(term[j])
            else:
                np.testing.assert_array_equal(np.asarray(nobs[j]), np.asarray(s2_raw.obs[j]))
                assert int(es2.t[j]) == int(es.t[j]) + 1 and fin[j] == 0.0
        es = es2
    assert n_cut > 0, "no time-limit cut exercised"
    assert n_term > 0, "no brax termination exercised (hopper should fall under random actions)"


def test_gae_masks_bootstrap_at_done():
    """A done at t (terminal or time-limit) zeroes BOTH the bootstrap gamma*V(s_{t+1}) and the lambda-chain."""
    T, N = 6, 3
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.normal(size=(T + 1, N)).astype(np.float32))
    r = jnp.asarray(rng.normal(size=(T, N)).astype(np.float32))
    d = np.zeros((T, N), bool)
    d[2, 0] = d[5, 1] = True
    tgt, adv = A.gae(v, r, jnp.asarray(d), 0.99, 0.9)
    np.testing.assert_allclose(adv[2, 0], r[2, 0] - v[2, 0], rtol=1e-6)          # no bootstrap, no chain
    np.testing.assert_allclose(adv[5, 1], r[5, 1] - v[5, 1], rtol=1e-6)
    # reference loop
    ref = np.zeros((T, N), np.float32)
    nxt = np.zeros(N, np.float32)
    for t in reversed(range(T)):
        nd = 1.0 - d[t]
        delta = np.asarray(r[t]) + 0.99 * nd * np.asarray(v[t + 1]) - np.asarray(v[t])
        nxt = delta + 0.99 * 0.9 * nd * nxt
        ref[t] = nxt
    np.testing.assert_allclose(np.asarray(adv), ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(tgt), ref + np.asarray(v[:-1]), rtol=1e-5, atol=1e-6)
