"""Standing rule 2: our JAX Cleaner reproduces the official env (asym-rlpo@1a065b0 extra-cleaner-v0 + TimeLimit(100))
exactly: 200 random joint-action sequences of 100 steps from reset; states (845-d), observations (54-d), rewards."""
import os

import pytest

from _helpers import ORACLE, oracle_mark

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from envs import make_env  # noqa: E402

PATH = os.path.join(ORACLE, "cleaner_traces.npz")
pytestmark = oracle_mark(PATH, "scripts/official/export_cleaner.py")


def test_cleaner_matches_official():
    d = np.load(PATH)
    env = make_env("cleaner")
    s0, o0 = env.reset(jax.random.PRNGKey(0))
    assert np.array_equal(np.asarray(s0), d["s0"][0]) and np.array_equal(np.asarray(o0), d["o0"][0])

    def roll(acts):
        def body(s, a):
            s2, o, r, done = env.step(None, s, a)
            return s2, (s2, o, r)
        return jax.lax.scan(body, s0, acts)[1]
    S, O, R = map(np.asarray, jax.jit(jax.vmap(roll))(jnp.asarray(d["act"], jnp.int32)))
    assert np.array_equal(S, d["st"]) and np.array_equal(O, d["ob"]) and np.array_equal(R, d["rw"])
