"""Standing rule 2: our JAX Car-Flag reproduces the official env (asym-rlpo@1a065b0 extra-car-flag-v0 + TimeLimit(100))
step by step: 1,000 action sequences from the official start states, float64 replay, tolerance 1e-9."""
import os

import pytest

from _helpers import ORACLE, oracle_mark

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from envs import carflag_reset, carflag_step  # noqa: E402

PATH = os.path.join(ORACLE, "carflag_traces.npz")
pytestmark = oracle_mark(PATH, "scripts/official/export_carflag.py")


def test_carflag_matches_official():
    d = np.load(PATH)
    with jax.enable_x64(True):
        def roll(s0, acts):
            def body(s, a):
                s2, o, r, done = carflag_step(None, s, jnp.maximum(a, 0))
                return s2, (s2, o, r, done)
            return jax.lax.scan(body, s0, acts)[1]
        S, O, R, D = map(np.asarray, jax.jit(jax.vmap(roll))(jnp.asarray(d["s0"]), jnp.asarray(d["act"])))
        st0, ob0 = carflag_reset(jax.random.PRNGKey(0), jnp.float64)
        assert st0.shape == (3,) and float(ob0[1]) == 0.0 and float(ob0[2]) == 0.0
    assert np.allclose(d["o0"][:, 1:], 0.0) and np.allclose(d["o0"][:, 0], d["s0"][:, 0])
    for i in range(len(d["len"])):
        L = d["len"][i] if d["len"][i] > 0 else 100
        assert np.allclose(O[i, :L], d["ob"][i, :L], atol=1e-9), f"obs seq {i}"
        assert np.allclose(S[i, :L], d["st"][i, :L], atol=1e-9), f"state seq {i}"
        assert np.array_equal(R[i, :L], d["rw"][i, :L]), f"reward seq {i}"
        env_done = d["dn"][i, :L].copy()
        if d["len"][i] == 100 and d["rw"][i, L - 1] == 0:
            env_done[-1] = False                   # time limit, not a flag
        assert np.array_equal(D[i, :L], env_done), f"done seq {i}"
