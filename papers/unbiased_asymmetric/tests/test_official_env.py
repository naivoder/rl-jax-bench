"""Standing rule 2: our JAX categorical envs reproduce the official gym-pomdps@b52e520 + FlatPaddingWrapper +
TimeLimit(100) streams step by step (1,000 random action sequences per env, forced start states; the four files
are deterministic in T and O, so identical actions must give identical (state, obs, reward, done) streams).
Oracle data: scripts/official/export_env_traces.py (official venv)."""
import os

import pytest

from _helpers import ORACLE, oracle_mark

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from data import load_pomdp  # noqa: E402
from envs import cat_step, make_cat_env  # noqa: E402

PATH = os.path.join(ORACLE, "env_traces.npz")
pytestmark = oracle_mark(PATH, "scripts/official/export_env_traces.py")


@pytest.mark.parametrize("name", ["heavenhell_3", "heavenhell_4", "shopping_5", "shopping_6"])
def test_env_matches_official(name):
    d = np.load(PATH)
    g = lambda k: d[f"{name}/{k}"]  # noqa: E731
    p = load_pomdp(name)
    env = make_cat_env(p)
    act, st, ob, rw, dn, ln = g("act"), g("st"), g("ob"), g("rw"), g("dn"), g("len")
    assert np.all(ob[:, 0] == p.n_obs), "first observation must be the extra reset observation"

    def roll(s0, acts, key):
        def body(c, x):
            s, alive = c
            a, k = x
            s2, o, r, done = cat_step(env, k, s, jnp.maximum(a, 0))
            return (s2, alive & ~done), (s2, o, r, done)
        _, ys = jax.lax.scan(body, (s0, True), (acts, jax.random.split(key, acts.shape[0])))
        return ys

    S2, O, R, D = jax.jit(jax.vmap(roll))(jnp.asarray(g("s0"), jnp.int32), jnp.asarray(act, jnp.int32),
                                          jax.random.split(jax.random.PRNGKey(0), act.shape[0]))
    S2, O, R, D = map(np.asarray, (S2, O, R, D))
    for i in range(act.shape[0]):
        L = ln[i]
        assert np.all(O[i, :L] == ob[i, 1:L + 1]), f"obs mismatch seq {i}"
        assert np.allclose(R[i, :L], rw[i, :L]), f"reward mismatch seq {i}"
        env_done = dn[i, :L].copy()
        if L == 100 and st[i, L] != -1:    # time-limit truncation: the official TimeLimit sets done, the env did not
            env_done[-1] = False
        assert np.all(D[i, :L] == env_done), f"done mismatch seq {i}"
        nonterm = st[i, 1:L + 1] >= 0     # official state is -1 after a terminal step
        assert np.all(S2[i, :L][nonterm] == st[i, 1:L + 1][nonterm]), f"state mismatch seq {i}"


@pytest.mark.parametrize("name", ["heavenhell_3", "heavenhell_4", "shopping_5", "shopping_6"])
def test_parsed_arrays_match_rl_parsers(name):
    d = np.load(PATH)
    p = load_pomdp(name)
    for k, ours in (("T", p.T), ("O", p.O), ("R", p.R), ("D", p.D), ("start", p.start)):
        assert np.array_equal(np.asarray(ours, float), np.asarray(d[f"{name}/{k}"], float)), k
