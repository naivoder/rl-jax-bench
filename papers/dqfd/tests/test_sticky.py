"""The sticky-action env flag (src/envs.py).  CPU, < 30 s.
(1) sticky = 0 leaves the state pytree and the trajectory untouched (the env of the main runs, bit for bit);
(2) sticky = 1 always repeats the previous effective action (initially 0);
(3) sticky = 0.1 repeats about 10 % of the time (original MinAtar semantics: the repeated action is the last EFFECTIVE one)."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs import MinAtar


def _roll(env, actions, key):
    obs, st = env.reset(key)
    out = []
    for i, a in enumerate(actions):
        obs, st, r, term, trunc = env.step(jax.random.fold_in(key, i), st, jnp.int32(a))
        out.append((np.asarray(obs), float(r)))
        if bool(term):
            break
    return out, st


def test_sticky0_is_unchanged():
    k = jax.random.PRNGKey(0)
    e0, e1 = MinAtar("Breakout"), MinAtar("Breakout", sticky_action_prob=0.0)
    acts = np.random.RandomState(0).randint(0, 3, 40)
    (a, _), (b, _) = _roll(e0, acts, k), _roll(e1, acts, k)
    assert len(a) == len(b) and all((x[0] == y[0]).all() and x[1] == y[1] for x, y in zip(a, b))


def test_sticky1_repeats_initial_noop():
    k = jax.random.PRNGKey(1)
    e0, es = MinAtar("Breakout"), MinAtar("Breakout", sticky_action_prob=1.0)
    (a, _), (b, _) = _roll(e0, [0] * 5, k), _roll(es, [2] * 5, k)          # sticky=1 ignores the chosen action 2 and keeps 0
    assert all((x[0] == y[0]).all() for x, y in zip(a, b))


def test_sticky_rate():
    env = MinAtar("Breakout", sticky_action_prob=0.1)
    obs, st = env.reset(jax.random.PRNGKey(2))
    n, rep, last = 0, 0, 0
    key = jax.random.PRNGKey(3)
    rs = np.random.RandomState(1)
    for i in range(300):
        a = int(rs.randint(0, 3))
        _, st2, _, term, trunc = env.step(jax.random.fold_in(key, i), st, jnp.int32(a))
        eff = int(st2[1])
        if a != last:
            n += 1; rep += int(eff == last)
        st, last = st2, eff
        if bool(term):
            obs, st = env.reset(jax.random.fold_in(key, 10_000 + i)); last = 0
    assert 0.03 < rep / n < 0.2, (rep, n)
