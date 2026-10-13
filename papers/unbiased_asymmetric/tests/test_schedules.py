"""Official bookkeeping (scripts/official/export_schedules.py): avg-100 window, log Dispenser (pre-batch count), target
Dispenser and linear negentropy schedule (post-batch count), replayed through our production `learn`."""
import os

import pytest

from _helpers import ORACLE, oracle_mark

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from algorithm import Episode, entropy_weight, init_run_state, make_update  # noqa: E402
from envs import make_env  # noqa: E402
from model import NetSpec, init_params  # noqa: E402

PATH = os.path.join(ORACLE, "schedules.npz")
pytestmark = oracle_mark(PATH, "scripts/official/export_schedules.py")


def test_bookkeeping_matches_official():
    d = np.load(PATH)
    env = make_env("heavenhell_3")
    spec = NetSpec(env.kind, env.n_actions, env.n_obs, env.n_states, emb=4, hidden=4, head=(4,))
    C = 100
    cfg = dict(gamma=0.99, cap=C, episodes_per_update=2, max_env_steps=int(d["T_MAX"]), target_period=10_000,
               n_log=200, alpha_pi=1e-3, alpha_v=1e-3, adam_eps=1e-4, lambda0=float(d["LAM0"]),
               lambda_final_factor=0.1, negentropy_nsteps=20_000)
    upd = make_update(env, spec, "a2c", cfg)
    pa, pc = init_params(spec, "a2c", jax.random.PRNGKey(0))
    rs = init_run_state(pa, pc, jax.random.PRNGKey(1), cfg)
    learn = jax.jit(upd.learn)
    n = len(d["lam"])
    nt, nl, lam_ours = [], [], []
    for u in range(n):
        L = d["lens"][u]
        rw = np.zeros((2, C), np.float32)
        rw[:, 0] = d["rets"][u]
        eps = Episode(states=jnp.zeros((2, C + 1), jnp.int32), obs=jnp.full((2, C + 1), env.n_obs, jnp.int32),
                      actions=jnp.zeros((2, C + 1), jnp.int32), rewards=jnp.asarray(rw),
                      mask=jnp.asarray(np.arange(C)[None] < L[:, None]), trunc=jnp.asarray(L == C),
                      length=jnp.asarray(L, jnp.int32))
        rs = learn(rs, eps)
        nt.append(int(rs.next_target))
        nl.append(int(rs.next_log))
        lam_ours.append(float(entropy_weight(rs.t_env, cfg["lambda0"], 0.1, 20_000)))
    k = int(rs.log_idx)
    assert k == len(d["log_t"])
    assert np.array_equal(np.asarray(rs.logs["t"][:k]), d["log_t"])
    assert np.allclose(np.asarray(rs.logs["avg100"][:k]), d["log_avg"], atol=1e-5)
    assert np.array_equal(nt, d["next_tgt"]) and np.array_equal(nl, d["next_log"])
    assert np.allclose(lam_ours, d["lam"], rtol=1e-6)
