"""Env dynamics parity ON THE BRANCHES a learning policy reaches (recheck).

The small-scale env fixture (test_oracle_parity.py::test_env_parity) is a random-policy stream: it has 0/1000 Acrobot
terminations, 0/1000 MountainCar goals and 0/1000 Pendulum speed-clip binds, so it cannot see a wrong terminal reward
or a wrong clip/integration order. Here 2000 states per env are sampled ON those branches (Acrobot 848 terminal,
MountainCar 702 goal, Pendulum 316 clip-binding transitions) and one gymnax step_env is compared with gymnasium 0.29.1
(all three envs) and gym 0.15.7 (Acrobot, MountainCar; its Pendulum-v0 integrates theta with the UNCLIPPED thdot and
is used as a power check). Tolerance: |ours - theirs| <= 1e-5 + 1e-5 |theirs|; terminated flags exactly equal.

Run: uv run python -m pytest tests/test_env_branch_parity.py -q
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

Z = np.load(os.path.join(REPO, "tests", "fixtures", "official", "envs_branch.npz"))
ENVS = {"acro": ("Acrobot-v1", ["joint_angle1", "joint_angle2", "velocity_1", "velocity_2"]),
        "mc": ("MountainCar-v0", ["position", "velocity"]),
        "pend": ("Pendulum-v1", ["theta", "theta_dot"])}


def ours(k):
    import gymnax
    env_id, fields = ENVS[k]
    env, prm = gymnax.make(env_id)
    _, s0 = env.reset(jax.random.PRNGKey(0), prm)
    S, A = jnp.asarray(Z[f"in/{k}_S"]), jnp.asarray(Z[f"in/{k}_A"])

    def one(sv, a):
        st = s0.replace(**{f: sv[i] for i, f in enumerate(fields)}, time=0)
        if k == "pend":
            st = st.replace(last_u=jnp.float32(0))
        o, s2, r, term, _ = env.step_env(jax.random.PRNGKey(0), st, a, prm)
        return o, r, term, jnp.stack([getattr(s2, f) for f in fields])

    o, r, term, s2 = jax.jit(jax.vmap(one))(S, A)
    return dict(obs=np.asarray(o, np.float64), r=np.asarray(r, np.float64), term=np.asarray(term, bool),
                s2=np.asarray(s2, np.float64))


def bad(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float((np.abs(a - b) - 1e-5 * np.abs(b)).max())


@pytest.mark.parametrize("k", list(ENVS))
@pytest.mark.parametrize("oracle", ["gymnasium", "gym0157"])
def test_env_branches_match(k, oracle):
    if oracle == "gym0157" and k == "pend":
        pytest.skip("gym 0.15.7 Pendulum-v0 uses the other integration order (power check below)")
    o, p = ours(k), f"oracle_{oracle}/{k}"
    assert bad(o["obs"], Z[p + "_obs"]) <= 1e-5
    assert bad(o["r"], Z[p + "_r"]) <= 1e-5
    if k != "pend":
        assert np.array_equal(o["term"], Z[p + "_term"].astype(bool))
        assert o["term"].sum() > 500   # the branch is exercised
    else:
        assert not o["term"].any() and not Z[p + "_term"].astype(bool).any()


def test_power_pendulum_integration_order():
    """The v0 order (theta advanced with the unclipped thdot) is caught: on the clip-binding rows it differs from
    gymnasium by >> the tolerance, while gymnax matches gymnasium."""
    o = ours("pend")
    S, A = Z["in/pend_S"].astype(np.float64), Z["in/pend_A"][:, 0].astype(np.float64)
    u = np.clip(A, -2, 2)
    td_un = S[:, 1] + (3 * 10 / 2 * np.sin(S[:, 0]) + 3 * u) * 0.05
    th_v0 = S[:, 0] + td_un * 0.05                  # v0: integrate first, clip after
    td_v1 = np.clip(td_un, -8, 8)
    th_v1 = S[:, 0] + td_v1 * 0.05                  # v1: clip first
    bind = np.abs(td_un) > 8
    assert bind.sum() > 200
    gym_th = Z["oracle_gymnasium/pend_s2"][:, 0]
    assert np.abs(th_v1 - gym_th).max() < 1e-5
    assert np.abs(th_v0 - gym_th)[bind].max() > 1e-2            # the mutant fails
    assert np.abs(o["s2"][:, 0] - gym_th).max() < 1e-5           # ours does not
    assert np.abs(Z["oracle_gym0157/pend_s2"][:, 0] - gym_th)[bind].max() > 1e-2   # and v0 really differs


def test_power_acrobot_terminal_reward():
    """A '-1 on the terminal step too' reward (instead of 0) would be caught on the terminal rows."""
    o = ours("acro")
    mut = np.where(o["term"], -1.0, o["r"])
    assert bad(mut, Z["oracle_gymnasium/acro_r"]) > 0.5
    assert bad(o["r"], Z["oracle_gymnasium/acro_r"]) <= 1e-5
