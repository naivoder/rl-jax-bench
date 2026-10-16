"""Smoke test (< 30 s on CPU): shapes, one full training iteration for every arm on a tiny pure-JAX env with the
brax-env interface, pairing (identical configs -> identical lanes), the one-factor config diff, NaN freeze,
frozen eval statistics.   JAX_PLATFORMS=cpu uv run python -m pytest tests/test_smoke.py -q
"""
import os
import sys
from typing import NamedTuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import configs as C  # noqa: E402
from src import data as D  # noqa: E402
from src import model as M  # noqa: E402
from src.train import lane_cfg, seed_keys, eval_points  # noqa: E402


class S(NamedTuple):
    q: jax.Array
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


class PointEnv:
    """2-d point mass: action = acceleration in [-1,1]^2, reward = -|q|, episode ends when |q| > 3."""
    observation_size, action_size = 4, 2

    def reset(self, key):
        q = jax.random.uniform(key, (4,), minval=-1, maxval=1)
        return S(q, q, jnp.zeros(()), jnp.zeros(()))

    def step(self, s, a):
        pos, vel = s.q[:2], s.q[2:] + 0.1 * a
        q = jnp.concatenate([pos + 0.1 * vel, vel])
        return S(q, q, -jnp.linalg.norm(q[:2]), (jnp.linalg.norm(q[:2]) > 3).astype(jnp.float32))


ST = A.Static(num_envs=4, unroll=8, epochs=2, minibatch=8, episode_length=20)
SPEC = D.EnvSpec("point", 4, 2, 20)


def _lanes(arms, seeds=(0,)):
    cfg = C.load()
    static = dict(cfg["static"])
    env = PointEnv()
    root = jax.random.PRNGKey(0)
    vmax = max(C.resolve(cfg, a)["v_width"] for a in arms)
    ls, cf, ks = [], [], []
    for a in arms:
        r = C.resolve(cfg, a)
        for s in seeds:
            k = seed_keys(root, s)
            p = M.init_params(k["init"], 4, 2, v_width=r["v_width"], v_width_max=vmax, pi_last_gain=r["pi_last_gain"])
            ls.append(A.init_lane(env, SPEC, p, ST, k["env"]))
            cf.append(lane_cfg(r, static))
            ks.append({x: k[x] for x in ("env", "act", "mb")})
    stack = lambda xs: jax.tree.map(lambda *z: jnp.stack(z), *xs)
    return env, stack(ls), stack(cf), stack(ks)


def test_one_iteration_all_arms_and_pairing():
    cfg = C.load()
    arms = C.arm_list(cfg, "small") + ["REC"]               # REC twice: pairing check
    env, ls, cf, ks = _lanes(arms)
    it = jax.jit(jax.vmap(lambda l, c, k: A.iteration(env, l, c, ST, k)))
    ls2, aux = it(ls, cf, ks)
    L = len(arms)
    assert ls2.it.shape == (L,) and int(ls2.it[0]) == 1
    assert ls2.obs.shape == (L, 4, 4)
    assert aux["v_loss"].shape == (L,)
    assert np.all(np.isfinite(np.asarray(aux["v_loss"]))) and not np.asarray(ls2.bad).any()
    # one optimiser step per minibatch per epoch
    assert int(ls2.opt.t[0]) == ST.epochs * (ST.num_envs * ST.unroll // ST.minibatch)
    # identical configs -> identical lanes (the arm never enters a key)
    for x in jax.tree.leaves(ls2.params):
        np.testing.assert_array_equal(np.asarray(x[0]), np.asarray(x[-1]))
    # the obs stats saw exactly T*N observations
    assert float(ls2.obs_ms.count[0, 0]) == ST.num_envs * ST.unroll
    # value-norm stats: updated once per epoch under recompute (REC), once per iteration for BASE
    i_base = arms.index("BASE")
    assert float(ls2.val_ms.count[0, 0]) == ST.epochs * ST.num_envs * ST.unroll
    assert float(ls2.val_ms.count[i_base, 0]) == ST.num_envs * ST.unroll
    # value-norm off: stats untouched
    assert float(ls2.val_ms.count[arms.index("A7_valuenorm_none"), 0]) == 0
    assert float(ls2.obs_ms.count[arms.index("A1_obsnorm_none"), 0]) == 0


def test_one_factor_config_diff():
    cfg = C.load()
    rec = C.resolve(cfg, "REC")
    for a in cfg["primary"] + cfg["secondary"] + cfg["diagnostic"]:
        if a == "BASE":
            continue
        if a == "X3_base_head":
            b, x = C.resolve(cfg, "BASE"), C.resolve(cfg, a)
            assert [k for k in b if b[k] != x[k]] == ["logp_presquash"]
            continue
        r = C.resolve(cfg, a)
        ref = C.resolve(cfg, "A3a_shared_bc1") if a in ("A3b_shared_bc0p1", "A3c_shared_bc10") else rec
        diff = [k for k in ref if ref[k] != r[k]]       # A3b/A3c: baseline cost varied around the shared arm
        assert len(diff) == 1, (a, diff)
    # cold-review arms 
    assert [k for k in rec if rec[k] != C.resolve(cfg, "A3x_sep_v64")[k]] == ["v_width"]
    assert [k for k in rec if rec[k] != C.resolve(cfg, "X5_rec_lr_to0_1M")[k]] == ["lr_horizon_frames"]
    a5 = C.resolve(cfg, "A5a_lr_1em4")
    assert [k for k in a5 if a5[k] != C.resolve(cfg, "X6_lr1em4_to0_1M")[k]] == ["lr_horizon_frames"]


def test_shared_arm_effective_capacity():
    """: A3a (shared 64x2 trunk + linear value head) differs from REC in C47 AND value width (256 -> effective 64).
    Against the control A3x_sep_v64 (separate 64x2 value MLP) the EFFECTIVE configs differ only in ``shared``."""
    cfg = C.load()
    st = cfg["static"]
    eff = lambda r: {**r, "v_width": st["pi_width"] if r["shared"] else r["v_width"]}
    a3a, ctl, rec = (eff(C.resolve(cfg, a)) for a in ("A3a_shared_bc1", "A3x_sep_v64", "REC"))
    assert [k for k in a3a if a3a[k] != ctl[k]] == ["shared"]
    assert sorted(k for k in a3a if a3a[k] != rec[k]) == ["shared", "v_width"]


def test_eval_frozen_stats_and_schedule():
    env, ls, cf, ks = _lanes(["REC"])
    one = jax.tree.map(lambda x: x[0], ls)
    c1 = jax.tree.map(lambda x: x[0], cf)
    before = jax.tree.map(np.asarray, one.obs_ms)
    ret, ln = jax.jit(lambda p, o, c: A.evaluate(env, p, o, c, ST, jax.random.PRNGKey(1), 6))(one.params, one.obs_ms, c1)
    assert ret.shape == (6,) and int(ln.max()) <= ST.episode_length
    for x, y in zip(jax.tree.leaves(before), jax.tree.leaves(one.obs_ms)):
        np.testing.assert_array_equal(x, np.asarray(y))
    static = dict(C.load()["static"])
    pts = eval_points(static, 489)
    assert pts == [0, 49, 98, 147, 196, 245, 293, 342, 391, 440, 489]


def test_nan_lane_is_frozen_not_dropped():
    env, ls, cf, ks = _lanes(["REC", "REC"])
    cf = cf._replace(lr=cf.lr.at[1].set(jnp.nan))
    ls2, _ = jax.jit(jax.vmap(lambda l, c, k: A.iteration(env, l, c, ST, k)))(ls, cf, ks)
    assert bool(ls2.bad[1]) and not bool(ls2.bad[0])
    for x, y in zip(jax.tree.leaves(ls2.params), jax.tree.leaves(ls.params)):
        np.testing.assert_array_equal(np.asarray(x[1]), np.asarray(y[1]))       # frozen at the last finite params


def test_obs_stat_ordering():
    """Pins the official ordering (learner.training_step: update_observation_normalization_statistics BEFORE
    ppo_training_step; behaviour logits come from the actor's pre-update inference variables; README):
    behaviour mu/std are computed with the PRE-update obs stats, the loss with the POST-update stats. So the ratio
    on epoch 0's first minibatch is != 1 when obs-norm is on, and exactly 1 when it is off (A1)."""
    env, ls, cf, ks = _lanes(["REC", "A1_obsnorm_none"])
    one = lambda t, i: jax.tree.map(lambda x: x[i], t)
    for i, on in ((0, True), (1, False)):
        l, c, k = one(ls, i), one(cf, i), one(ks, i)
        # warm the stats once so that pre/post-update stats differ from the init (mean 0, std 1)
        l, _ = A.iteration(env, l, c, ST, k)
        pool = jax.vmap(env.reset)(jax.random.split(jax.random.fold_in(k["env"], 99), ST.num_envs))
        l2, batch, _, _ = A.collect(env, l, c, ST, jax.random.PRNGKey(7), pool)
        obs_n_pre = D.normalize_obs(l.obs_ms, batch.obs[:-1], c.obs_norm, c.obs_clip)
        mu_pre, _, _ = M.forward(l.params, obs_n_pre, c.shared)
        np.testing.assert_allclose(np.asarray(batch.mu_b), np.asarray(mu_pre), rtol=1e-6, atol=1e-6)
        l3, aux = A.update(l2, batch, c, ST, jax.random.PRNGKey(8))
        post = D.meanstd_update(l.obs_ms, batch.obs[:-1]) if on else l.obs_ms
        for x, y in zip(jax.tree.leaves(post), jax.tree.leaves(l3.obs_ms)):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6)
        kl0 = float(aux["approx_kl_mb0"])
        if on:
            assert abs(kl0) > 1e-7, kl0          # pre/post stats differ -> ratio != 1 before any step
        else:
            assert kl0 == 0.0, kl0               # no obs-norm -> ratio exactly 1 on the first minibatch


def test_lr_horizon():
    """: the default lr horizon is the training length (decay to 0 when scoring stops, paper App. B.5 / C31);
    X5/X6 are then effectively REC/A5a; the lr2M_* arms and a static 2M override reproduce the pre-fix  schedule."""
    cfg = C.load()
    st = dict(cfg["static"])
    assert st["lr_horizon_frames"] == st["total_env_steps"] == 1001472
    steps = lambda arm, s=st: C.lr_horizon_steps(s, C.resolve(cfg, arm).get("lr_horizon_frames"))
    assert steps("REC") == steps("X5_rec_lr_to0_1M") == 489 * 10 * 32
    assert steps("A5a_lr_1em4") == steps("X6_lr1em4_to0_1M")
    assert steps("lr2M_rec") == steps("lr2M_A5a") == 977 * 10 * 32
    pre = dict(st, lr_horizon_frames=2000000)                 # the pre-fix static block
    assert steps("REC", pre) == steps("lr2M_rec") == 312640
    rec = C.resolve(cfg, "REC")
    assert [k for k in rec if rec[k] != C.resolve(cfg, "lr2M_rec")[k]] == ["lr_horizon_frames"]
    a5 = C.resolve(cfg, "A5a_lr_1em4")
    assert [k for k in a5 if a5[k] != C.resolve(cfg, "lr2M_A5a")[k]] == ["lr_horizon_frames"]
