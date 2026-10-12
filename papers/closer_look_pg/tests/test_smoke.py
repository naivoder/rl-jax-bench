"""Smoke test (<30 s, JAX_PLATFORMS=cpu): env + nets shapes, one rollout, one PPO and one TRPO update, the
gradient instrument on a real batch, frozen normalisers, disjoint instrument keys, config traceability."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src import common  # noqa: F401  (bootstrap + compile cache)
from src import algorithm as A
from src import data as D
from src.configs import default as C
from src.model import Nets
from src.train import batch_from_traj

T = 64


def _setup(alg):
    cfg = C.algo_config(alg, "hopper")
    cfg.update(minibatches=2, value_minibatches=2, policy_epochs=1, value_epochs=1)
    env, spec = D.make_env("hopper", cfg["physics_backend"])
    nets = Nets(spec.obs_dim, spec.act_dim, cfg)
    return cfg, env, spec, nets


def test_rollout_update_and_instrument():
    cfg, env, spec, nets = _setup("ppo")
    assert (spec.obs_dim, spec.act_dim) == (11, 3)
    rollout = jax.jit(D.make_rollout(env, cfg, nets), static_argnums=(3,))
    update, init_state = A.make_train_update(nets, cfg)
    ts = init_state(jax.random.PRNGKey(0))
    norm = D.norm_init(spec.obs_dim)
    traj, norm1 = rollout(ts.pi, norm, jax.random.PRNGKey(1), T, jnp.bool_(True))
    assert traj.obs.shape == (T, 11) and traj.act.shape == (T, 3) and traj.rew.shape == (T,)
    assert float(norm1.obs.n) >= T  # pushed every step (+ reset obs)
    # frozen normalisers are bit-identical after an instrument rollout
    _, norm2 = rollout(ts.pi, norm1, jax.random.PRNGKey(2), T, jnp.bool_(False))
    for a, b in zip(jax.tree_util.tree_leaves(norm1), jax.tree_util.tree_leaves(norm2)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    b = batch_from_traj(nets, cfg, ts.v, traj)
    ts1, st = jax.jit(update)(ts, b, 0, jax.random.PRNGKey(3))
    assert np.isfinite(np.asarray(ts1.pi)).all() and not np.array_equal(np.asarray(ts1.pi), np.asarray(ts.pi))
    g = A.pg_vector(nets, ts.pi, b.obs, b.act, b.adv)
    assert g.shape == (nets.n_pi,) and np.isfinite(np.asarray(g)).all()
    ve = A.value_errors(nets.v_apply(ts1.v, b.obs), b.values, b.adv, b.ret, b.not_done)
    assert all(np.isfinite(float(v)) for v in ve.values())


def test_trpo_step_respects_kl():
    """TRPO update on a synthetic batch (the rollout path is covered above; avoids a second MJX compile)."""
    cfg = C.algo_config("trpo", "hopper")
    cfg.update(value_minibatches=2, value_epochs=1)
    nets = Nets(11, 3, cfg)
    update, init_state = A.make_train_update(nets, cfg)
    ts = init_state(jax.random.PRNGKey(0))
    k = jax.random.split(jax.random.PRNGKey(4), 3)
    obs = jax.random.normal(k[0], (T, 11)); act = jax.random.normal(k[1], (T, 3))
    b = A.Batch(obs, act, nets.logp(ts.pi, obs, act), jax.random.normal(k[2], (T,)), jnp.zeros(T), jnp.zeros(T),
                jnp.ones(T))
    _, st = jax.jit(update)(ts, b, 0, jax.random.PRNGKey(3))
    assert float(st["post_kl"]) <= cfg["kl_delta"] + 1e-6 and float(st["ls_accepted"]) == 1.0


def test_gae_last_index_quirk():
    r = jnp.array([1.0, 1.0, 1.0]); v = jnp.array([0.5, 0.5, 2.0]); nd = jnp.array([1.0, 1.0, 1.0])
    adv, ret = A.gae(r, v, nd, 0.99, 0.95)
    # last index bootstraps with V(s_{T-1}) itself (official): delta = 1 + 0.99*2 - 2
    assert abs(float(adv[-1]) - (1 + 0.99 * 2 - 2)) < 1e-6
    assert abs(float(ret[-1]) - 1.0) < 1e-6  # returns are not bootstrapped


def test_instrument_keys_disjoint():
    root = jax.random.PRNGKey(0)
    ks = [common.k_instr(root, 150, b) for b in range(6)] + [jax.random.fold_in(jax.random.fold_in(root, common.K_TRAIN), 150)]
    raw = {tuple(np.asarray(jax.random.key_data(k) if hasattr(jax.random, "key_data") else k).ravel()) for k in ks}
    assert len(raw) == len(ks)


def test_config_keys_traced():
    src = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "sources.yaml")))
    flat = set()
    for blk in src.values():
        if isinstance(blk, dict):
            flat |= set(blk)
    for blk_name, blk in (("common", C.COMMON), ("ppo", C.PPO), ("trpo", C.TRPO), ("instruments", C.INSTRUMENTS)):
        for k in blk:
            ok = k in flat or any(f.endswith(k) or f.startswith(k) for f in flat)
            assert ok, f"{blk_name}.{k} has no sources.yaml entry"
