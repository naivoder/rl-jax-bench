"""Smoke test (< 30 s, JAX_PLATFORMS=cpu): shapes, replay add/sample/gather, one DQfD update, a short interaction chunk,
config <-> sources.yaml consistency, hp table sanity."""
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src import algorithm as A, replay as R
from src.envs import MinAtar, pack_obs, unpack_obs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(REPO, "src/configs/default.yaml")))
SRC = yaml.safe_load(open(os.path.join(REPO, "src/configs/sources.yaml")))


def test_config_keys_all_have_sources():
    for sec, kv in CFG.items():
        assert sec in SRC, sec
        for k, v in kv.items():
            assert k in SRC[sec], f"{sec}.{k} lacks a sources.yaml entry"
            sv = SRC[sec][k]["value"] if isinstance(SRC[sec][k], dict) and "value" in SRC[sec][k] else None
            if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                assert abs(float(sv) - float(v)) < 1e-12, f"{sec}.{k}: default.yaml {v} != sources.yaml {sv}"


def test_pack_roundtrip_and_hp_table():
    x = jax.random.bernoulli(jax.random.PRNGKey(0), 0.3, (5, 10, 10, 6))
    assert bool(jnp.all(unpack_obs(pack_obs(x), (10, 10, 6)) == x.astype(jnp.float32)))
    hp = A.stack_hp(A.ARM_NAMES, 2, CFG)
    assert hp["lam_e"].shape == (2 * len(A.ARM_NAMES),)
    d = {n: A.arm_hp(n, CFG) for n in A.ARM_NAMES}
    assert d["DQfD"]["lam_e"] == 1.0 and d["NoMargin"]["lam_e"] == 0.0 and d["NoNstep"]["lam_n"] == 0.0
    assert d["ADET"]["lam_ce"] == 1.0 and d["PDD"]["lam3"] == 0.0 and d["PDD"]["demo_on"] == 0.0
    assert d["HER"]["eps_d"] == d["HER"]["eps_a"] and d["RBS"]["rbs"] == 1.0 and d["IMIT"]["lam_td"] == 0.0


def test_one_update_and_interaction_chunk():
    t0 = time.time()
    env = MinAtar("Breakout")
    nd, cap = 300, 200
    L = R.make_layout(nd, cap, 50, 10, 0.99, 0.4, 64)
    cfg = {**CFG, "per": {**CFG["per"], "agent_capacity": cap}, "run": {**CFG["run"], "bin_width": 20, "episode_log_cap": 50}}
    fns = A.Fns(env, L, cfg, 100)
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.integers(0, 256, (nd, 50)), jnp.uint8)
    end = jnp.asarray(np.arange(nd) % 60 == 59)
    demo = dict(obs=obs, nobs=obs, act=jnp.asarray(rng.integers(0, 3, nd), jnp.int32), rew=jnp.asarray(rng.random(nd), jnp.float32) * 0.69,
                term=end, end=end)
    hp = {k: jnp.asarray(v, jnp.float32) for k, v in A.arm_hp("DQfD", cfg).items()}
    buf = R.init_buffer(L, demo, hp)
    params = fns.init_params(jax.random.PRNGKey(0))
    opt = fns.tx.init(params)
    p2, tp2, o2, b2, dg = jax.jit(fns.update)(params, params, opt, buf, jax.random.PRNGKey(1), 0.6, hp, jnp.array(True), jnp.array(0))
    assert float(dg["loss"]) > 0 and float(dg["demo_share"]) == 1.0
    assert any(float(jnp.abs(a - b).max()) > 0 for a, b in zip(jax.tree_util.tree_leaves(p2), jax.tree_util.tree_leaves(params)))
    assert float(jnp.sum(b2["pa"] != buf["pa"])) > 0                        # priorities moved
    c = fns.init_interact(params, params, opt, buf, jax.random.PRNGKey(2))
    c = jax.jit(lambda c, k: fns.interact_chunk(c, hp, k, 12, jnp.array(True)))(c, jax.random.PRNGKey(3))
    assert int(c["t"]) == 12 and int(c["cnt"]) == 12 and int(c["buf"]["nw"]) == 12
    assert bool(jnp.all(jnp.isfinite(jax.tree_util.tree_leaves(c["params"])[0])))
    assert time.time() - t0 < 25


def test_im2col_conv_equals_flax_conv():
    import flax.linen as nn
    from src.model import Conv3x3
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 10, 10, 5))
    m = Conv3x3(7); p = m.init(jax.random.PRNGKey(1), x)
    ref = nn.Conv(7, (3, 3), padding="SAME").apply({"params": {"kernel": p["params"]["kernel"], "bias": p["params"]["bias"] + 0.1}}, x)
    ours = m.apply({"params": {"kernel": p["params"]["kernel"], "bias": p["params"]["bias"] + 0.1}}, x)
    assert float(jnp.abs(ref - ours).max()) < 1e-5
