"""Smoke test (< 30 s on CPU): shapes of the environment step and of every network, and one optimisation step of
each loss (PPO, BC cross-entropy, BC-MSE, AE) is finite and changes the parameters."""
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common  # noqa: F401,E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from src.configs.config import EnvConfig, NetConfig  # noqa: E402
from src import data, algorithm as alg  # noqa: E402
from src.model import ActorCritic, TrackEncoder, AEDecoder  # noqa: E402

T0 = [None]   # set by the first smoke test (collection of other test modules must not count)


def test_env_shapes():
    T0[0] = time.time()
    cfg = EnvConfig()
    s = data.reset(jax.random.PRNGKey(0), cfg)
    s2, o, info = jax.jit(lambda s, a: data.step(s, a, cfg))(s, jnp.array([9, 9], jnp.int32))
    assert o.tl.shape == (17, 7) and o.tl_mask.shape == (17,) and o.sv_az.shape == (100,)
    assert int(s2.t) == 1 and np.isfinite(float(info["r_sv"])) and np.isfinite(float(info["gospa_d"]))
    img = data.render_raster(o.sv_az, o.sv_el, o.sv_age, o.sv_valid, cfg)
    assert img.shape == (48, 48) and img.dtype == jnp.float32


def test_networks_and_one_step_each():
    ecfg = replace(EnvConfig(), n_sv=16, k_max=17)
    B = 8
    rng = np.random.default_rng(0)
    obs = data.Obs(jnp.asarray(rng.uniform(-1, 1, (B, 17, 7)), jnp.float32), jnp.asarray(rng.uniform(size=(B, 17)) < 0.3),
                   jnp.asarray(rng.uniform(-60, 60, (B, 16)), jnp.float32), jnp.asarray(rng.uniform(-60, 60, (B, 16)), jnp.float32),
                   jnp.asarray(rng.integers(1, 99, (B, 16)), jnp.float32), jnp.ones((B, 16), bool))
    a = jnp.asarray(rng.integers(0, 19, (B, 2)), jnp.int32)
    zdim = {"flat": 119, "bigru": 64, "mhsa": 64}
    for enc in ("flat", "bigru", "mhsa"):
        ncfg = replace(NetConfig(), encoder=enc)
        net = ActorCritic(ncfg, ecfg)
        p = jax.jit(net.init)(jax.random.PRNGKey(1), obs)
        la, le, v = jax.jit(net.apply)(p, obs)
        assert la.shape == (B, 19) and le.shape == (B, 19) and v.shape == (B,)
        assert jax.eval_shape(lambda q: TrackEncoder(ncfg).apply(q, obs.tl, obs.tl_mask),
                              {"params": p["params"]["track_enc"]} if enc != "flat" else {}).shape == (B, zdim[enc])
    # one PPO step (mhsa = the heaviest encoder)
    opt = alg.ppo_optimizer(3e-4, 0.5, 1e-5)
    st = opt.init(p)
    old, _ = alg.logp_entropy(la, le, a)

    def loss(q):
        la2, le2, v2 = net.apply(q, obs)
        pg, vl, el, _ = alg.ppo_loss_terms(la2, le2, v2, a, old, jnp.asarray(rng.normal(size=B)), jnp.zeros(B), 0.2)
        return pg + 0.5 * vl, (alg.bc_ce_loss(la2, le2, a, 1e-3), alg.bc_mse_loss(la2, le2, a))
    (l, (ce, mse)), g = jax.jit(jax.value_and_grad(loss, has_aux=True))(p)
    u, st = opt.update(g, st, p)
    p2 = optax.apply_updates(p, u)
    assert np.isfinite(float(l)) and np.isfinite(float(ce)) and np.isfinite(float(mse))
    assert float(optax.global_norm(jax.tree.map(lambda x, y: x - y, p2, p))) > 0
    # AE step on U(-1,1) noise
    enc = TrackEncoder(replace(NetConfig(), encoder="bigru"))
    pe = enc.init(jax.random.PRNGKey(2), obs.tl, obs.tl_mask)
    dec = AEDecoder(256, 119)
    pd = dec.init(jax.random.PRNGKey(3), jnp.zeros((1, 64)))
    x = jax.random.uniform(jax.random.PRNGKey(4), (16, 17, 7), jnp.float32, -1, 1)
    l = jax.jit(lambda pe, pd: alg.ae_loss(dec.apply(pd, enc.apply(pe, x, jnp.ones((16, 17), bool))).reshape(x.shape),
                                           x))(pe, pd)
    assert np.isfinite(float(l)) and 0.2 < float(l) < 1.0     # ~1/3 at init


def test_runtime_budget():
    assert T0[0] is not None and time.time() - T0[0] < 300.0  # generous wall-clock bound; about 20 s on an idle desktop CPU
