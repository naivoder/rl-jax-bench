"""Recheck fix-stage options (see README). Every option defaults to the pre-fix
behaviour; these tests pin both the default and the alternative.

    ssv_norm: "fifo" (primary, p_max over the FIFO) vs "single" (p_max = phi_0(0)); the SR and the raster scale by
         exactly p_max_fifo / phi_0(0), and the default equals the explicit "fifo"
   truth_vel_sigma: None ties the truth velocity to init_vel_sigma (bitwise unchanged); a number moves the
         truth only, never the filter prior
"""
import math
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common  # noqa: F401,E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.configs.config import EnvConfig  # noqa: E402
from src import data  # noqa: E402


def _rollout(cfg, n=60, seed=3):
    s = data.reset(jax.random.PRNGKey(seed), cfg)
    srs, obs = [], None
    for t in range(n):
        a = jnp.array([(7 * t) % cfg.n_a, (3 * t + 1) % cfg.n_a], jnp.int32) if t % 4 else jnp.array([9, 9], jnp.int32)
        s, obs, info = data.step(s, a, cfg)
        srs.append(float(info["r_sv"]))
    return np.array(srs), obs


def test_SA1_ssv_norm_default_is_fifo():
    base = replace(EnvConfig(), gospa_metrics=False)
    assert base.ssv_norm == "fifo"
    assert data.ssv_pmax(base) == data.ssv_pmax(replace(base, ssv_norm="fifo"))
    s0 = base.sigma0_deg
    want = sum(1.0 / (2 * math.pi * (s0 * (1 + base.gamma_sv) ** k) ** 2) for k in range(base.n_sv))
    assert abs(data.ssv_pmax(base) - want) < 1e-12 * want


def test_SA1_ssv_norm_single_scales_sr_and_raster_exactly():
    base = replace(EnvConfig(), gospa_metrics=False)
    single = replace(base, ssv_norm="single")
    phi0 = 1.0 / (2 * math.pi * base.sigma0_deg ** 2)
    assert abs(data.ssv_pmax(single) - phi0) < 1e-15 * phi0
    fac = data.ssv_pmax(base) / phi0
    assert 40.0 < fac < 47.0                     # 43.8 at gamma 0.01, N_SV 100 
    sr_f, o_f = _rollout(base)
    sr_s, o_s = _rollout(single)
    np.testing.assert_allclose(sr_s, sr_f * fac, rtol=1e-12, atol=0)
    ras = lambda o, c: np.asarray(data.render_raster(o.sv_az, o.sv_el, o.sv_age, o.sv_valid, c))
    np.testing.assert_allclose(ras(o_s, single), ras(o_f, base) * fac, rtol=2e-5, atol=0)


def test_SA11_truth_vel_sigma_decouples_truth_from_filter():
    base = EnvConfig()
    k = jax.random.PRNGKey(11)
    t0 = np.asarray(data.sample_truth(k, base))
    np.testing.assert_array_equal(t0, np.asarray(data.sample_truth(k, replace(base, truth_vel_sigma=None))))
    np.testing.assert_array_equal(t0, np.asarray(data.sample_truth(k, replace(base, truth_vel_sigma=base.init_vel_sigma))))
    # filter-only change: truth identical, initiator prior changes
    filt = replace(base, init_vel_sigma=10.0, truth_vel_sigma=30.0)
    np.testing.assert_array_equal(t0, np.asarray(data.sample_truth(k, filt)))
    _, C_b = data.initiate_state(jnp.array([0.0, 0.0, 20_000.0]), base)
    _, C_f = data.initiate_state(jnp.array([0.0, 0.0, 20_000.0]), filt)
    assert abs(float(C_b[1, 1]) - 900.0) < 1e-6 and abs(float(C_f[1, 1]) - 100.0) < 1e-6
    # truth-only change: velocities scale, positions identical
    t2 = np.asarray(data.sample_truth(k, replace(base, truth_vel_sigma=15.0)))
    np.testing.assert_array_equal(t2[:, [0, 2, 4]], t0[:, [0, 2, 4]])
    np.testing.assert_allclose(t2[:, [1, 3]], t0[:, [1, 3]] * 0.5, rtol=1e-12)


def test_SA3_pretrain_reports_cnn_liveness():
    from src.configs.config import Config
    from src.train import make_pretrain, Flags, lane_key
    cfg = Config()
    cfg = replace(cfg, net=replace(cfg.net, encoder="flat"), env=replace(cfg.env, T=40),
                  pre=replace(cfg.pre, bc_episodes=2, bc_epochs=1))
    net, pretrain_lane = make_pretrain(cfg)
    fl = Flags(*(jnp.float32(v) for v in (0, 1, 0, 1)))
    params, out = jax.jit(pretrain_lane)(lane_key(cfg, "flat", 0), fl)
    a0, a1 = float(out["cnn_alive_pre_bc"]), float(out["cnn_alive_ppo_start"])
    assert 0.5 < a0 <= 1.0              # orthogonal init: almost every unit is live at init
    assert 0.0 <= a1 <= 1.0


def test_SA16_bc_scope_option():
    from src.configs.config import Config
    from src.train import make_pretrain, Flags, lane_key
    base = Config()
    assert base.pre.bc_train_extractor is False
    fl = Flags(*(jnp.float32(v) for v in (0, 1, 0, 1)))                # no AE, BC on
    res = {}
    for scope in (False, True):
        cfg = replace(base, net=replace(base.net, encoder="mhsa"), env=replace(base.env, T=40),
                      pre=replace(base.pre, bc_episodes=2, bc_epochs=1, ae_steps=20, bc_train_extractor=scope))
        net, pretrain_lane = make_pretrain(cfg)
        k = lane_key(cfg, "mhsa", 0)
        p0 = net.init(jax.random.fold_in(k, 0), __import__("src.model", fromlist=["dummy_obs"]).dummy_obs(
            replace(cfg.env, gospa_metrics=False), (1,)))
        p, _ = jax.jit(pretrain_lane)(k, fl)
        d = jax.tree.map(lambda a, b: float(jnp.max(jnp.abs(a - b))), p["params"]["track_enc"], p0["params"]["track_enc"])
        res[scope] = max(jax.tree.leaves(d))
    assert res[False] == 0.0            # frozen: extractor bit-identical to its (non-AE) init
    assert res[True] > 1e-6             # imitation default: the extractor moves
