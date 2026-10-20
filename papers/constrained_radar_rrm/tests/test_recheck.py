"""The re-evaluation unit checks for the review fixes, CPU, < 30 s.

  - OA's Case III L_bm knowledge: E[cos^2 dtheta] closed form vs Monte Carlo under the prior bearing variance.
  - eq. 30 literal budget term: g_req = 1 uses the requested (clipped) dwells, 0 the executed ones; r = U - lam g.
  - eq. 34 literal state: lambda_{t-1} carried and observed; prev_tau holds the chosen (clipped) dwell.
  - T_init censoring and the re-digitised replica windows.
  - v2 evaluation slot keys never collide with the scenario-table keys.
"""
import json
import os

import numpy as np

from src import common  # noqa: F401
import jax
import jax.numpy as jnp

from src import data, env as E, eval as ev, physics as ph
from src.common import CFG_DIR, EnvCfg


def _state_with_tracks(cfg, scen, t, idx):
    pos, _ = data.truth(scen, t, cfg)
    s = E.reset(cfg, lit=True)
    K = cfg.K
    act = jnp.zeros(K, bool).at[jnp.arange(len(idx))].set(True)
    tgt = jnp.zeros(K, jnp.int32).at[jnp.arange(len(idx))].set(jnp.asarray(idx))
    x = jnp.zeros((K, 4)).at[jnp.arange(len(idx)), :2].set(pos[jnp.asarray(idx)])
    P = jnp.tile(jnp.diag(jnp.array([100.0, 100.0, 1.0, 1.0])), (K, 1, 1))
    return s._replace(t=jnp.asarray(t, jnp.int32), act=act, tgt=tgt, x=x, P=P, lam=jnp.asarray(7000.0),
                      lam_prev=jnp.asarray(3000.0))


def test_lbm_expectation_mc():
    cfg = EnvCfg.from_yaml(lbm=True)
    xp = jnp.array([[3000.0, 2000.0, 0.0, 0.0]])
    Pp = jnp.diag(jnp.array([4.0e5, 9.0e5, 1.0, 1.0]))[None]
    fac = float(E.lbm_factor_hat(cfg, xp, Pp)[0])
    h = np.array([-2000.0, 3000.0]) / (3000.0 ** 2 + 2000.0 ** 2)
    s2 = h @ np.diag([4.0e5, 9.0e5]) @ h
    d = np.random.default_rng(0).normal(0.0, np.sqrt(s2), 400000)
    mc = np.mean(np.cos(d) ** 2)
    assert abs(1.0 / fac - mc) < 2e-3, (1.0 / fac, mc)
    assert fac > 1.0


def test_budget_term_requested_vs_executed_and_literal_state():
    cfg = EnvCfg.from_yaml()
    scen = data.replica_case1(cfg)
    s = _state_with_tracks(cfg, scen, 2500, [0, 1])
    v = E.pre(cfg, scen, s)
    lut = ev.make_lut(cfg)
    tau_raw = jnp.array([2.0, 1.5, 9.0, 0.0, 0.0])                 # sum over occupied = 3.5 s > T0; slot 2 empty
    key = jax.random.PRNGKey(0)
    base = dict(snr0=1.8585, s_ref=5.62)
    s_ex, o_ex = E.post(cfg, dict(base, g_req=jnp.asarray(0.0)), lut, scen, s, v, tau_raw, key)
    s_rq, o_rq = E.post(cfg, dict(base, g_req=jnp.asarray(1.0), obs_lit=jnp.asarray(1.0)), lut, scen, s, v,
                        tau_raw, key)
    assert np.isclose(float(o_ex["g"]), 2.5 / 2.5 - 0.9)            # executed: scaled to T0
    assert np.isclose(float(o_rq["g"]), 3.5 / 2.5 - 0.9)            # requested, clipped, occupied slots only
    assert np.isclose(float(o_rq["r"]), float(o_rq["U"]) - 7000.0 * (3.5 / 2.5 - 0.9))
    assert np.isclose(float(o_rq["U"]), float(o_ex["U"]))           # the physics is the same; only the penalty differs
    assert float(s_rq.lam) > float(s_ex.lam)                        # eq. 33 sees the larger g
    # eq. 34 literal: lambda_{t-1} carried, chosen dwell in the state, used by the observation when lit = 1
    assert float(s_rq.lam_prev) == 7000.0
    keep = np.asarray(s_rq.act)
    assert np.allclose(np.asarray(s_rq.prev_tau)[keep], np.array([2.0, 1.5])[keep[:2]])
    assert np.allclose(np.asarray(s_ex.prev_tau)[keep], (np.array([2.0, 1.5]) * 2.5 / 3.5)[keep[:2]])
    ob = E.obs_ddpg(s_rq, cfg, 1e-4, 1e-4, "affine", jnp.asarray(1.0))
    assert np.isclose(float(ob[-1]), 7000.0 * 1e-4, rtol=1e-6)
    ob0 = E.obs_ddpg(s_rq, cfg, 1e-4, 1e-4, "affine", jnp.asarray(0.0))
    assert np.isclose(float(ob0[-1]), float(s_rq.lam) * 1e-4, rtol=1e-6)


def test_tinit_censoring():
    cfg = EnvCfg.from_yaml()
    scen = data.replica_case1(cfg)
    s = E.reset(cfg)
    seen = jnp.zeros(cfg.M, bool).at[jnp.array([0, 1, 2])].set(True)
    conf = (-jnp.ones(cfg.M, jnp.int32)).at[0].set(12)
    s = s._replace(t=jnp.asarray(10000, jnp.int32), seen=seen, conf_slot=conf)
    ts, n, nv = E.tinit_stats(scen, s)
    cs = E.tinit_cens_sum(scen, s)
    born, death = np.asarray(scen.born), np.asarray(scen.death)
    assert int(n) == 1 and int(nv) == 2 and int(ts) == 12 - born[0]
    assert int(cs) == (death[1] - born[1]) + (death[2] - born[2])


def test_replica_v2_windows():
    d = json.load(open(os.path.join(CFG_DIR, "scenario_case1.json")))["targets"]
    for t in d:
        if t["label"] != "Target 2":
            assert t["removed"] - t["born"] == 3000, t
    t4 = [t for t in d if t["label"] == "Target 4"][0]
    assert abs(t4["born"] - 6761) <= 25 and abs(t4["removed"] - 9796) <= 25   # Fig. 4 pixel reading
    old = json.load(open(os.path.join(CFG_DIR, "scenario_case1_v1.json")))["targets"]
    assert [t["label"] for t in old] == [t["label"] for t in d]


def test_v2_slot_keys_do_not_collide():
    sk = jax.random.fold_in(jax.random.PRNGKey(20261019), 0x7E57)
    tab = {tuple(np.asarray(jax.random.key_data(jax.random.fold_in(sk, c)) if hasattr(jax.random, "key_data")
                            else jax.random.fold_in(sk, c)).tolist()) for c in (0x5A, 7, 0xC1)}
    old, ev.SLOT_KEY = ev.SLOT_KEY, "v2"
    try:
        ks = jax.vmap(lambda t: ev.slot_key(sk, t))(jnp.arange(10000))
        ks = {tuple(x) for x in np.asarray(ks).tolist()}
        assert not (ks & tab)
        ev.SLOT_KEY = "legacy"
        kl = jax.vmap(lambda t: ev.slot_key(sk, t))(jnp.arange(10000))
        kl = {tuple(x) for x in np.asarray(kl).tolist()}
        assert len(kl & tab) == 3                                  # the documented legacy collisions (t = 7, 90, 193)
    finally:
        ev.SLOT_KEY = old
