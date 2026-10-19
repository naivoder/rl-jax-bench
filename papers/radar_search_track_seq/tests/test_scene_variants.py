"""Scene options added after the review (see README):

   del_stat="frob" deletes on the Frobenius norm (this paper's text), "trace" on the trace (library, primary)
   gospa_lib: the library-reading GOSPA (a track deleted at t counts with its last state at t) equals Stone Soup's
      MultiManager + GOSPAMetric over all tracks ever (fixture ss_metrics.npz, recorded by scripts/official_metric_parity.py)
   clutter: n_fa_slots = 0 leaves the step bitwise unchanged; > 0 gives the declared false-alarm rate
"""
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common  # noqa: F401,E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.configs.config import EnvConfig, K_SMALL, K_BASE_ACT  # noqa: E402
from src import data, baselines as B, eval as E  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "official")


def _coast_life(cfg, stat):
    """Steps until an unrevisited new track exceeds del_thr under the given statistic (numpy reference)."""
    x, C0 = data.initiate_state(jnp.array([0.0, 0.0, 20_000.0]), replace(cfg, init_jacobian="analytic"))
    F, Q = (np.asarray(a) for a in data.cv_FQ(cfg.dt, cfg.q_filt))
    P = np.asarray(C0)
    for k in range(1, 100_000):
        P = F @ P @ F.T + Q
        v = np.trace(P) if stat == "trace" else np.sqrt(np.sum(P ** 2))
        if v > cfg.del_thr:
            return k
    return None


@pytest.mark.parametrize("stat", ["trace", "frob"])
def test_F3_deleter_statistic(stat):
    cfg = replace(EnvConfig(), del_stat=stat, init_jacobian="analytic")
    ref = _coast_life(cfg, stat)
    tr = data.empty_tracks(cfg)
    z = jnp.array([[0.0, 0.0, 20_000.0]])
    tr, nt, _ = data.tracker_step(tr, jnp.int32(0), z, jnp.array([True]), jnp.int32(0), cfg)
    none = jnp.zeros((1, 3)), jnp.array([False])
    alive = 0
    for k in range(1, ref + 2):
        tr, nt, aux = data.tracker_step(tr, nt, none[0], none[1], jnp.int32(k), cfg)
        if bool(aux["kill"].any()):
            alive = k
            break
    assert alive == ref, (stat, alive, ref)


def test_F3_frob_outlives_trace():
    cfg = EnvConfig()
    assert _coast_life(cfg, "frob") > _coast_life(cfg, "trace")


def test_F6_no_clutter_is_bitwise_primary_and_clutter_rate():
    cfg = replace(EnvConfig(), T=200)
    sk = E.scenario_keys(jax.random.PRNGKey(15), K_SMALL, 2)
    pk = jax.vmap(lambda k: jax.random.fold_in(k, K_BASE_ACT))(sk)
    pi, po = B.make_random(cfg)
    a = jax.tree.map(np.asarray, E.eval_many(pi, po, sk, pk, cfg))
    b = jax.tree.map(np.asarray, E.eval_many(pi, po, sk, pk, replace(cfg, n_fa_slots=0, clutter_rate=0.7)))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    c = jax.tree.map(np.asarray, E.eval_many(pi, po, sk, pk, replace(cfg, n_fa_slots=4, clutter_rate=0.5)))
    rate = c["n_fa"].sum() / (2 * cfg.T)
    assert abs(rate - 0.5) < 4 * np.sqrt(0.5 * (1 - 0.5 / 4) / (2 * cfg.T)), rate
    assert c["n_init"].sum() > a["n_init"].sum()


def test_F5_library_gospa_vs_stonesoup_metric_manager():
    """Re-runs the P9 fixture's rollouts (3 small-scale scenarios x Random / Coverage / hand / ceiling, analytic Jacobian,
    T = 1000) with gospa_lib on and compares with Stone Soup's library pipeline GOSPA on the same detections."""
    path = os.path.join(FIX, "ss_metrics.npz")
    if not os.path.exists(path):
        pytest.skip("ss_metrics.npz not generated")
    d = np.load(path)
    cfg = replace(EnvConfig(), init_jacobian="analytic", gospa_metrics=True, gospa_lib=True)
    sk = E.scenario_keys(jax.random.PRNGKey(15), K_SMALL, 3)
    pk = jax.vmap(lambda k: jax.random.fold_in(jax.random.fold_in(k, K_BASE_ACT), 0))(sk)
    pols = {"random": B.make_random(cfg), "coverage": B.make_coverage(cfg), "hand": B.make_hand_rule(cfg, 12, 4500.0),
            "ceiling": B.make_ceiling(cfg)}
    live, lib, ovf = [], [], []
    for name, (pi, po) in pols.items():
        ev = jax.tree.map(np.asarray, jax.jit(lambda x, y: E.eval_many(pi, po, x, y, cfg))(sk, pk))
        live.append(np.stack([ev[k] for k in ("gospa_d", "gospa_loc", "gospa_miss", "gospa_false")], -1))
        lib.append(np.stack([ev[k] for k in ("gospa_lib_d", "gospa_lib_loc", "gospa_lib_miss", "gospa_lib_false")], -1))
        ovf.append(ev["lib_kill_ovf"])
    live, lib = np.concatenate(live), np.concatenate(lib)
    assert np.all(np.concatenate(ovf) == 0)
    assert list(d["policy"]) == sum([[n] * 3 for n in pols], [])
    np.testing.assert_allclose(live, d["ss_live_gospa"], rtol=1e-6, atol=1e-3)     # P9 unchanged
    np.testing.assert_allclose(lib, d["ss_lib_gospa"], rtol=1e-6, atol=1e-3)
