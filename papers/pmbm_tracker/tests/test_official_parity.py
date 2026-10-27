"""Differential tests against the AUTHORS' PMBM code (Garcia-Fernandez et al., github.com/Agarciafernandez/MTT,
folder 'PMBM filter', commit cc5c6df0e9343dc1345d91ed6ab95f01db4d1821), run under GNU Octave 10.3 (work copy with two
documented edits: murty.m instead of kBest2DAssign.m, a logging hook; see scripts/official/ and the README
'Checked against').  The fixtures in tests/fixtures/official/ are RECORDED from that code (scripts/official/build_fixtures.py); the
authors' code itself is not part of this repo.

Per case: our filter is run on exactly the recorded measurement set with exactly the recorded parameters, and after
EVERY step the label-free digest of the pruned posterior (Poisson mixture, global-hypothesis weights, per-hypothesis
Bernoulli counts / sums of existence probabilities, top-3 hypotheses in full) and the three pre-pruning estimates are
compared with the authors' numbers.  Tolerances: weights / existence 1e-9 absolute; means 1e-8 absolute;
covariances 1e-9 relative (entries up to 2e4); observed worst case is ~1e-12 / 1e-11 / 1e-15.
"""
import glob
import os

import numpy as np
import pytest

import conftest  # noqa: F401  (sets sys.path / JAX_PLATFORMS)
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "official"))

import jax.numpy as jnp  # noqa: E402

from src import algorithm as al  # noqa: E402
from src import kbest  # noqa: E402
import parity_lib as pl  # noqa: E402
import diff_run  # noqa: E402

FIX = sorted(glob.glob(os.path.join(HERE, "fixtures", "official", "*.npz")))
STATE_KEYS = [k for k in ("pois_w", "pois_m", "pois_C", "hyp_w", "hyp_nb", "hyp_sum_r", "hyp_sum_m", "top_r", "top_m", "top_C")]
TOL = dict(pois_w=1e-9, pois_m=1e-8, pois_C=1e-9, hyp_w=1e-9, hyp_nb=0.0, hyp_sum_r=1e-9, hyp_sum_m=1e-7, top_r=1e-9,
           top_m=1e-8, top_C=1e-9)   # C: relative, see below


def _load(path):
    return dict(np.load(path))


def _params(fx):
    """Rebuild the .mat-style parameter dict the oracle driver was given (plus a leading run axis on z)."""
    P = {k[6:]: v for k, v in fx.items() if k.startswith("param_")}
    P["z"] = fx["z"][None]
    P["zvalid"] = fx["zvalid"][None].astype(float)
    return P


def _worst_vs_fixture(fx, rec, K):
    """Max differences (digest keys, tolerance-normalised) of our per-step records against the authors' fixture."""
    worst = {k: 0.0 for k in STATE_KEYS}
    worst["est"] = 0.0
    struct_bad = []
    for k in range(K):
        ours = pl.digest_state(rec[k]["post"])
        for key in STATE_KEYS:
            a, b = fx["dig_" + key][k], ours[key]
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                struct_bad.append((k + 1, key))
                continue
            ok = ~np.isnan(a)
            diff = np.abs(a[ok] - b[ok])
            if key in ("pois_C", "top_C"):
                diff = diff / (1.0 + np.abs(a[ok]))     # relative (entries up to 2e4)
            worst[key] = max(worst[key], float(diff.max()) if diff.size else 0.0)
        Eo = pl.digest_est({e: rec[k]["est"][e] for e in (1, 2, 3)})
        a = fx["est"][k]
        if not np.array_equal(np.isnan(a), np.isnan(Eo)):
            struct_bad.append((k + 1, "est"))
            continue
        ok = ~np.isnan(a)
        worst["est"] = max(worst["est"], float(np.abs(a[ok] - Eo[ok]).max()) if ok.any() else 0.0)
        if rec[k]["n_hyp"] != int(fx["n_glob_pruned"][k]):
            struct_bad.append((k + 1, "n_hyp"))
    return worst, struct_bad


def _run_ours(fx, K=None, cfg_extra=None, **param_over):
    P = _params(fx)
    for k, v in param_over.items():
        P[k] = np.asarray(v, float).reshape(np.asarray(P[k]).shape)
    K = K or fx["z"].shape[0]
    M = int(fx["zvalid"].sum(-1).max()) + 2
    return diff_run.run_ours(P, 0, K, cfg_over=dict(T=192, H=200, **(cfg_extra or {})), M=M)


@pytest.mark.parametrize("path", FIX, ids=[os.path.basename(p)[:-4] for p in FIX])
def test_pmbm_matches_official_step_by_step(path):
    fx = _load(path)
    K = fx["z"].shape[0]
    rec = _run_ours(fx)
    worst, bad = _worst_vs_fixture(fx, rec, K)
    assert not bad, f"{os.path.basename(path)}: structural differences (counts) at {bad[:5]}"
    for key in STATE_KEYS:
        assert worst[key] <= TOL[key], f"{os.path.basename(path)} {key}: worst diff {worst[key]:.3e} > {TOL[key]:.1e}"
    assert worst["est"] <= 1e-8, f"estimates: worst diff {worst['est']:.3e}"
    for k in range(K):
        assert rec[k]["n_children"] == int(fx["n_children"][k]) or rec[k]["ovf"][5] > 0, \
            f"step {k + 1}: pre-pruning child count {rec[k]['n_children']} != {int(fx['n_children'][k])}"
    assert rec[-1]["ovf"][0] == 0 and rec[-1]["ovf"][1] == 0 and rec[-1]["ovf"][2] == 0, f"our caps bound: {rec[-1]['ovf']}"


@pytest.mark.parametrize("what,over", [
    ("p_d", dict(p_d=0.9 + 1e-6)),                 # detection probability off by 1e-6
    ("clutter", dict(intensity_clutter=10.0 / 90000.0 * (1 + 1e-6))),
    ("p_s", dict(p_s=0.99 - 1e-7)),
    ("gate", dict(gating=2.0)),
    ("prune", dict(T_pruning=2e-4)),
])
def test_parity_check_is_sensitive(what, over):
    """Negative control: the comparison above MUST flag a deliberately perturbed filter (otherwise it proves nothing)."""
    fx = _load(os.path.join(HERE, "fixtures", "official", "A_paper.npz"))
    K = 12
    rec = _run_ours(fx, K=K, **over)
    worst, bad = _worst_vs_fixture(fx, rec, K)
    flagged = bool(bad) or any(worst[k] > TOL[k] for k in STATE_KEYS) or worst["est"] > 1e-8
    assert flagged, f"perturbation {what} not detected: worst {worst}"


@pytest.mark.parametrize("path", [p for p in FIX if "mur_n" in np.load(p).files], ids=lambda p: os.path.basename(p)[:-4])
def test_kbest_matches_official_murty(path):
    """Our k-best assignment (kbest.kbest_assign) vs the authors' Murty output on the PMBM update problems they logged."""
    fx = _load(path)
    n = int(fx["mur_n"])
    assert n >= 20
    n_checked = n_ties = 0
    for q in range(n):
        g = lambda name: fx[f"mur{q:03d}_{name}"]
        C = g("cost_matrix_log")                       # rows: Nprev old tracks, then one 'new' row per measurement
        M = C.shape[1]
        nprev = C.shape[0] - M
        cost_old = -C[:nprev].T if nprev else np.zeros((M, 0))
        cost_new = -np.array([C[nprev + m, m] for m in range(M)])
        k = int(g("kbest")[0, 0]) if np.ndim(g("kbest")) else int(g("kbest"))
        asg, cost, ovf = kbest.kbest_assign(jnp.asarray(cost_old[None]), jnp.asarray(cost_new), jnp.ones(M, bool),
                                            jnp.asarray([k]), k)
        asg, cost = np.asarray(asg)[0], np.asarray(cost)[0]
        n_valid = int(np.isfinite(cost).sum())
        off_child = np.asarray(g("glob_weight_log_children")).reshape(-1)
        off_cost = -(off_child - float(np.asarray(g("globWeightLog_pred")).reshape(-1)[0])
                     - float(np.asarray(g("cost_misdetection_sum")).reshape(-1)[0]))
        assert len(off_cost) == n_valid, f"problem {q}: {len(off_cost)} official children vs {n_valid} ours"
        assert np.allclose(np.sort(off_cost), np.sort(cost[:n_valid]), rtol=0, atol=1e-9), f"problem {q}: k-best costs differ"
        # assignment sets (translate official rows -> track index / -1 = new), compared inside groups of equal cost
        off_asg = np.asarray(g("assign_meas_to_track"))
        off_t = np.where(off_asg <= nprev, off_asg - 1, -1).astype(int)
        for lo in range(len(off_cost)):
            pass
        order_o = np.argsort(off_cost, kind="stable")
        grp_o = np.round(off_cost[order_o], 8)
        sets_o, sets_u = {}, {}
        for i in range(n_valid):
            sets_o.setdefault(grp_o[i], set()).add(tuple(off_t[order_o[i]]))
            sets_u.setdefault(np.round(cost[i], 8), set()).add(tuple(asg[i]))
        top = max(sets_o)                              # the last cost group may be cut by k in different ways when tied
        for key in sets_o:
            if key == top and n_valid == k:
                continue
            assert sets_o[key] == sets_u.get(key), f"problem {q}: assignments at cost {key} differ"
            n_ties += len(sets_o[key]) > 1
        n_checked += 1
    assert n_checked == n


def test_fixtures_present():
    assert len(FIX) >= 5, "parity fixtures missing (scripts/official/build_fixtures.py)"


def test_gospa_matches_official():
    """src/eval.gospa vs the authors' GOSPA.m (400 random point-set pairs incl. empty sets and pairs beyond the cut-off;
    official argument order GOSPA(x=truth, y=estimate)) and vs ComputeGOSPAerror.m on a recorded estimate sequence."""
    from src import eval as ev
    fx = dict(np.load(os.path.join(HERE, "fixtures", "official_misc", "gospa.npz")))
    nx, ny = fx["nx"].astype(int), fx["ny"].astype(int)
    worst = 0.0
    for i in range(len(nx)):
        X, Y = fx["xs"][i, :, :nx[i]].T, fx["ys"][i, :, :ny[i]].T
        g = ev.gospa(Y, X, p=2, c=10.0, alpha=2.0)
        worst = max(worst, abs(g["d"] - fx["off_d"][i]), abs(g["loc"] - fx["off_loc"][i]),
                    abs(g["missed"] - fx["off_mis"][i]), abs(g["false"] - fx["off_fal"][i]))
    assert worst < 1e-9, worst
    est, ne, Xt, alive = fx["est"], fx["n_est"].astype(int), fx["truth_X"], fx["truth_alive"]
    worst = 0.0
    for k in range(est.shape[2]):
        E = est[:, :ne[k], k].T[:, [0, 2]] if ne[k] else np.zeros((0, 2))
        Tp = np.array([Xt[i, k][[0, 2]] for i in range(4) if alive[i, k]]).reshape(-1, 2)
        g = ev.gospa(E, Tp, p=2, c=10.0, alpha=2.0)
        worst = max(worst, abs(g["d"] ** 2 - fx["off_sq"][k]), abs(g["loc"] - fx["off_sl"][k]),
                    abs(g["missed"] - fx["off_sm"][k]), abs(g["false"] - fx["off_sf"][k]))
    assert worst < 1e-9, worst


@pytest.mark.parametrize("case", ["A_paper", "C_stress"])
def test_pmbm_jax_kbest_backend_matches_official(case):
    """The jit/vmap production path (pure-JAX k-best, backend='jax') against the same recorded authors' numbers."""
    fx = _load(os.path.join(HERE, "fixtures", "official", case + ".npz"))
    K = fx["z"].shape[0]
    rec = _run_ours(fx, K=K, cfg_extra=dict(backend="jax", queue_cap=512))
    assert rec[-1]["ovf"][3] == 0, f"k-best queue overflow: {rec[-1]['ovf']}"
    worst, bad = _worst_vs_fixture(fx, rec, K)
    assert not bad, bad[:5]
    for key in STATE_KEYS:
        assert worst[key] <= TOL[key], f"{case} {key}: {worst[key]:.3e}"
    assert worst["est"] <= 1e-8


def test_noprune_fixture_detects_the_global_weight_floor():
    """Negative control: on the T_pruning = 0 fixture (authors' HEAD code, no global-weight floor) a filter with the 1e-4 floor MUST be flagged."""
    fx = _load(os.path.join(HERE, "fixtures", "official", "Z_noprune.npz"))
    K = 12
    rec = _run_ours(fx, K=K, T_pruning=1e-4)
    worst, bad = _worst_vs_fixture(fx, rec, K)
    assert bad or any(worst[k] > TOL[k] for k in STATE_KEYS) or worst["est"] > 1e-8
