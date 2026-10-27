import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import stats as sst  # noqa: E402

from src import data  # noqa: E402

CFG = data.load_scenario_cfg()
ROOT = jax.random.PRNGKey(18)


def test_cfg_values():
    assert CFG["q"] == 0.01 and CFG["p_s"] == 0.99 and CFG["n_steps"] == 81 and CFG["midpoint"] == 41
    assert np.allclose(CFG["Q"][:2, :2], 0.01 * np.array([[1 / 3, 1 / 2], [1 / 2, 1]]))
    assert data.load_scenario_cfg(q=0.02)["Q"][1, 1] == pytest.approx(0.02)
    assert data.table2_config(0) == (0.95, 10) and data.table2_config(14) == (0.6, 20) and data.table2_config(4) == (0.9, 15)


def test_truth_structure_and_determinism():
    tr = data.make_truth(jax.random.fold_in(ROOT, 0), CFG)
    X, alive = np.asarray(tr["X"]), np.asarray(tr["alive"])
    assert X.shape == (4, 81, 4) and alive.shape == (4, 81)
    assert alive[0].sum() == 40 and alive[0, :40].all() and not alive[0, 40:].any()  # target 1: k=1..40
    assert alive[1:].all()
    assert (X[~alive] == 0).all()
    # the dynamics hold exactly in law: residual of forward step has covariance Q (checked statistically below);
    # deterministic check: same key -> identical
    tr2 = data.make_truth(jax.random.fold_in(ROOT, 0), CFG)
    assert np.array_equal(np.asarray(tr2["X"]), X)
    tr3 = data.make_truth(jax.random.fold_in(ROOT, 7), CFG)
    assert not np.array_equal(np.asarray(tr3["X"]), X)
    # crossing scenario: all targets are near [150,150] at the midpoint
    assert np.abs(X[1:, 40, [0, 2]] - 150).max() < 3.0


def test_truth_law_over_2000_draws():
    keys = jax.vmap(lambda i: jax.random.fold_in(ROOT, 100 + i))(np.arange(2000))
    T = jax.vmap(lambda k: data.make_truth(k, CFG)["X"])(keys)  # [2000,4,81,4]
    T = np.asarray(T)
    mid = T[:, 1:, 40, :].reshape(-1, 4)  # targets 2-4 alive at midpoint -> 6000 draws
    assert np.allclose(mid.mean(0), [150, 0, 150, 0], atol=0.02)
    assert np.allclose(mid.var(0), 0.1, rtol=0.1)
    # forward increments: x_{k+1} - F x_k ~ N(0, Q)
    F, Q = CFG["F"], CFG["Q"]
    res = T[:, 1:, 41:, :] - np.einsum("ij,ntkj->ntki", F, T[:, 1:, 40:-1, :])
    res = res.reshape(-1, 4)
    assert np.allclose(res.mean(0), 0, atol=0.01)
    assert np.allclose(np.cov(res.T), Q, atol=0.003)
    # backward: F x_{k-1} - x_k ... x_{k-1} = F^{-1}(x_k + w) -> F x_{k-1} - x_k = w
    resb = np.einsum("ij,ntkj->ntki", F, T[:, 1:, :40, :]) - T[:, 1:, 1:41, :]
    assert np.allclose(np.cov(resb.reshape(-1, 4).T), Q, atol=0.003)


def test_measurement_statistics_and_determinism():
    tr = data.make_truth(jax.random.fold_in(ROOT, 0), CFG)
    pd, lam = 0.8, 12.0
    z, zv, nd = data.make_measurements_runs(ROOT, tr, 5, pd, lam, 200, CFG, 48)
    z, zv = np.asarray(z), np.asarray(zv)
    assert int(np.asarray(nd).sum()) == 0
    # valid first
    assert (np.diff(zv.astype(int), axis=-1) <= 0).all()
    assert (z[~zv] == 0).all()
    alive = np.asarray(tr["alive"])
    n_alive = alive.sum(0)  # [81]
    nmeas = zv.sum(-1)  # [R,K]
    # total count: Binomial detections + Poisson clutter; mean per step
    assert np.allclose(nmeas.mean(0), pd * n_alive + lam, atol=1.2)  # per-step se ~0.25 (200 runs); 81 steps -> 4.5 sigma band
    assert abs(nmeas.mean() - (pd * n_alive.mean() + lam)) < 0.08
    # clutter count mean/var: reconstruct as measurements farther than 6 sigma from every alive target ~ clutter (approx),
    # instead use the exact tree: rerun with pd=0 so that every measurement is clutter
    z0, zv0, _ = data.make_measurements_runs(ROOT, tr, 5, 0.0, lam, 300, CFG, 48)
    c = np.asarray(zv0).sum(-1).ravel()
    assert abs(c.mean() - lam) < 0.1 and abs(c.var() - lam) < 0.6  # 24300 samples
    zc = np.asarray(z0)[np.asarray(zv0)]
    assert zc.min() >= 0 and zc.max() <= 300  # uniform support, never clipped (support is the region itself)
    assert sst.kstest(zc[:, 0] / 300, "uniform").pvalue > 1e-3 and sst.kstest(zc[:, 1] / 300, "uniform").pvalue > 1e-3
    # detections: pd=1, lam=0 -> exactly the alive targets, z = Hx + N(0,I) noise
    z1, zv1, _ = data.make_measurements_runs(ROOT, tr, 0, 1.0, 0.0, 50, CFG, 8)
    assert (np.asarray(zv1).sum(-1) == n_alive[None, :]).all()
    X = np.asarray(tr["X"])
    resid = []
    for k in range(81):
        pos = X[alive[:, k], k][:, [0, 2]]
        resid.append((np.asarray(z1)[:, k, : len(pos)] - pos[None]).reshape(-1, 2))
    resid = np.concatenate(resid)
    assert abs(resid.mean()) < 0.03 and abs(resid.std() - 1) < 0.03
    # detection rate
    z2, zv2, _ = data.make_measurements_runs(ROOT, tr, 0, pd, 0.0, 500, CFG, 8)
    rate = np.asarray(zv2).sum() / (500 * alive.sum())
    assert abs(rate - pd) < 0.01
    # same key -> identical; different config id -> different
    za, zva, _ = data.make_measurements_runs(ROOT, tr, 5, pd, lam, 200, CFG, 48)
    assert np.array_equal(np.asarray(za), z) and np.array_equal(np.asarray(zva), zv)
    zb, _, _ = data.make_measurements_runs(ROOT, tr, 6, pd, lam, 200, CFG, 48)
    assert not np.array_equal(np.asarray(zb), z)
    # single run API matches the batched run with the same key
    cfg_key = jax.random.fold_in(jax.random.fold_in(ROOT, 1), 5)
    zs, zvs, _ = data.make_measurements(jax.random.fold_in(cfg_key, 3), tr, pd, lam, CFG, 48)
    assert np.array_equal(np.asarray(zs), z[3]) and np.array_equal(np.asarray(zvs), zv[3])


def test_dropped_reporting():
    tr = data.make_truth(jax.random.fold_in(ROOT, 0), CFG)
    with pytest.warns(UserWarning):
        _, _, nd = data.make_measurements_runs(ROOT, tr, 0, 0.9, 20.0, 20, CFG, 20)
    assert int(np.asarray(nd).sum()) > 0
    with pytest.raises(ValueError):
        data.make_measurements_runs(ROOT, tr, 0, 0.9, 20.0, 20, CFG, 20, on_drop="raise")


def test_npz_roundtrip_and_hash(tmp_path):
    tr = data.make_truth(jax.random.fold_in(ROOT, 0), CFG)
    z, zv, nd = data.make_measurements_runs(ROOT, tr, 3, 0.9, 10.0, 4, CFG, 48)
    p1, p2, p3 = (str(tmp_path / n) for n in ("a.npz", "b.npz", "t.npz"))
    s1 = data.save_measurements(p1, tr, z, zv, meta=dict(pd=0.9, lambda_c=10.0, config_id=3))
    s2 = data.save_measurements(p2, tr, z, zv, meta=dict(pd=0.9, lambda_c=10.0, config_id=3))
    assert s1 == s2 and s1 == data.sha256_file(p1)  # file hash is reproducible (fixed zip timestamps)
    tr2, z2, zv2, meta = data.load_measurements(p1)
    assert np.array_equal(np.asarray(z2), np.asarray(z)) and np.array_equal(np.asarray(zv2), np.asarray(zv))
    assert np.array_equal(np.asarray(tr2["X"]), np.asarray(tr["X"])) and meta["config_id"] == 3
    with np.load(p1) as d:
        assert d["z"].shape == (4, 81, 48, 2) and d["zvalid"].shape == (4, 81, 48) and d["X"].shape == (4, 81, 4) and d["alive"].shape == (4, 81)
    data.save_truth(p3, tr)
    assert np.array_equal(np.asarray(data.load_truth(p3)["X"]), np.asarray(tr["X"]))
    h = data.sha256_arrays(dict(z=np.asarray(z), zvalid=np.asarray(zv)))
    assert h == data.sha256_arrays(dict(zvalid=np.asarray(zv), z=np.asarray(z))) and len(h) == 64
    # oracle export
    from scipy.io import loadmat
    data.export_oracle_mat(p1, str(tmp_path / "o.mat"), CFG)
    m = loadmat(str(tmp_path / "o.mat"))
    assert m["z"].shape == (4, 81) and m["X_truth"].shape == (16, 81)
    nz = int(np.asarray(zv)[2, 10].sum())
    assert m["z"][2, 10].shape == (2, nz)
    assert np.allclose(m["z"][2, 10], np.asarray(z)[2, 10][:nz].T)
    assert np.allclose(m["X_truth"][4:8, :], np.asarray(tr["X"])[1].T)
    assert list(m["t_death"].ravel()) == [41, 82, 82, 82]


def test_clutter_honours_region_lower_bound():
    """Clutter is uniform on [lo, lo + size], not [0, size]."""
    cfg = dict(CFG)
    cfg["region"] = np.array([[100.0, 400.0], [-50.0, 250.0]])
    tr = data.make_truth(jax.random.fold_in(ROOT, 0), CFG)
    z, zv, nd = data.make_measurements_runs(ROOT, tr, 0, 0.0, 30.0, 20, cfg, 80)      # p_d = 0 -> clutter only
    zc = np.asarray(z)[np.asarray(zv)]
    assert zc.shape[0] > 100
    assert zc[:, 0].min() >= 100.0 and zc[:, 0].max() <= 400.0 and zc[:, 1].min() >= -50.0 and zc[:, 1].max() <= 250.0
    assert zc[:, 0].mean() > 200.0 and zc[:, 1].mean() < 150.0 and zc[:, 1].mean() > 50.0   # centred on the shifted box


def test_zero_counter_rule_flags_every_counter():
    """train.zero_counter_violations is the stop rule; each counter alone must trip it."""
    from src import train
    ok = dict(n_meas_dropped=0, diag=dict(pmbm=dict(ovf=[0] * 6, est_over=0), phd=dict(est_over=0)))
    assert train.zero_counter_violations(ok) == {}
    import copy
    for path, key in ((("n_meas_dropped",), "n_meas_dropped"), (("diag", "pmbm", "est_over"), "pmbm.est_over"),
                      (("diag", "phd", "est_over"), "phd.est_over")):
        r = copy.deepcopy(ok)
        d = r
        for p_ in path[:-1]:
            d = d[p_]
        d[path[-1]] = 1
        assert key in train.zero_counter_violations(r), key
    for i, name in enumerate(("pois_evict", "slot_full", "cap_rows_dropped", "kbest_ovf", "meas_ignored", "child_trunc")):
        r = copy.deepcopy(ok)
        r["diag"]["pmbm"]["ovf"][i] = 2
        assert f"pmbm.ovf.{name}" in train.zero_counter_violations(r)


def test_key_tree_gibbs_branch_does_not_collide_with_step_keys():
    """The reserved Gibbs branch fold_in(run, 0) differs from every step key fold_in(run, k), k = 1..81 (steps start at 1)."""
    run = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(ROOT, 1), 3), 0)
    g = np.asarray(jax.random.key_data(jax.random.fold_in(run, 0)) if hasattr(jax.random, "key_data") and jax.dtypes.issubdtype(run.dtype, jax.dtypes.prng_key) else jax.random.fold_in(run, 0))
    for k in range(1, 82):
        sk = jax.random.fold_in(run, k)
        sk = np.asarray(jax.random.key_data(sk) if jax.dtypes.issubdtype(sk.dtype, jax.dtypes.prng_key) else sk)
        assert not np.array_equal(g, sk), k


def test_paper_pmbm_config_is_the_paper_path():
    """The paper path has H = N_h (no per-track cap can bind) and enough slots; PMBMConfig's own defaults are generic sizes."""
    from src.algorithm import PMBMConfig
    from src.model import paper_pmbm_config
    c = paper_pmbm_config()
    assert c.Nh == 200 and c.H == c.Nh and c.T >= 256 and c.M >= 48 and not c.h_cap
    assert (c.prune_pois, c.prune_r, c.est_gamma) == (1e-5, 1e-5, 0.4)          # paper §VII
    assert PMBMConfig().H < c.H          # the generic default is NOT the paper filter (documented)
    assert c.prune_w == 0.0              # Alg. 1 keeps the top N_h global hypotheses, no weight floor (the authors' HEAD code has 1e-4)
    assert paper_pmbm_config(prune_w=1e-4).prune_w == 1e-4      # labelled sensitivity row


def test_fig3_truth_is_the_papers_scenario():
    """The digitized Fig. 3 truth has the Sec. VII structure (target 1 dies after k = 40, all four meet near (150,150) at k = 41,
    the k = 41 midpoint law N([150,0,150,0], 0.1 I4)) and is pinned by sha256 so a silent re-extraction is caught."""
    import hashlib
    t = data.load_fig3_truth()
    X, alive = np.asarray(t["X"]), np.asarray(t["alive"])
    assert X.shape == (4, 81, 4) and alive.sum(1).tolist() == [40, 81, 81, 81]
    P = X[:, 40, [0, 2]]
    assert np.max(np.abs(P[1:] - 150.0)) < 3.0 and np.max(np.abs(X[0, 39, [0, 2]] - 150.0)) < 5.0      # midpoints ~ N(150, 0.1) (+ one step for target 1)
    step = np.linalg.norm(np.diff(X[:, :, [0, 2]], axis=1), axis=2)
    assert np.all(step[alive[:, 1:] & alive[:, :-1]] < 4.0)                                       # no digitisation jump
    raw = np.load(data.DATA_FIG3_NPZ)
    assert str(raw["pdf_sha256"]).startswith("4b33fd12215c88e4")                                   # the PDF the figure was read from


def test_fig4_curves_reproduce_table2_row():
    """The vector-extracted Fig. 4 curves give Table II (0.9, 10) through the RMS-over-81-steps statistic with k = 1 = 10."""
    import json, os
    J = json.load(open(os.path.join(os.path.dirname(data.DATA_FIG3_NPZ), "fig4_paper_curves.json")))["curves"]
    want = dict(est1=2.23, est2=2.34, est3=2.36, tomb=2.65, momb=2.37, cphd=3.39, phd=7.05)
    for a, w in want.items():
        v = [J[a].get(str(k), 10.0) for k in range(2, 82)]          # PHD gap at k = 41..44 = OSPA 10 in every run
        assert abs(np.sqrt((100.0 + np.sum(np.square(v))) / 81) - w) < 0.01 + (0.01 if a == "phd" else 0.0), a
