"""Metric tests.

Stone Soup 1.9.1 cross-check: reference values are cached in tests/ss_ref_cache.npz; a live re-run uses the python named by
$STONESOUP_PYTHON (an environment with stonesoup==1.9.1) in a subprocess running tests/ss_ref.py.
Conventions, characterised from the Stone Soup source (stonesoup/metricgenerator/ospametric.py):
  * OSPAMetric.compute_OSPA_distance: ((1/n) * sum(cost[row,col]**p))**(1/p) on the square cost matrix padded with c,
    n = max(m, n); identical to our ospa(). Stone Soup returns 0 for two empty sets only after a timestamp check that
    raises for empty/empty input via GOSPA, so empty/empty and one-empty cases are tested against the closed form (0, c).
  * GOSPAMetric.compute_gospa_metric: alpha is hard-coded to 2; distance = (sum assigned min(d,c)^p + (c^p/2)(#unassigned))^(1/p)
    (a pair at the cut-off is one miss + one false, c^p/2 each); 'localisation','missed','false' are the UN-ROOTED p-th power cost
    components (distance^p = loc + missed + false). This is exactly the official GOSPA.m decomposition and our gospa()
    {d, loc, missed, false}. With switching_penalty = 0 (default) no switching term enters. The base distance is Euclidean
    on the state vectors; we pass the 2-d position vectors, so no extra mapping is needed.
  Therefore the comparison needs NO normalisation mapping: values agree directly to <= 1e-9.
"""
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import eval as ev  # noqa: E402

SS_PY = os.environ.get("STONESOUP_PYTHON", "")
HERE = os.path.dirname(os.path.abspath(__file__))


def test_ospa_hand_example():
    # one-step toy of tests/test_grads.py: estimate {0.75, 2.4} vs truth {1.0} (1-d, embedded in the first coordinate)
    est = np.array([[0.75, 0.0], [2.4, 0.0]])
    tru = np.array([[1.0, 0.0]])
    assert ev.ospa(est, tru) == pytest.approx(np.sqrt((0.25 ** 2 + 100) / 2), abs=1e-12)
    assert ev.ospa(est, tru) == pytest.approx(7.07328, abs=1e-5)
    assert ev.ospa(tru, est) == pytest.approx(ev.ospa(est, tru), abs=1e-14)  # symmetric


def test_empty_sets():
    a = np.array([[1.0, 2.0]])
    assert ev.ospa(np.zeros((0, 2)), np.zeros((0, 2))) == 0.0
    assert ev.ospa(a, np.zeros((0, 2))) == 10.0 and ev.ospa(np.zeros((0, 2)), a) == 10.0
    g = ev.gospa(np.zeros((0, 2)), np.zeros((0, 2)))
    assert g["d"] == 0.0
    g = ev.gospa(a, np.zeros((0, 2)))
    assert g["d"] == pytest.approx(np.sqrt(50.0)) and g["false"] == 50.0 and g["missed"] == 0.0
    g = ev.gospa(np.zeros((0, 2)), np.vstack([a, a]))
    assert g["missed"] == 100.0 and g["false"] == 0.0


def test_gospa_decomposition_cutoff():
    # one pair beyond the cut-off: counts as one miss + one false (c^2/2 each); one close pair: localisation
    est = np.array([[0.0, 0.0], [100.0, 0.0]])
    tru = np.array([[3.0, 4.0], [0.0, 100.0]])
    g = ev.gospa(est, tru)
    assert g["loc"] == pytest.approx(25.0) and g["missed"] == pytest.approx(50.0) and g["false"] == pytest.approx(50.0)
    assert g["d"] ** 2 == pytest.approx(g["loc"] + g["missed"] + g["false"])
    # OSPA of the same pair of sets: (25 + 100)/2
    assert ev.ospa(est, tru) ** 2 == pytest.approx(62.5)


def _random_pairs(rng, n):
    E, T = [], []
    for _ in range(n):
        m, k = rng.integers(0, 7), rng.integers(0, 7)
        if m == 0 and k == 0:
            m = 1
        scale = rng.choice([3.0, 8.0, 20.0])
        base = rng.uniform(0, 30, (max(m, k), 2))
        e = base[:m] + rng.normal(0, scale, (m, 2))
        t = base[:k] + rng.normal(0, scale / 4, (k, 2))
        E.append(e)
        T.append(t)
    return E, T


def _ospa_with_sumd_assignment(e, t, p, c):
    """OSPA evaluated at the assignment that minimises sum of DISTANCES (Stone Soup's choice), not of d^p."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    d = np.minimum(cdist(e, t), c)
    n = max(len(e), len(t))
    sq = np.full((n, n), c)
    sq[: len(e), : len(t)] = d
    r, cc = linear_sum_assignment(sq)
    return float(((sq[r, cc] ** p).sum() / n) ** (1.0 / p))


def test_vs_stonesoup_191():
    """Cross-check on 240 random set pairs, p in {1, 2}, c = 10.

    GOSPA (alpha = 2): agrees with Stone Soup directly (distance + localisation/missed/false), <= 1e-9, for p=1 and 2.
    OSPA: agrees directly for p = 1. For p = 2 Stone Soup 1.9.1 DEVIATES from the OSPA definition: compute_OSPA_distance picks the
    assignment with linear_sum_assignment(cost_matrix) on the plain distances (cost = min(d, c)) and only then raises the chosen
    costs to the power p, i.e. it minimises sum d rather than sum d^p. Its value is therefore >= the true OSPA and equals it only
    when both criteria pick the same assignment. We assert (i) ours <= Stone Soup (we are optimal; brute force checked in
    test_ospa_bruteforce), (ii) Stone Soup == our OSPA evaluated at Stone Soup's sum-d assignment, to 1e-9, (iii) the
    two coincide on most pairs. Reference values come from tests/ss_ref_cache.npz (regenerated live with PMBM_SS_LIVE=1 or if
    missing) so that the ~12 s Stone Soup import is not paid on every run.
    """
    rng = np.random.default_rng(1)
    E, T = _random_pairs(rng, 240)
    cache = os.path.join(HERE, "ss_ref_cache.npz")
    ref = None
    if os.path.exists(cache) and not os.environ.get("PMBM_SS_LIVE"):
        with np.load(cache) as z:
            ref = z["ref"]
    if ref is None:
        if not SS_PY or not os.path.exists(SS_PY):
            pytest.skip("tests/ss_ref_cache.npz is missing and $STONESOUP_PYTHON does not point at a python with stonesoup==1.9.1")
        Eo = np.empty(len(E), dtype=object)
        To = np.empty(len(T), dtype=object)
        for i in range(len(E)):
            Eo[i], To[i] = E[i], T[i]
        with tempfile.TemporaryDirectory() as td:
            inp, outp = os.path.join(td, "in.npz"), os.path.join(td, "out.npy")
            np.savez(inp, E=Eo, T=To, c=10.0)
            r = subprocess.run([SS_PY, "-I", os.path.join(HERE, "ss_ref.py"), inp, outp], capture_output=True, text=True, cwd=td, timeout=300)
            assert r.returncode == 0, r.stderr[-2000:]
            ref = np.load(outp)
        np.savez(cache, ref=ref, stonesoup="1.9.1")
    n_cut = n_same = 0
    for i, (e, t) in enumerate(zip(E, T)):
        for ip, p in enumerate((1.0, 2.0)):
            g = ev.gospa(e, t, p=p)
            assert g["d"] == pytest.approx(ref[i, ip, 1], abs=1e-9), (i, p)
            assert g["loc"] == pytest.approx(ref[i, ip, 2], abs=1e-9), (i, p)
            assert g["missed"] == pytest.approx(ref[i, ip, 3], abs=1e-9), (i, p)
            assert g["false"] == pytest.approx(ref[i, ip, 4], abs=1e-9), (i, p)
            o = ev.ospa(e, t, p=p)
            if p == 1.0:
                assert o == pytest.approx(ref[i, ip, 0], abs=1e-9), i
            else:
                ss = ref[i, ip, 0]
                if len(e) and len(t):
                    assert o <= ss + 1e-9, i
                    assert ss == pytest.approx(_ospa_with_sumd_assignment(e, t, p, 10.0), abs=1e-9), i
                else:
                    assert o == pytest.approx(ss, abs=1e-9), i
                n_same += abs(o - ss) < 1e-9
        n_cut += int(ev.gospa(e, t)["missed"] > 0 and ev.gospa(e, t)["false"] > 0)
    assert n_cut > 20  # the sample exercises cut-off pairs and cardinality mismatches
    assert n_same > 150  # the criteria coincide on most pairs; the deviation is real but rare-ish


def test_ospa_bruteforce():
    """Our OSPA (p=2) is the true minimum over all assignments (the case Stone Soup gets wrong)."""
    import itertools
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(11)
    for _ in range(60):
        m, n = rng.integers(1, 6), rng.integers(1, 6)
        e, t = rng.uniform(0, 25, (m, 2)), rng.uniform(0, 25, (n, 2))
        k = max(m, n)
        D = np.full((k, k), 10.0)
        D[:m, :n] = np.minimum(cdist(e, t), 10.0)
        best = min(sum(D[i, p[i]] ** 2 for i in range(k)) for p in itertools.permutations(range(k)))
        assert ev.ospa(e, t) == pytest.approx(np.sqrt(best / k), abs=1e-10)


def test_evaluate_run_matches_standalone():
    rng = np.random.default_rng(3)
    K, E = 12, 6
    X = np.zeros((3, K, 4))
    X[..., [0, 2]] = rng.uniform(0, 40, (3, K, 2))
    alive = rng.random((3, K)) < 0.7
    X[~alive] = 0
    mask = rng.random((K, E)) < 0.4
    ex = rng.uniform(0, 40, (K, E, 4))
    r = ev.evaluate_run(mask, ex, dict(X=X, alive=alive))
    for k in range(K):
        e = ex[k][mask[k]][:, [0, 2]]
        t = X[alive[:, k], k][:, [0, 2]]
        assert r["ospa2"][k] == pytest.approx(ev.ospa(e, t) ** 2, abs=1e-10)
        g = ev.gospa(e, t)
        assert r["gospa2"][k] == pytest.approx(g["d"] ** 2, abs=1e-10)
        assert r["g_loc"][k] + r["g_mis"][k] + r["g_fal"][k] == pytest.approx(r["gospa2"][k], abs=1e-10)
        assert r["g_mis"][k] == pytest.approx(g["missed"], abs=1e-10) and r["g_fal"][k] == pytest.approx(g["false"], abs=1e-10)
    r2 = ev.evaluate_run(mask, ex, dict(X=X, alive=alive), k_axis=[2, 5])
    assert np.allclose(r2["ospa2"], r["ospa2"][[2, 5]])


def test_evaluate_run_speed():
    import time
    rng = np.random.default_rng(4)
    K, E = 81, 16
    X = np.zeros((4, K, 4))
    X[..., [0, 2]] = rng.uniform(100, 200, (4, K, 2))
    alive = np.ones((4, K), bool)
    mask = rng.random((K, E)) < 0.25
    ex = rng.uniform(100, 200, (K, E, 4))
    t = time.time()
    for _ in range(20):
        ev.evaluate_run(mask, ex, dict(X=X, alive=alive))
    per_run = (time.time() - t) / 20
    assert per_run < 0.02  # 1500 runs (15 configs x 100) x 2 algorithms -> well under a minute


def test_aggregation():
    rng = np.random.default_rng(5)
    d2 = rng.gamma(2.0, 3.0, (100, 81))
    assert ev.joint_rms(d2) == pytest.approx(np.sqrt(d2.sum() / (81 * 100)))
    assert np.allclose(ev.rms_per_step(d2), np.sqrt((d2 ** 2 * 0 + d2).mean(0)))
    s = ev.run_stat(d2)
    assert np.allclose(s, d2.mean(1))
    # delta-method SE vs bootstrap SE: agree within ~10 %
    b = ev.boot_se_joint(s, n_boot=10000, seed=0)
    assert abs(b["se"] / ev.se_delta(s) - 1) < 0.1
    assert b["ci95"][0] < ev.joint_rms(d2) < b["ci95"][1]
    # reproducible
    assert ev.boot_se_joint(s, 2000, 7) == ev.boot_se_joint(s, 2000, 7)
    # per-step bootstrap SE ~ analytic delta SE of the per-step RMS
    se_b = ev.boot_se_rms_per_step(d2, n_boot=2000, seed=1)
    se_d = d2.std(0, ddof=1) / (2 * np.sqrt(100) * np.sqrt(d2.mean(0)))
    assert abs(np.median(se_b / se_d) - 1) < 0.1
    cell = ev.table_cell(d2, n_boot=2000)
    assert cell["n"] == 100 and cell["value"] == pytest.approx(ev.joint_rms(d2))


def test_stats_wrappers():
    rng = np.random.default_rng(6)
    a = rng.normal(5.0, 1.0, 100)
    b = a + 0.3 + rng.normal(0, 0.3, 100)
    r = ev.paired_test(a, b, n_boot=500)
    assert r["diff_mean"] < 0 and r["t_p"] < 1e-6 and r["boot95"][1] < 0  # a lower than b
    h = ev.holm_adjust({"x": 0.01, "y": 0.04, "z": 0.5})
    assert h["x"]["p_holm"] == pytest.approx(0.03) and h["y"]["p_holm"] == pytest.approx(0.08) and h["z"]["p_holm"] == pytest.approx(0.5)
    lo, hi = ev.boot_ci_stat(a, n=500)
    assert lo < a.mean() < hi
