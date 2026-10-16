"""ORACLE side of the E0 differential test. Runs ONLY in the isolated official venv:

    $OFFICIAL_ROOT/.venv/bin/python \
        scripts/official/rliable_reference.py [--quick]

Reads data/e0/inputs.npz (made by make_inputs.py) and data/e0/dmc_reported.json, calls rliable@3ccd9f4 (+ the
Colab's helper cells, exec'd from the notebook) and writes data/e0/oracle.npz. Nothing from here is imported by src/.

Sections (keys in oracle.npz):
  pt_*      point estimates for all 200 inputs (rliable.metrics, library.*_distributions, scipy trim_mean)
  rp_*      exact replay: the global np.random stream is seeded, rliable's bootstrap is run, and the SAME
            np.random.choice calls are regenerated to recover the indices it used (R7) -> our code replays them
  dist_*    in-distribution: 50,000-rep StratifiedBootstrap of [median, IQM, mean, OG] + conf_int for
            percentile / basic / bc / bca (bca with arch's pooled jackknife mean, R9)
  task_*, poi_*, prof_*, rank_*, stud_*, sub_*: task bootstrap, independent two-sample P(X>Y), profiles,
            Colab get_rank_matrix, Colab mean_CI, Colab subsample_scores_mat_with_replacement
"""
import json
import os
import re
import sys
import time

import numpy as np
import scipy.stats
from rliable import library as rly
from rliable import metrics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "external", "data"))  # large inputs, see README
NB = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "rliable/deep_rl_precipice_colab.ipynb")
QUICK = "--quick" in sys.argv


def colab_defs(names):
    nb = json.load(open(NB))
    ns = {"np": np, "scipy": scipy, "collections": __import__("collections")}
    for c in nb["cells"]:
        src = "".join(c["source"])
        for name in names:
            m = re.search(rf"^def {name}\(.*?(?=^\S)", src + "\nEND", flags=re.S | re.M)
            if m and name not in ns:
                exec(m.group(0), ns)
    return ns


def func4(x):
    return np.array([metrics.aggregate_median(x), metrics.aggregate_iqm(x), metrics.aggregate_mean(x),
                     metrics.aggregate_optimality_gap(x)])


def taus_for(x):
    return np.concatenate([np.linspace(-0.5, 3.0, 36), np.sort(x.ravel())[:: max(1, x.size // 10)][:10]])


def main():
    t0 = time.time()
    z = np.load(os.path.join(DATA_ROOT, "e0", "inputs.npz"))
    n_in = 200
    out = {}
    # ---------------------------------------------------------------- points
    for i in range(n_in):
        x, y = z[f"x{i:03d}"], z[f"y{i:03d}"]
        t = taus_for(x)
        out[f"pt_{i:03d}_agg"] = np.array([metrics.aggregate_mean(x), metrics.aggregate_median(x),
                                           metrics.aggregate_iqm(x), metrics.aggregate_optimality_gap(x),
                                           metrics.aggregate_optimality_gap(x, gamma=0.5)])
        out[f"pt_{i:03d}_trim"] = np.array([scipy.stats.trim_mean(x, a, axis=None) for a in
                                            (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)])
        out[f"pt_{i:03d}_taus"] = t
        out[f"pt_{i:03d}_prof"] = rly.score_distributions(x, t)
        out[f"pt_{i:03d}_avgprof"] = rly.average_score_distributions(x, t)
        out[f"pt_{i:03d}_poi"] = np.array([metrics.probability_of_improvement(x, y),
                                           metrics.probability_of_improvement(y, x),
                                           metrics.probability_of_improvement(x, x)])
    for nm in ("toy_x", "toy_y"):
        x = z[nm]
        out[f"pt_{nm}_agg"] = np.array([metrics.aggregate_mean(x), metrics.aggregate_median(x),
                                        metrics.aggregate_iqm(x), metrics.aggregate_optimality_gap(x)])
    out["pt_toy_poi"] = np.array([metrics.probability_of_improvement(z["toy_x"], z["toy_y"])])
    print(f"[INFO] points done t={time.time() - t0:.1f}s", flush=True)

    # ---------------------------------------------------------------- exact replay (shared RNG stream)
    B = 200
    for i in range(n_in):
        x = z[f"x{i:03d}"]
        np.random.seed(1000 + i)
        res = rly.StratifiedBootstrap(x).apply(func4, B)
        np.random.seed(1000 + i)
        idx = np.stack([np.random.choice(x.shape[0], x.shape, replace=True) for _ in range(B)])
        chk = np.stack([func4(x[ii, np.arange(x.shape[1])[None, :]]) for ii in idx])
        assert np.array_equal(chk, res), f"index regeneration failed on input {i}"
        out[f"rp_{i:03d}_idx"] = idx.astype(np.int32)
        out[f"rp_{i:03d}_res"] = res
    # task bootstrap replay (two choice calls per replicate) on 20 inputs
    for i in range(0, n_in, 10):
        x = z[f"x{i:03d}"]
        N, M = x.shape
        np.random.seed(5000 + i)
        res = rly.StratifiedBootstrap(x, task_bootstrap=True).apply(func4, B)
        np.random.seed(5000 + i)
        ri, ti = [], []
        for _ in range(B):
            ri.append(np.random.choice(N, x.shape, replace=True))
            ti.append(np.random.choice(M, (1, M), replace=True)[0])
        chk = np.stack([func4(x[a, b[None, :]]) for a, b in zip(ri, ti)])
        assert np.array_equal(chk, res), f"task replay failed on input {i}"
        out[f"rptask_{i:03d}_idx"] = np.stack(ri).astype(np.int32)
        out[f"rptask_{i:03d}_tidx"] = np.stack(ti).astype(np.int32)
        out[f"rptask_{i:03d}_res"] = res
    # independent two-sample replay (P(X>Y)) on 20 pairs
    for i in range(0, n_in, 10):
        x, y = z[f"x{i:03d}"], z[f"y{i:03d}"]
        np.random.seed(7000 + i)
        bs = rly.StratifiedIndependentBootstrap(x, y)
        res = bs.apply(lambda a, b: np.array([metrics.probability_of_improvement(a, b)]), B)
        np.random.seed(7000 + i)
        ix, iy = [], []
        for _ in range(B):
            ix.append(np.random.choice(x.shape[0], x.shape, replace=True))
            iy.append(np.random.choice(y.shape[0], y.shape, replace=True))
        cols = lambda a: np.arange(a.shape[1])[None, :]  # noqa: E731
        chk = np.array([[metrics.probability_of_improvement(x[a, cols(x)], y[b, cols(y)])] for a, b in zip(ix, iy)])
        assert np.array_equal(chk, res), f"poi replay failed on input {i}"
        out[f"rppoi_{i:03d}_ix"] = np.stack(ix).astype(np.int32)
        out[f"rppoi_{i:03d}_iy"] = np.stack(iy).astype(np.int32)
        out[f"rppoi_{i:03d}_res"] = res
    print(f"[INFO] replay done t={time.time() - t0:.1f}s", flush=True)

    # ---------------------------------------------------------------- in distribution, 50,000 reps
    REPS = 2000 if QUICK else 50000
    sel = [(17 * j) % n_in for j in range(25)]  # 25 inputs covering all 16 (N, M) shapes and all 5 generators
    out["meta_sel"] = np.array(sel)
    for i in sel:
        x = z[f"x{i:03d}"]
        np.random.seed(9000 + i)
        bs = rly.StratifiedBootstrap(x)
        ci = {"percentile": bs.conf_int(func4, reps=REPS, method="percentile", size=0.95)}
        out[f"dist_{i:03d}_res"] = bs._results.copy()
        for m in ("basic", "bc", "bca"):
            try:
                ci[m] = bs.conf_int(func4, reps=REPS, method=m, size=0.95, reuse=True)
            except RuntimeError as e:  # arch: bias-correction proportion 0 or 1, or jackknife variance ~0
                ci[m] = np.full((2, 4), np.nan)
                print(f"[WARN] input {i} method {m}: {str(e)[:80]}", flush=True)
        for m, v in ci.items():
            out[f"dist_{i:03d}_ci_{m}"] = v
        print(f"[INFO] dist input {i} shape={x.shape} t={time.time() - t0:.1f}s", flush=True)
    # single-statistic bca (per-output jackknife mean, no pooling quirk) via get_interval_estimates
    for i in sel[:8]:
        x = z[f"x{i:03d}"]
        np.random.seed(11000 + i)
        pe, ie = rly.get_interval_estimates({"a": x}, lambda s: np.array([metrics.aggregate_iqm(s)]),
                                            method="bca", reps=10000 if not QUICK else 1000)
        out[f"bca1_{i:03d}_ci"] = ie["a"]
    # task bootstrap in distribution
    for i in sel[:6]:
        x = z[f"x{i:03d}"]
        np.random.seed(13000 + i)
        bs = rly.StratifiedBootstrap(x, task_bootstrap=True)
        out[f"task_{i:03d}_ci"] = bs.conf_int(func4, reps=REPS, method="percentile")
        out[f"task_{i:03d}_res"] = bs._results.copy()
    # P(X>Y) intervals (2,000 reps, A.26) and profiles (2,000 reps) through the public API
    for i in sel[:10]:
        x, y = z[f"x{i:03d}"], z[f"y{i:03d}"]
        np.random.seed(15000 + i)
        pe, ie = rly.get_interval_estimates({"p": [x, y]}, lambda a, b: np.array([metrics.probability_of_improvement(a, b)]),
                                            reps=2000)
        out[f"poi_{i:03d}_pe"], out[f"poi_{i:03d}_ci"] = np.atleast_1d(pe["p"]), ie["p"]
        np.random.seed(17000 + i)
        t = np.linspace(-0.5, 3.0, 36)
        prof, pci = rly.create_performance_profile({"a": x}, t, reps=2000)
        out[f"prof_{i:03d}_pt"], out[f"prof_{i:03d}_ci"] = prof["a"], pci["a"]
    print(f"[INFO] intervals done t={time.time() - t0:.1f}s", flush=True)

    # ---------------------------------------------------------------- Colab helpers
    ns = colab_defs(["get_rank_matrix", "subsample_scores_mat", "mean_CI", "subsample_scores_mat_with_replacement"])
    algs = {"a": z["x002"], "b": z["y002"], "c": z["x006"][:, :z["x002"].shape[1]]}
    np.random.seed(19000)
    out["rank_mat"] = ns["get_rank_matrix"](algs, n=200000 if not QUICK else 20000, algorithms=["a", "b", "c"])
    dmc = json.load(open(os.path.join(REPO, "data", "e0", "dmc_reported.json")))
    for steps in ("100k", "500k"):
        for alg in ("CURL", "SUNRISE", "CURL-D2RL"):
            n = 5 if alg == "SUNRISE" else 10
            sd = np.sqrt(np.mean(np.square(dmc["stds"][steps][alg])))
            mu = np.mean(dmc["means"][steps][alg])
            out[f"stud_{steps}_{alg}"] = np.concatenate([[mu / 1000], ns["mean_CI"](mu, sd, n)])
    x = z["x003"]  # (100, 6) pool
    np.random.seed(21000)
    subs = np.stack([ns["subsample_scores_mat_with_replacement"](x, num_samples=5) for _ in range(50)])
    np.random.seed(21000)
    sidx = np.stack([np.random.choice(x.shape[0], size=(5, x.shape[1]), replace=True) for _ in range(50)])
    assert np.array_equal(subs, np.stack([x[s, np.arange(x.shape[1])[None, :]] for s in sidx]))
    out["sub_idx"], out["sub_res"] = sidx.astype(np.int32), subs
    # ALE (Dopamine 5-run final scores, human-normalised by our data.py and saved to data/e0/ale_final.npz)
    ale_p = os.path.join(REPO, "data", "e0", "ale_final.npz")
    if os.path.exists(ale_p):
        ale = np.load(ale_p)
        names = sorted(ale.files)
        for j, nm in enumerate(names):
            out[f"ale_{j}_pt"] = func4(ale[nm])
            np.random.seed(23000 + j)
            out[f"ale_{j}_ci"] = rly.StratifiedBootstrap(ale[nm]).conf_int(func4, reps=REPS, method="percentile")
        for j, a in enumerate(names):
            for k, b in enumerate(names):
                if a != b:
                    out[f"ale_poi_{j}_{k}"] = np.array([metrics.probability_of_improvement(ale[a], ale[b])])
        out["ale_names"] = np.array(names)
        print(f"[INFO] ale done t={time.time() - t0:.1f}s", flush=True)
    out["meta_reps"] = np.array(REPS)
    path = os.path.join(DATA_ROOT, "e0", "oracle_quick.npz" if QUICK else "oracle.npz")
    np.savez(path, **out)
    import hashlib
    print(f"[INFO] wrote {path} sha256={hashlib.sha256(open(path, 'rb').read()).hexdigest()} t={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
