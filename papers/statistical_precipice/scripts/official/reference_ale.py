"""ORACLE reference run (standing rule 2, the official-code check): the authors' own pipeline end to end on the only public per-run
data that survives the withdrawn bucket -- Dopamine's baselines/atari/data/<game>.json. Runs ONLY in the isolated
official venv (rliable@3ccd9f4, arch 7.0.0, numpy 1.26.4, scipy 1.13.1, pandas 2.2.2, inflection 0.5.1):

    $OFFICIAL_ROOT/.venv/bin/python -I \
        scripts/official/reference_ale.py [--quick]

Independence from our code: nothing from src/ is imported. The raw JSON is parsed here with pandas; run identity is
recovered as "a new run starts when Iteration == 0" (same rule as , re-implemented); human/random scores and
the normaliser come from the Colab's own cells 59 (MuZero table, Surround/Defender removed), 60 (record scores) and
61 (score_normalization_all_atari / normalized_scores / convert_to_matrix), exec'd from the notebook. The ALE figures
are then computed exactly as the Colab cells 63, 66/67, 68, 70 and 72 do (reps as in the Colab).

Differences forced by the data (not choices): the Dopamine files hold 199 iterations (0..198), the bucket's
atari_200_iters_scores.npy held 200, so "final" = iteration 198 and the Fig 10 frame 200 maps to index 198.

Output: data/official/reference_ale.npz (+ .json summary). Keys documented inline.
"""
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import scipy.stats
from rliable import library as rly
from rliable import metrics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "external", "data"))  # large inputs, see README
NB = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "rliable/deep_rl_precipice_colab.ipynb")
QUICK = "--quick" in sys.argv
DOP = os.path.join(DATA_ROOT, "dopamine")
# Dopamine label -> the paper's Fig 9 label (same mapping as the Colab's legend)
AGENTS = {"DQN": "DQN (Nature)", "DQN (Adam + MSE in JAX)": "DQN (Adam)", "C51": "C51", "IQN": "IQN",
          "Rainbow": "Rainbow"}


def colab_cells(idx):
    nb = json.load(open(NB))
    ns = {"np": np, "inflection": __import__("inflection")}
    for i in idx:
        src = "".join(nb["cells"][i]["source"])
        src = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("!", "%")))
        exec(src, ns)
    return ns


def parse(path):
    """{agent: (runs, iterations)} -- pandas re-implementation of the run-identity rule."""
    df = pd.DataFrame(json.load(open(path)))
    out = {}
    for agent, g in df.groupby("Agent", sort=False):
        g = g.reset_index(drop=True)
        run = (g["Iteration"] == 0).cumsum() - 1
        g = g.assign(run=run)
        piv = g.pivot(index="run", columns="Iteration", values="Value")
        assert not piv.isna().any().any(), (path, agent)
        assert (g.groupby("run")["Iteration"].apply(lambda s: list(s) == list(range(len(s))))).all(), (path, agent)
        out[agent] = piv.to_numpy(dtype=np.float64)
    return out


def main():
    t0 = time.time()
    ns = colab_cells([59, 60, 61])
    HUMAN, RANDOM = ns["ALL_HUMAN_SCORES"], ns["ALL_RANDOM_SCORES"]
    files = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join(DOP, "*.json"))}
    games = sorted(HUMAN)  # Colab: convert_to_matrix stacks sorted(keys)
    assert len(games) == 55, len(games)
    lower = {g: g.lower() for g in games}
    assert all(lower[g] in files for g in games), [g for g in games if lower[g] not in files]
    raw = {g: parse(files[lower[g]]) for g in games}
    out = {"games": np.array(games), "human": np.array([HUMAN[g] for g in games]),
           "random": np.array([RANDOM[g] for g in games]),
           "record": np.array([ns["ALL_HUMAN_RECORD"][g] for g in games])}
    final, iters, final_rec = {}, {}, {}
    for d, p in AGENTS.items():
        cube = {g: raw[g][d] for g in games}  # (runs, iters)
        n_it = {v.shape for v in cube.values()}
        assert n_it == {(5, 199)}, (d, n_it)
        final[p] = ns["normalized_scores"]({g: v[:, -1] for g, v in cube.items()}, record_scores=False)
        final_rec[p] = ns["normalized_scores"]({g: v[:, -1] for g, v in cube.items()}, record_scores=True)
        # Colab cell 62 stores iteration-first arrays and transposes (1, 2, 0); we hand it (iters, runs)
        it = ns["normalized_scores"]({g: v.T for g, v in cube.items()}, record_scores=False)
        iters[p] = np.transpose(it, axes=(1, 2, 0))  # (runs, games, iterations)
    algs = list(AGENTS.values())
    for j, a in enumerate(algs):
        out[f"norm_{j}"] = final[a]
        out[f"normrec_{j}"] = final_rec[a]
    out["algs"] = np.array(algs)
    print(f"[INFO] parsed + normalised t={time.time() - t0:.1f}s", flush=True)

    MEDIAN, IQM, MEAN, OG = (metrics.aggregate_median, metrics.aggregate_iqm, metrics.aggregate_mean,
                             metrics.aggregate_optimality_gap)
    agg = lambda x: np.array([MEDIAN(x), IQM(x), MEAN(x), OG(x)])  # noqa: E731  (Colab cell 66/67)
    R_AGG = 2000 if QUICK else 50000
    # Fig 9 -- three bootstrap seeds (the bucket-era Colab seeds nothing; R7: rliable uses the global np.random)
    for s in (0, 1, 2):
        np.random.seed(s)
        pe, ie = rly.get_interval_estimates(final, agg, reps=R_AGG)
        for j, a in enumerate(algs):
            out[f"fig9_s{s}_{j}_pt"], out[f"fig9_s{s}_{j}_ci"] = pe[a], ie[a]
        print(f"[INFO] fig9 seed {s} t={time.time() - t0:.1f}s", flush=True)
    # Fig A.20 -- task + run bootstrap (cell 67)
    np.random.seed(0)
    pe, ie = rly.get_interval_estimates(final, agg, reps=R_AGG, task_bootstrap=True)
    for j, a in enumerate(algs):
        out[f"a20_{j}_ci"] = ie[a]
    # Fig 10 left / A.22 -- profiles, tau = linspace(0, 8, 81), reps 2000 (cell 63)
    tau = np.linspace(0.0, 8.0, 81)
    np.random.seed(0)
    prof, pci = rly.create_performance_profile(final, tau, reps=2000)
    out["tau"] = tau
    for j, a in enumerate(algs):
        out[f"prof_{j}"], out[f"prof_{j}_ci"] = prof[a], pci[a]
    # Fig 10 right -- IQM at frames {1, 10, 25, ..., 200}M, reps 2000 (cell 68); frame 200 -> index 198 here
    frames = np.array([1, 10, 25, 50, 75, 100, 125, 150, 175, 199]) - 1
    out["fig10_frames_idx"] = frames
    sub = {a: v[:, :, frames] for a, v in iters.items()}
    iqm_t = lambda sc: np.array([metrics.aggregate_iqm(sc[..., f]) for f in range(sc.shape[-1])])  # noqa: E731
    np.random.seed(0)
    pe, ie = rly.get_interval_estimates(sub, iqm_t, reps=2000)
    for j, a in enumerate(algs):
        out[f"fig10_{j}_pt"], out[f"fig10_{j}_ci"] = pe[a], ie[a]
    print(f"[INFO] profiles + fig10 t={time.time() - t0:.1f}s", flush=True)
    # A.25 -- DP-25 / DP-50 / superhuman probability (cell 70) on human- and record-normalised scores
    ns70 = colab_cells([])  # fresh namespace for cell 70's helper
    nb = json.load(open(NB))
    exec(re.search(r"^def difficulty_progress\(.*?(?=^\S)", "".join(nb["cells"][70]["source"]) + "\nEND",
                   flags=re.S | re.M).group(0), ns70)
    dp = ns70["difficulty_progress"]
    alt = lambda x: np.array([dp(x, 0.25), dp(x, 0.5), np.mean(x > 1)])  # noqa: E731
    np.random.seed(0)
    pe, ie = rly.get_interval_estimates(final, alt, reps=R_AGG)
    for j, a in enumerate(algs):
        out[f"a25_{j}_pt"], out[f"a25_{j}_ci"] = pe[a], ie[a]
    # A.27 -- P(X > Y), every ordered pair, reps 1000 (cell 72's value; the caption says 2,000)
    pairs = {f"{x}|{y}": (final[x], final[y]) for x in algs for y in algs if x != y}
    np.random.seed(0)
    pe, ie = rly.get_interval_estimates(pairs, metrics.probability_of_improvement, reps=1000)
    names = sorted(pairs)
    out["poi_names"] = np.array(names)
    out["poi_pt"] = np.array([float(pe[k]) for k in names])
    out["poi_ci"] = np.stack([np.ravel(ie[k]) for k in names])
    print(f"[INFO] A.25 + A.27 t={time.time() - t0:.1f}s", flush=True)

    # Diagnostic for the DQN rows (standing rule 1): which iteration / smoothing would the paper's numbers need?
    for j, a in enumerate(algs):
        out[f"diag_iqm_by_iter_{j}"] = np.array([IQM(iters[a][:, :, i]) for i in range(iters[a].shape[-1])])
        out[f"diag_mean_by_iter_{j}"] = np.array([MEAN(iters[a][:, :, i]) for i in range(iters[a].shape[-1])])

    # Exact CI-construction replay: arch conf_int (percentile / basic / bc / bca) on a seeded stream; the SAME
    # np.random.choice calls are regenerated so our ci_from_dist can be fed the identical replicate set.
    REPLAY_B = 500
    for j, a in enumerate(algs):
        x = final[a]
        for m in ("percentile", "basic", "bc", "bca"):
            np.random.seed(31000 + j)
            bs = rly.StratifiedBootstrap(x)
            out[f"cir_{j}_{m}"] = bs.conf_int(agg, reps=REPLAY_B, method=m, size=0.95)
            res = bs._results.copy()
        np.random.seed(31000 + j)
        idx = np.stack([np.random.choice(x.shape[0], x.shape, replace=True) for _ in range(REPLAY_B)])
        chk = np.stack([agg(x[ii, np.arange(x.shape[1])[None, :]]) for ii in idx])
        assert np.array_equal(chk, res), f"replay failed for {a}"
        out[f"cir_{j}_idx"] = idx.astype(np.int8)
        out[f"cir_{j}_res"] = res
    print(f"[INFO] CI replay t={time.time() - t0:.1f}s", flush=True)

    out["meta_reps_agg"] = np.array(R_AGG)
    os.makedirs(os.path.join(DATA_ROOT, "official"), exist_ok=True)
    path = os.path.join(DATA_ROOT, "official", "reference_ale_quick.npz" if QUICK else "reference_ale.npz")
    np.savez(path, **out)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    summ = {"rliable_commit": "3ccd9f4dea577a04d3d2b557f259aac08badbd81", "reps_agg": R_AGG, "wall_s": time.time() - t0,
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "npz_sha256": sha,
            "fig9": {a: {"pt": out[f"fig9_s0_{j}_pt"].tolist(),
                         "ci_seed": {s: out[f"fig9_s{s}_{j}_ci"].tolist() for s in (0, 1, 2)}}
                     for j, a in enumerate(algs)}}
    json.dump(summ, open(path[:-4] + ".json", "w"), indent=1)
    print(f"[INFO] wrote {path} sha256={sha} t={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
