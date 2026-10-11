"""recheck oracle check: the authors' ContinualBackprop + AdamGnT, unmodified, in BOTH code versions
(a6b7958 = post-f00ca40 'standard'; repo_pre_f00ca40 = paper-era dcc3619 AdamGnT), on OUR re-sample streams, with the
SAME torch seed per stream (identical init and CBP randomness up to the first divergence).  The paired shift
(paper-era - standard) of the authors' code is compared with ours (recheck_d12_rs_pe - full_scr_d12_rs over the same steps).

    JAX_PLATFORMS=cpu uv run python -m scripts.oracle.prefix_reference [--steps 300000] [--seeds 0 1 2 3] [--procs 3]

Outputs runs/oracle_ref/prefix/<case>_<tree>/seed_<s>.npz and runs/oracle_ref/prefix/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402

from scripts.oracle.reference import ORACLE_PY, RUNNER, export  # noqa: E402
from src.eval import load  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "runs", "oracle_ref", "prefix")
OFF = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
TREES = {"standard": os.path.join(OFF, "repo"),
         "paper_era": os.path.join(OFF, "repo_pre_f00ca40")}
CASES = {"overall_0.01": (0.01, "adaptable_contribution"), "adaptation_0.01": (0.01, "adaptation"),
         "random_0.001": (0.001, "random")}


def run_one(case, tree, seed, steps):
    out = os.path.join(OUT, f"{case}_{tree}", f"seed_{seed}.npz")
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rho, ut = CASES[case]
    full = dict(np.load(export("resample", seed, 1_000_000)))
    inp = dict(X=full["X"][:steps], Y=full["Y"][:steps], act="tanh", opt="adam", lr=0.01, rho=rho, mt=10,
               util_type=ut, wd=0.0, sigma=0.0, torch_seed=10_000 + seed)
    ip = out.replace(".npz", "_in.npz")
    np.savez(ip, **inp)
    t = time.time()
    subprocess.run([ORACLE_PY, RUNNER, "cbp_run", ip, out], check=True, capture_output=True,
                   env={**os.environ, "LOP_ORACLE_REPO": TREES[tree]})
    os.remove(ip)
    print(f"[INFO] prefix_ref {case} {tree} seed={seed} {time.time() - t:.0f}s", flush=True)
    return out


def summarise(seeds, steps):
    nb = steps // 10_000
    res = {"steps": steps, "seeds": seeds, "cases": {}}
    for case in CASES:
        o = {tr: np.array([np.load(os.path.join(OUT, f"{case}_{tr}", f"seed_{s}.npz"))["mse_per_10k"].mean() for s in seeds])
             for tr in TREES}
        ours = {}
        for tr, tag in (("standard", "full_scr_d12_rs"), ("paper_era", "recheck_d12_rs_pe")):
            r = load(tag)
            ours[tr] = np.array([np.mean(r[s]["arms"][case]["mse_per_10k"][:nb]) for s in sorted(r)])
        po, pu = paired_compare(o["paper_era"], o["standard"]), paired_compare(ours["paper_era"], ours["standard"])
        same = {tr: ours[tr][seeds] for tr in ours}      # ours on the SAME streams (seed s = stream s, sorted 0..99)
        pss = paired_compare(same["paper_era"], same["standard"])
        res["cases"][case] = {"official_standard": mean_se(o["standard"]), "official_paper_era": mean_se(o["paper_era"]),
                              "official_shift": [po["diff_mean"], po["diff_se"]], "official_shift_per_seed": (o["paper_era"] - o["standard"]).tolist(),
                              "ours_standard_n100": mean_se(ours["standard"]), "ours_paper_era_n100": mean_se(ours["paper_era"]),
                              "ours_shift_n100": [pu["diff_mean"], pu["diff_se"]],
                              "ours_same_streams_standard": mean_se(same["standard"]), "ours_same_streams_paper_era": mean_se(same["paper_era"]),
                              "ours_same_streams_shift": [pss["diff_mean"], pss["diff_se"]],
                              "ours_same_streams_shift_per_seed": (same["paper_era"] - same["standard"]).tolist(),
                              "official_minus_ours_shift_per_seed": ((o["paper_era"] - o["standard"]) - (same["paper_era"] - same["standard"])).tolist()}
        print(case, json.dumps(res["cases"][case]))
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.summary_only:
        return summarise(a.seeds, a.steps)
    jobs = [(c, tr, s) for c in CASES for tr in TREES for s in a.seeds]
    with ThreadPoolExecutor(a.procs) as ex:
        list(ex.map(lambda j: run_one(*j, a.steps), jobs))
    summarise(a.seeds, a.steps)


if __name__ == "__main__":
    main()
