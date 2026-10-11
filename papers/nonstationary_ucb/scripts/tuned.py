"""X3-confirm: write src/configs/tuned_e{1,2}.yaml from the X3 grid argmins (seeds 0-4) for a re-run on seeds 20-24.

    uv run python scripts/tuned.py      (after the sweep runs; reads runs/full/sweep_e*/)

Selection = argmin of the mean final regret over seeds 0-4 x 1000 reps within each arm's 12-13 configurations
(pre-declared, README Full-scale). The confirmation seeds are never used for selection.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ENV = {"e1": ["env:", "  kind: abrupt", "  T: 10000                    # paper §5", "  breakpoints: [3000, 5000]   # paper §5; ",
              "  segments:                   # ", "    - [0.5, 0.3, 0.4]", "    - [0.5, 0.9, 0.4]", "    - [0.5, 0.3, 0.4]"],
       "e2": ["env:", "  kind: periodic", "  T: 10000                    # paper §5", "  base: 0.5                   # paper §5",
              "  amp: 0.4                    # paper §5", "  periods: 3                  # ASSUMED"]}

# Default: the pre-declared one-factor sweep -> tuned_e*. "--joint": the joint grid (sweepj_e*) -> tunedj_e*.
SRC, DST = ("sweepj", "tunedj") if "--joint" in sys.argv else ("sweep", "tuned")
for e, exp_id in (("e1", 1), ("e2", 2)):
    recs = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "runs", "full", f"{SRC}_{e}", "seed_*", "metrics.json"))]
    assert sorted(r["seed"] for r in recs) == [0, 1, 2, 3, 4], "selection must use seeds 0-4 only"
    recs.sort(key=lambda r: r["seed"])
    cfgp = dict(recs[0]["config"]["policies"])
    if SRC == "sweepj":  # merge the small-xi extension (sweepjx) into the joint grid, paired by seed
        xr = sorted([json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "runs", "full", f"sweepjx_{e}", "seed_*", "metrics.json"))],
                    key=lambda r: r["seed"])
        assert [r["seed"] for r in xr] == [0, 1, 2, 3, 4], "run scripts/full.sh sweepjx first"
        for r, x in zip(recs, xr):
            assert r["env"]["uniforms_fp_sha256"] == x["env"]["uniforms_fp_sha256"]
            r["policies"].update(x["policies"])
        cfgp.update(xr[0]["config"]["policies"])
    lines = [f"# X3-confirm: every arm at its X3 grid argmin on {e.upper()} (EXTENSION, labelled; written by scripts/tuned.py).",
             f"# Selected on seeds 0-4 (runs/full/{SRC}_{e}); evaluated on seeds 20-24 only. Same exp_id/cfg_idx as {e}.yaml (CRN).",
             f"name: {DST}_{e}", f'label: "X3-confirm tuned arms ({SRC}) on {e.upper()} (EXTENSION)"', f"exp_id: {exp_id}                     # = {e}.yaml (CRN)", f"cfg_idx: 0                    # = {e}.yaml",
             "reps_per_seed: 2000           # ASSUMED", "log_every: 100               # ASSUMED (final value only)", "tie: random                   # ASSUMED",
             "report_t: [10000]"] + ENV[e] + ["policies:"]
    for algo in ("ucb1", "exp3s", "ducb", "swucb"):
        keys = [k for k, v in cfgp.items() if v["algo"] == algo]
        m = {k: np.mean([r["policies"][k]["final_regret_mean"] for r in recs]) for k in keys}
        best = min(m, key=m.get)
        d = {kk: vv for kk, vv in cfgp[best].items() if kk != "algo"}
        body = ", ".join(f"{kk}: {vv}" for kk, vv in d.items())
        lines.append(f"  {algo}: {{{body}}}   # X3 argmin '{best}' over {len(keys)} configs (mean {m[best]:.2f} on seeds 0-4)")
    with open(os.path.join(ROOT, "src", "configs", f"{DST}_{e}.yaml"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))
