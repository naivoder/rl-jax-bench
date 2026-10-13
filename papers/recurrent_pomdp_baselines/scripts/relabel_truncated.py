"""A run shorter than the paper's horizon must not carry the paper's metric name.

Rewrites existing run outputs in place (idempotent): every runs/**/metrics.json whose env has a paper horizon
(Pendulum-V/-P 51k, Semi-Circle 1.548M, Wind 757.5k env steps; author logs) and whose run is shorter gets
``final_last20`` renamed to ``final_last20_of_run`` plus ``final_metric_key``, ``paper_horizon_env_steps`` and
``horizon_fraction``; the values are unchanged.  Full-horizon runs get ``final_metric_key = final_last20``.
New runs write these keys themselves (src/train.py write_outputs).  The per-seed aggregates runs/seed_<n>/metrics.json
get a ``cells_metric_key`` map.
"""
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = {"Pendulum-V": 51000, "Pendulum-P": 51000, "SemiCircle": 1548000, "Wind": 757500}


def main():
    n = {"renamed": 0, "full": 0, "no_paper_horizon": 0}
    for f in sorted(glob.glob(os.path.join(REPO, "runs", "**", "metrics.json"), recursive=True)):
        m = json.load(open(f))
        if "config" not in m:
            continue
        env = m["config"]["env"]
        if env not in PAPER:
            n["no_paper_horizon"] += 1
            continue
        T = PAPER[env]
        run = m["total_env_steps_run"]
        m["paper_horizon_env_steps"] = T
        m["horizon_fraction"] = run / T
        if run < T:
            if "final_last20" in m:
                m["final_last20_of_run"] = m.pop("final_last20")
            m["final_metric_key"] = "final_last20_of_run"
            n["renamed"] += 1
        else:
            m["final_metric_key"] = "final_last20"
            n["full"] += 1
        json.dump(m, open(f, "w"), indent=1)
    for f in sorted(glob.glob(os.path.join(REPO, "runs", "seed_*", "metrics.json"))):
        a = json.load(open(f))
        a["cells_metric_key"] = {"cells": "final_last20_of_run (small stage: Pendulum 20k of 51k, Semi-Circle 150k of 1.548M)",
                                 "full_cells": "final_last20 (51k = paper horizon)"}
        json.dump(a, open(f, "w"), indent=1)
    print(n)


if __name__ == "__main__":
    main()
