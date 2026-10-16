"""Reference curves from the AUTHORS' runs, binned with the paper's metric (official plot.py
avg_return_curve == src/eval.curve, equality tested in tests/test_official_parity.py::test_metric_matches_official_plot).
Reads the official seed_*.pkl when the run finished, else parses the --debug log ("Episodic Return: R, Time Step T"),
which carries the same (return, termination step) pairs, so a still-running reference can be read mid-flight.

Usage: python scripts/official/official_curves.py <run_dir> <total_steps_to_bin> [stride=50000]
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import eval as ev  # noqa: E402

PAT = re.compile(r"Episodic Return: ([-\d.e+]+), Time Step (\d+)")


def load(run_dir):
    runs = {}
    for pk in sorted(glob.glob(os.path.join(run_dir, "data_*", "**", "seed_*.pkl"), recursive=True)):
        ret, steps, _ = pickle.load(open(pk, "rb"))
        runs[int(re.search(r"seed_(\d+)", pk).group(1))] = ("pkl", np.array(steps, float), np.array(ret, float))
    for lg in sorted(glob.glob(os.path.join(run_dir, "log_seed*.txt"))):
        s = int(re.search(r"log_seed(\d+)", lg).group(1))
        if s in runs:
            continue
        m = PAT.findall(open(lg).read())
        runs[s] = ("log", np.array([float(t) for _, t in m]), np.array([float(r) for r, _ in m]))
    return runs


def main(run_dir, total, stride=50_000):
    runs = load(run_dir)
    out = dict(run_dir=run_dir, stride=stride, total=total, seeds={})
    bins = []
    for s, (src, t, r) in sorted(runs.items()):
        e = np.stack([t, r, np.zeros_like(t)], 1)
        b = ev.run_bins(e, total, stride)
        out["seeds"][s] = dict(source=src, last_step=float(t.max()) if len(t) else 0.0, n_episodes=int(len(t)),
                               bins=b.tolist())
        if len(t) and t.max() > total - stride:
            bins.append(b)
    if bins:
        B = np.stack(bins)
        out["n_complete"] = len(bins)
        out["mean_bins"] = B.mean(0).tolist()
        out["se_bins"] = (B.std(0, ddof=1) / np.sqrt(len(bins))).tolist() if len(bins) > 1 else None
        out["final_bin_per_seed"] = B[:, -1].tolist()
        out["final_bin_mean"] = float(B[:, -1].mean())
        out["final_bin_se"] = float(B[:, -1].std(ddof=1) / np.sqrt(len(bins))) if len(bins) > 1 else None
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1], int(float(sys.argv[2])), int(sys.argv[3]) if len(sys.argv) > 3 else 50_000)
