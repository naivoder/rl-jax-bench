"""Summarise a run_grid output: lr chosen per (cell, arm) on tuning seeds (>= 100) by final eval return
(2D-LH: lowest final eval length), then the evaluation seeds (< 100) at that lr: mean +- SE per metric.
Writes runs/<tag>/report.json and prints a table."""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import mean_se  # noqa: E402


def main(tag, key="ret", lower_better=False, fixed_lr=None):
    S = json.load(open(os.path.join("runs", tag, "summary.json")))["runs"]
    by = defaultdict(list)
    for r in S:
        by[(r["cell"], r["arm"], r["lr"], r["alpha"])].append(r)
    out = {}
    cells = sorted({r["cell"] for r in S})
    arms = []
    for r in S:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    for c in cells:
        for a in arms:
            cands = [k for k in by if k[0] == c and k[1] == a]
            if not cands:
                continue
            def tune_score(k):
                v = [r["final"][key] for r in by[k] if r["seed"] >= 100]
                return np.mean(v) if v else np.nan
            if fixed_lr is not None:
                best = [k for k in cands if abs(k[2] - fixed_lr) < 1e-12][0]
            else:
                scores = {k: tune_score(k) for k in cands}
                best = (min if lower_better else max)(scores, key=lambda k: scores[k])
            ev = sorted([r for r in by[best] if r["seed"] < 100], key=lambda r: r["seed"])
            row = {"lr": best[2], "alpha": best[3], "seeds": [r["seed"] for r in ev],
                   "tune_scores": {f"{k[2]:g}": float(tune_score(k)) for k in cands} if fixed_lr is None else None}
            for m in ev[0]["final"]:
                vals = [r["final"][m] for r in ev]
                mu, se = mean_se(vals)
                row[m] = {"mean": mu, "se": se, "per_seed": [round(v, 4) for v in vals]}
            row["train_steps"] = [r["train_steps"] for r in ev]
            out[f"{c}/{a}"] = row
            print(f"{c:10s} {a:20s} lr={best[2]:<8g} " + " ".join(
                f"{m}={row[m]['mean']:+.3f}+-{row[m]['se']:.3f}" for m in ev[0]["final"] if m in (key, "len", "chose_locked", "opened_lock", "chose_good", "success", "near_opt_1.1")))
    json.dump(out, open(os.path.join("runs", tag, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    tag = sys.argv[1]
    key = sys.argv[2] if len(sys.argv) > 2 else "ret"
    lower = len(sys.argv) > 3 and sys.argv[3] == "lower"
    main(tag, key, lower)
