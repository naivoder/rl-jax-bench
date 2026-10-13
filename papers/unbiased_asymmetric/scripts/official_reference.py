"""Reduction: official-code reference runs (asym-rlpo@1a065b0, CPU, logs parsed from WANDB json lines)
vs the paper's Fig. 3 read-offs vs ours (small-scale, runs/small/summary.json). Writes runs/official_reference/summary.json.

Per run: final = mean of the last 10 of the 200 avg-100 log points (as ours); value at a step t = mean of the log
points in (t - 5% of t, t]. Authors-vs-ours: Welch t-test (two different implementations: unpaired).
    uv run python scripts/official_reference.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import ttest_ind

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from eval import PAPER_AT_1M_HH3, PAPER_FINAL, official_curves  # noqa: E402

ALGO = {"asym-a2c": "a2c_asym_hs", "asym-a2c-state": "a2c_asym_s", "a2c": "a2c"}
ENV = {"shopping5": "shopping_5", "heavenhell3": "heavenhell_3"}
PAPER_AT_2M_HH3 = dict(a2c_asym_hs=0.98, a2c_asym_s=0.15, a2c=0.03)   # Fig. 3a read at 2M (300-dpi render, +-0.03)


def window(t, y, at):
    t, y = np.asarray(t), np.asarray(y)
    sel = (t > at * 0.95) & (t <= at)
    return float(y[sel].mean()) if sel.any() else float("nan")


def ms(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return None
    return dict(n=len(x), mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None,
                min=float(x.min()), max=float(x.max()), per_run=x.tolist())


def main():
    curves = official_curves("*.log")
    groups = {}
    for name, c in curves.items():
        m = re.match(r"(defaults_)?(shopping5|heavenhell3)_(asym-a2c-state|asym-a2c|a2c)_s(\d+)\.log$", name)
        if not m:
            continue
        variant = "defaults" if m.group(1) else "paper"
        if c["t_last"] < 1.9e6:
            print(f"[WARN] {name} incomplete (t={c['t_last']:.0f}), skipped")
            continue
        key = (variant, ENV[m.group(2)], ALGO[m.group(3)])
        groups.setdefault(key, []).append((int(m.group(4)), c))
    ours = json.load(open(os.path.join(REPO, "runs", "small", "summary.json")))["report"]
    out = {}
    for (variant, env, meth), runs in sorted(groups.items()):
        runs.sort(key=lambda x: x[0])
        row = {"seeds": [s for s, _ in runs]}
        row["final_2M"] = ms([c["last10"] for _, c in runs])
        for at in (0.25e6, 0.5e6, 0.75e6, 1.0e6, 1.5e6, 2.0e6):
            row[f"at_{at / 1e6:g}M"] = ms([window(c["t"], c["avg100"], at) for _, c in runs])
        if env == "shopping_5":
            row["paper_final"] = PAPER_FINAL[env][meth]
            o = ours["shopping_5"].get(meth)
            row["paper_1M"] = None
        else:
            row["paper_1M"], row["paper_2M"] = PAPER_AT_1M_HH3[meth], PAPER_AT_2M_HH3[meth]
            o = ours["heavenhell_3_1M"].get(meth)
        if o is not None:
            ours_fin = o["final_per_run"]
            row["ours_small_final"] = ms(ours_fin)
            off = row["final_2M"]["per_run"] if env == "shopping_5" else row["at_1M"]["per_run"]
            if len(off) >= 2 and variant == "paper":
                r = ttest_ind(off, ours_fin, equal_var=False)
                row["authors_minus_ours"] = dict(diff=float(np.mean(off) - np.mean(ours_fin)), welch_t=float(r.statistic),
                                                 p=float(r.pvalue), compared="final_2M" if env == "shopping_5" else "at_1M")
        out[f"{variant}/{env}/{meth}"] = row
        print(f"[METRIC] {variant}/{env}/{meth} seeds={row['seeds']} final2M={row['final_2M']['mean']:.3f}"
              f"±{row['final_2M']['se'] or 0:.3f} at1M={row['at_1M']['mean']:.3f}"
              + (f" ours={row['ours_small_final']['mean']:.3f}" if "ours_small_final" in row else "")
              + (f" p={row['authors_minus_ours']['p']:.3f}" if "authors_minus_ours" in row else ""))
    os.makedirs(os.path.join(REPO, "runs", "official_reference"), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, "runs", "official_reference", "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
