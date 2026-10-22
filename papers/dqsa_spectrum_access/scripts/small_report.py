"""Markdown tables for small-scale: per-config DQSA results (mean +- SE over seeds) vs the non-learning arms on the shared eval worlds.
    python scripts/small_report.py runs/small runs/small_baselines"""
import glob
import json
import os
import sys

import numpy as np

from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402

gdir, bdir = sys.argv[1], sys.argv[2]
KEYS = ["plateau_util", "plateau_coll", "plateau_idle", "pareto_strict", "pareto_tol", "jain_all"]


def fmt(x):
    m, se = mean_se(x)
    return f"{m:.3f} +- {se:.3f}" if se is not None else f"{m:.3f}"


def base(exp, arm):
    p = os.path.join(bdir, exp, f"{arm}.json")
    return json.load(open(p)) if os.path.exists(p) else None


print("| config | n seeds | plateau util | collision | idle | Pareto strict | Pareto tol | Jain | pattern | per-seed util |\n|---|---|---|---|---|---|---|---|---|---|")
for tagdir in sorted(glob.glob(os.path.join(gdir, "*"))):
    ms = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(tagdir, "seed_*", "metrics.json")))]
    if not ms:
        continue
    fs = [m["final"] for m in ms]
    pat = [k for k in fs[0] if k.startswith("pattern") or k.startswith("throughput")]
    patt = "; ".join(f"{k}={fmt([f[k] for f in fs])}" for k in pat) or "-"
    print(f"| {os.path.basename(tagdir)} | {len(ms)} | " + "| ".join(fmt([f[k] for f in fs]) for k in KEYS) + f" | {patt} | " +
          ", ".join(f"{f['plateau_util']:.2f}" for f in fs) + "|")
print()
print("| exp | arm | plateau util | plateau rate | collision | idle | note |\n|---|---|---|---|---|---|---|")
for exp in sorted(os.listdir(bdir)):
    for p in sorted(glob.glob(os.path.join(bdir, exp, "*.json"))):
        b = json.load(open(p))
        r = b["per_rep"]
        print(f"| {exp} | {b['arm']} | {fmt([x['plateau_util'] for x in r])} | {fmt([x['plateau_rate'] for x in r])} | "
              f"{fmt([x['plateau_coll'] for x in r])} | {fmt([x['plateau_idle'] for x in r])} | {'RECONSTRUCTED' if b['reconstructed'] else ''} |")
print()
print("Paired DQSA - Aloha (same eval worlds, action keys paired by seed = replicate id), plateau utilization:")
for tagdir in sorted(glob.glob(os.path.join(gdir, "*"))):
    ms = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(tagdir, "seed_*", "metrics.json")))]
    if len(ms) < 3:
        continue
    exp = ms[0]["exp"]
    b = base(exp, "aloha")
    if not b:
        continue
    by = {x["rep"]: x["plateau_util"] for x in b["per_rep"]}
    seeds = [m["seed"] for m in ms if m["seed"] in by]
    if len(seeds) < 3:
        continue
    a = np.array([m["final"]["plateau_util"] for m in ms if m["seed"] in by])
    c = np.array([by[s] for s in seeds])
    pc = paired_compare(a, c, n_boot=2000)
    print(f"- {os.path.basename(tagdir)}: diff {pc['diff_mean']:+.3f} +- {pc['diff_se']:.3f} (SE), wins {pc['wins']}/{pc['n']}, paired t p={pc['t_p']:.3g}, "
          f"t95 [{pc['t95'][0]:.4f}, {pc['t95'][1]:.4f}]" + (f", (percentile boot95 {[round(x, 4) for x in pc['boot95']]} under-covers at n < 10: not used)" if pc['n'] < 10 else f", boot95 {[round(x, 4) for x in pc['boot95']]}"))
