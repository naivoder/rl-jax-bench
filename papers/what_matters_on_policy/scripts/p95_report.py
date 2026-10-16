"""Markdown summary of runs/p95/p95_subsample.json (EXP-P95(a)): pooled coverage vs exact theory per N, the
3-MC-SE agreement count over all (choice, env, N, convention) cells, and rank stability per choice.
Usage: uv run python scripts/p95_report.py"""
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = json.load(open(os.path.join(REPO, "runs", "p95", "p95_subsample.json")))
print("| N | official CI: empirical (MC SE) | exact (hypergeom.) | exact (binomial) | l/h CI: empirical | exact (hypergeom.) | P(same order) mean | P(same best) mean |")
print("|---|---|---|---|---|---|---|---|")
for N, d in r["pooled"].items():
    o, l = d["official"], d["lh"]
    print(f"| {N} | {o['emp']:.4f} ({o['mcse']:.4f}) | {o['exact_hyper']:.4f} | {o['exact_binom']:.4f} | {l['emp']:.4f} | "
          f"{l['exact_hyper']:.4f} | {d['p_same_order_mean']:.3f} | {d['p_same_best_mean']:.3f} |")
ok = tot = 0
worst = []
for c in r["cells"]:
    for N, s in c["sub"].items():
        for conv in ("official", "lh"):
            z = (s[f"cov_{conv}"] - s[f"cov_{conv}_exact_hyper"]) / max(s[f"cov_{conv}_mcse"], 1e-12)
            tot += 1
            ok += abs(z) <= 3
            worst.append((abs(z), c["choice"], c["env"], N, conv))
worst.sort(reverse=True)
print(f"\ncells within 3 MC SE of the exact hypergeometric coverage: {ok}/{tot}; largest |z|: "
      + "; ".join(f"{w[0]:.2f} ({w[1]}, {w[2]}, N={w[3]}, {w[4]})" for w in worst[:3]))
print("\n| choice | env | k | P(same order) N=32 / 64 / 128 / 256 | P(same best) N=32 / 64 / 128 / 256 |")
print("|---|---|---|---|---|")
for c in r["cells"]:
    so = " / ".join(f"{c['sub'][str(N)]['p_same_order']:.2f}" for N in r["NS"])
    sb = " / ".join(f"{c['sub'][str(N)]['p_same_best']:.2f}" for N in r["NS"])
    print(f"| {c['choice']} | {c['env']} | {len(c['order'])} | {so} | {sb} |")
