"""Aggregate small-scale outputs into runs/small/report.json and print markdown tables (seed = unit; mean +- SE)."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from rl_jax_bench.stats import mean_se  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G2 = os.path.join(REPO, "runs", "small")
PAPER = dict(m_cov=dict(static=2630.72, random=948.15, coverage=574.27, flat=727, bigru=765, mhsa=391),
             gospa_d=dict(flat=1.25e7, bigru=1.20e7, mhsa=1.13e7, random=1.25e7, coverage=1.35e7, static=1.22e7),
             gospa_miss=dict(flat=1.33e11, bigru=1.26e11, mhsa=1.04e11, random=1.31e11, coverage=1.44e11, static=1.49e11))
KEYS = ["sr", "m_cov", "gospa_d", "gospa_loc", "gospa_miss", "gospa_false", "mean_tracks", "n_init", "life_median",
        "central3"]


def main():
    base = json.load(open(os.path.join(G2, "baselines.json")))
    rows = {}
    for arm, r in base["small"].items():
        rows[arm] = {k: (r.get(k), None, 1) for k in KEYS}
    learn = {}
    for f in sorted(glob.glob(os.path.join(G2, "*", "seed_*", "metrics.json"))):
        m = json.load(open(f))
        ev = m.get("eval_stochastic") or m.get("eval_greedy")
        learn.setdefault(m["arm"], []).append((m, ev))
    direction = {}
    for arm, lst in learn.items():
        rows[arm] = {}
        for k in KEYS:
            v = [e.get(k, np.nan) for _, e in lst]
            mu, se = mean_se(v)
            rows[arm][k] = (mu, se, len(v))
        cur = [m["curve"] for m, _ in lst if "curve" in m]
        if cur:
            direction[arm] = {k: [c[-1][k] - c[0][k] for c in cur] for k in ("ret_flip", "m_cov", "gospa_d")}
    rnd = rows["random"]
    ratios = {a: {k: rows[a][k][0] / rnd[k][0] for k in ("sr", "m_cov", "gospa_d", "gospa_miss")} for a in rows}
    sanity = {}
    for k, ref in PAPER.items():
        for a, pv in ref.items():
            if a in rows and rows[a][k][0] == rows[a][k][0]:
                sanity[f"{k}:{a}"] = rows[a][k][0] / pv
    st = base["small"]["static"]
    pg = st["gospa_miss"] / st["gospa_d"]
    out = dict(rows=rows, ratios_to_random=ratios, sanity_ours_over_paper=sanity, direction=direction,
               p_gospa_static_ratio=pg, p_gospa_rel=(pg / 12247.45 - 1), hand=base["hand"],
               dt_conditional=dict(life_median=base["dt_conditional"]["life_median"],
                                   switch=base["dt_conditional"]["switch"]))
    json.dump(out, open(os.path.join(G2, "report.json"), "w"), indent=1, default=float)
    fmt = lambda t: "nan" if t[0] is None or t[0] != t[0] else (f"{t[0]:.4g}" + (f" ± {t[1]:.2g}" if t[1] else ""))
    print("| arm | n | SR | M_cov | Σ GOSPA d | Σ loc | Σ missed | Σ false | tracks | init | life |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for a, r in rows.items():
        print(f"| {a} | {r['sr'][2]} | " + " | ".join(fmt(r[k]) for k in KEYS[:-1]) + " |")
    print("\nratios to Random:", json.dumps({a: {k: round(v, 3) for k, v in d.items()} for a, d in ratios.items()}))
    print("\nsanity ours/paper:", json.dumps({k: round(v, 2) for k, v in sanity.items()}))
    print("\ndirection (end - start of curve, per seed):", json.dumps(direction))
    print("\nP-GOSPA static ratio", pg, "rel", pg / 12247.45 - 1)


if __name__ == "__main__":
    main()
