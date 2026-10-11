"""Recheck aggregation ( long budget,  official multi schedule,  dispersion) -> runs/recheck/report.json

    python scripts/recheck_report.py
"""
import json
import os
import sys

import numpy as np
from scipy import stats

from rl_jax_bench.stats import mean_se, paired_compare, t_interval  # noqa: E402

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
LONG = {"state": {s: f"{R}/recheck/long" if s < 3 else f"{R}/recheck/long" for s in range(5)},
        "multi": {s: f"{R}/recheck/long" if s < 3 else f"{R}/recheck/long" for s in range(5)},
        "maml": {s: f"{R}/recheck/long" for s in range(3)}}
CHECK = (2500, 4000, 6000, 8000, 10000)
PAPER_X05 = {"multi": -50.2, "maml": -50.4, "state": -69.4}  # Fig. 5 digitized, x in [0.475, 0.525]e7
PAPER_BAND_HW_X05 = {"multi": 9.2, "maml": 8.4, "state": 13.3}
HEUR = json.load(open(f"{R}/recheck/heuristics.json"))


def load(path, arm, seed):
    return json.load(open(f"{path}/{arm}/seed_{seed}/metrics.json"))


def last5(d, it):
    ev = {e["it"]: e for e in d["evals"]}
    return float(np.mean([ev[i]["test_mean"] for i in range(it - 100, it + 1, 25)]))


def label(est, ci, paper):
    if abs(est - paper) <= 0.05 * abs(paper):
        return "MATCH"
    return "CLOSE" if ci[0] <= paper <= ci[1] else "DIVERGENT"


out = {"long": {}, "long_pairs": {}, "heuristic_cmp": {}}
per = {}
for arm, seeds in LONG.items():
    rows, ent, gre, ep_std = [], [], [], []
    for s, path in seeds.items():
        f = f"{path}/{arm}/seed_{s}/metrics.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        if d["final"]["it"] != 10000:
            continue
        rows.append([last5(d, it) for it in CHECK])
        ent.append(d["history"][-1]["entropy"])
        gre.append(float(np.mean([e["greedy_mean"] for e in d["evals"][-5:]])))
        ep_std.append(float(np.mean([e["test_std"] for e in d["evals"][-5:]])))
    if not rows:
        continue
    rows = np.asarray(rows)
    per[arm] = rows
    end = rows[:, -1]
    m, se = mean_se(end)
    ci = t_interval(end)
    out["long"][arm] = {"n": len(end), "seeds": sorted(seeds)[:len(end)], "per_seed_last5_at": dict(zip(map(str, CHECK), rows.T.round(2).tolist())),
                        "mean_at": dict(zip(map(str, CHECK), rows.mean(0).round(2).tolist())),
                        "end_mean": m, "end_se": se, "end_t95": ci, "paper_x05": PAPER_X05[arm],
                        "rel_dev_from_paper": (m - PAPER_X05[arm]) / abs(PAPER_X05[arm]),
                        "judgment_s7": label(m, ci, PAPER_X05[arm]),
                        "end_norm_entropy": float(np.mean(ent)), "greedy_last5": float(np.mean(gre)),
                        "across_episode_std": float(np.mean(ep_std)), "across_seed_std": float(np.std(end, ddof=1)),
                        "paper_band_halfwidth_x05": PAPER_BAND_HW_X05[arm]}
    for h in ("jsq_first", "spt_official", "lw_official", "spt_fastest_tie", "jsq_fastest_tie", "eft"):
        hm = HEUR[f"test_set/strict/{h}"]["mean"]
        dd = end - hm
        out["heuristic_cmp"][f"{arm}_vs_{h}"] = {"heur": hm, "diff_mean": float(dd.mean()),
                                                  "t95": t_interval(dd), "t_p": float(stats.ttest_1samp(dd, 0).pvalue),
                                                  "n_above": int((dd > 0).sum()), "n": len(dd)}
for a in ("multi", "maml"):
    if a in per and "state" in per:
        n = min(len(per[a]), len(per["state"]))
        out["long_pairs"][f"{a}-state"] = {str(it): paired_compare(per[a][:n, j], per["state"][:n, j])
                                           for j, it in enumerate(CHECK)}
if "maml" in per and "multi" in per:
    n = min(len(per["maml"]), len(per["multi"]))
    out["long_pairs"]["maml-multi"] = {"10000": paired_compare(per["maml"][:n, -1], per["multi"][:n, -1])}

#: official multi schedule vs full-scale 'groups' multi, paper config, 2,500 iterations, seeds 0-2
mv = {}
for s in range(3):
    f = f"{R}/recheck/mv_official/multi/seed_{s}/metrics.json"
    if os.path.exists(f):
        d = json.load(open(f))
        g = json.load(open(f"{R}/full/multi/seed_{s}/metrics.json"))
        ev_o = {e["it"]: e["test_mean"] for e in d["evals"]}
        ev_g = {e["it"]: e["test_mean"] for e in g["evals"]}
        mv[s] = {"official_last5": d["final"]["test_mean_last5"], "groups_last5": g["final"]["test_mean_last5"],
                 "official_at": {it: float(np.mean([ev_o[i] for i in range(it - 100, it + 1, 25) if i in ev_o]))
                                 for it in (500, 1000, 1500, 2000, 2500)},
                 "groups_at": {it: float(np.mean([ev_g[i] for i in range(it - 100, it + 1, 25) if i in ev_g]))
                               for it in (500, 1000, 1500, 2000, 2500)},
                 "baseline_mse_last50_official": float(np.mean([h["baseline_mse"] for h in d["history"][-50:]])),
                 "baseline_mse_last50_groups": float(np.mean([h["baseline_mse"] for h in g["history"][-50:]])),
                 "final_it": d["final"]["it"]}
if mv:
    ok = [s for s in mv if mv[s]["final_it"] == 2500]
    out["mv_official"] = {"per_seed": mv}
    if len(ok) >= 2:
        a = [mv[s]["official_last5"] for s in ok]
        b = [mv[s]["groups_last5"] for s in ok]
        out["mv_official"]["paired_last5"] = paired_compare(a, b)
        for it in (500, 1000, 1500, 2000):
            out["mv_official"][f"paired_at_{it}"] = paired_compare([mv[s]["official_at"][it] for s in ok],
                                                                   [mv[s]["groups_at"][it] for s in ok])
off = json.load(open(f"{R}/official/official_paperhp_summary.json"))
out["official_reference_s42"] = {k: {"last5_2100_2500": v.get("last5_2100_2500"),
                                     "at": {str(100 * (i + 1)): c for i, c in enumerate(v["curve_every100"])
                                            if 100 * (i + 1) in (500, 1000, 1500, 2000, 2500)}}
                                 for k, v in off.items() if isinstance(v, dict) and "curve_every100" in v}
json.dump(out, open(f"{R}/recheck/report.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k in ("long",)}, indent=1, default=float)[:4000])
for k, v in out["long_pairs"].items():
    for it, r in v.items():
        print(k, it, round(r["diff_mean"], 2), "+-", round(r["diff_se"], 2), "p", round(r["t_p"], 4), "rel",
              round(100 * r["rel_improvement"], 1), "rel_t95", [round(100 * x, 1) for x in r["rel_t95"]], "wins", r["wins"], "/", r["n"])
if "mv_official" in out:
    print(json.dumps(out["mv_official"], indent=1, default=float)[:3000])
