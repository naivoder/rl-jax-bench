"""Collect per-seed end-points of every experiment/arm of a tag into runs/seed_<n>/metrics.json (the required layout);
raw per-episode data stay in runs/<tag>/<exp>/<arm>/seed_<n>/metrics.json. Usage: python scripts/collect_seed_metrics.py <tag>"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tag = sys.argv[1]
per = {}
for p in glob.glob(os.path.join(REPO, "runs", tag, "*", "*", "seed_*", "metrics.json")):
    r = json.load(open(p))
    s = r["seed"]
    keep = {k: v for k, v in r["summary"].items() if k != "bins"}
    per.setdefault(s, {}).setdefault(r["exp"], {})[r["arm"]] = dict(summary=keep, source=os.path.relpath(p, REPO))
for p in glob.glob(os.path.join(REPO, "runs", "ppo", tag, "*", "*", "seed_*", "metrics.json")):
    r = json.load(open(p))
    per.setdefault(r["seed"], {}).setdefault("e2_ppo_" + r["env_id"], {})[r["arm"]] = dict(
        summary={k: v for k, v in r["summary"].items() if k != "curve_50bins"}, source=os.path.relpath(p, REPO))
# Review pass: Fig.-4 comparisons use the figure's 125k grid; add those per-seed end-points from the report
rp = os.path.join(REPO, "runs", tag, "report.json")
if os.path.exists(rp):
    rep = json.load(open(rp))
    exps = {"e3_Breakout": "full_e3_Breakout", "e3_Breakout_800k": "full_e3_Breakout@800000"}
    for key, exp in exps.items():
        for arm, a in rep.get(key, {}).get("arms", {}).items():
            seeds = sorted(q for q in per if exp in per[q] and arm in per[q][exp])
            grid = a.get("grid") or {str(rep[key].get("compared_at")): a}
            for x, g in grid.items():
                assert len(g["per_seed"]) == len(seeds), (key, arm)
                for q, v in zip(seeds, g["per_seed"]):
                    per[q][exp][arm].setdefault("fig4_125k_endpoint", {})[x] = v
for s, d in per.items():
    out = os.path.join(REPO, "runs", f"seed_{s}", "metrics.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    old[tag] = d
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(old, open(out, "w"), indent=1)
print("[INFO] wrote", len(per), "seed files for tag", tag)
