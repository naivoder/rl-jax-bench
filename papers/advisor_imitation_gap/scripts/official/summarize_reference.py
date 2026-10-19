"""Three-way table for the reference runs: the authors' published
per-run TSVs (paper), the authors' code re-run on our hardware (ref_runs/out/*.json, scripts/official/run_reference.py)
and ours at preliminary scale (runs/prelim_*/report.json). Writes runs/official_ref/summary.json and prints the table.
Runs in OUR venv (numpy only)."""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import OFFICIAL_ROOT  # noqa: E402
from src.eval import OFFICIAL_HP_RUNS, official_tsv  # noqa: E402

REF = os.path.join(OFFICIAL_ROOT, "ref_runs", "out")
LH_TSV = os.path.join(OFFICIAL_HP_RUNS, "lighthouse", "pairwise_imitation_results")
OUT = "runs/official_ref"
os.makedirs(OUT, exist_ok=True)


def ms(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                se=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None,
                median=float(np.median(x)) if len(x) else None, values=[float(v) for v in x])


refs = [json.load(open(f)) | {"name": os.path.basename(f)[:-5]} for f in sorted(glob.glob(f"{REF}/*.json"))]
summary = {"pd": {}, "lh": {}, "mg": {}}

# ---------------- Poisoned Doors: same hp draw (lr, tf, alpha) as the authors' TSV row with seed == draw
tsv = official_tsv("PoisonedDoors")
ours_pd = json.load(open("runs/prelim_pd/report.json"))
for exp, our_arm in (("advisor", "advisor"), ("ppo", "ppo"), ("bc", "bc"), ("bc_then_ppo", "bc_then_ppo")):
    rr = [r for r in refs if r["task"] == "PoisonedDoors" and r["exp"] == exp]
    rows = {int(r["seed"]): r for r in tsv if r["exp_type"] == exp}
    paired = [(r["draw"], r["reward"][-1], rows[r["draw"]]["reward"][-1]) for r in rr]
    band = [r["reward"][-1] for r in tsv if r["exp_type"] == exp and 3e-4 <= float(r["lr"]) <= 3e-3]
    summary["pd"][exp] = dict(
        authors_code_ours_hw=ms([p[1] for p in paired]), tsv_same_draws=ms([p[2] for p in paired]),
        draws=[p[0] for p in paired], tsv_lr_band=ms(band),
        ours_prelim=ours_pd[f"pd_privileged/{our_arm}"]["ret"])

# ---------------- 2D-Lighthouse: authors' 25 runs per cell vs their code re-run vs ours (preliminary)
ours_lh = json.load(open("runs/prelim_lh/report.json"))
for exp, our_arm in (("bc", "bc"), ("advisor_ppo", "advisor"), ("ppo", "ppo")):
    with open(os.path.join(LH_TSV, f"{exp}__2_15.tsv")) as fh:
        tr = list(csv.DictReader(fh, delimiter="\t"))
    for (i, j) in ((1, 1), (1, 15)):
        if exp == "ppo" and j != 1:
            continue
        sel = lambda r: int(float(r["view_radius"])) == i and (exp == "ppo" or int(float(r["expert_view_radius"])) == j)
        tv = [float(r["avg_ep_length"]) for r in tr if sel(r)]
        rr = [r["ep_length"][-1] for r in refs if r["task"] == "LightHouse" and r["exp"] == exp
              and r["hp"]["VIEW_RADIUS"] == i and (exp == "ppo" or r["hp"]["EXPERT_VIEW_RADIUS"] == j)]
        steps = [r["train_steps"][-1] for r in refs if r["task"] == "LightHouse" and r["exp"] == exp
                 and r["hp"]["VIEW_RADIUS"] == i and (exp == "ppo" or r["hp"]["EXPERT_VIEW_RADIUS"] == j)]
        key = f"i{i}_j{j}" if exp != "ppo" else f"i{i}"
        ok = f"i{i}_j{j}/{our_arm}" if exp != "ppo" else f"i{i}_j1/{our_arm}"
        summary["lh"][f"{key}/{exp}"] = dict(
            tsv=ms(tv) | {"iqr": [float(np.percentile(tv, 25)), float(np.percentile(tv, 75))] if tv else None},
            authors_code_ours_hw=ms(rr) | {"train_steps": steps},
            ours_prelim=ours_lh.get(ok, {}).get("len"))

# ---------------- MiniGrid LC Once
ours_mg = json.load(open("runs/prelim_mg/report.json")) if os.path.exists("runs/prelim_mg/report.json") else {}
tsv_mg = official_tsv("AskForHelpLavaCrossingOnce")
for exp in ("advisor", "bc", "ppo"):
    rr = [r for r in refs if r["task"] == "AskForHelpLavaCrossingOnce" and r["exp"] == exp and r.get("total_steps")]
    summary["mg"][f"proxy3e5/{exp}"] = dict(
        authors_code_ours_hw=ms([r["reward"][-1] for r in rr]) | {"success": [r.get("success", [None])[-1] for r in rr],
                                                                  "curves": [r["reward"] for r in rr]},
        ours_prelim=ours_mg.get(f"lc_once/{exp}", {}).get("ret"))
    rr = [r for r in refs if r["task"] == "AskForHelpLavaCrossingOnce" and r["exp"] == exp and not r.get("total_steps")]
    if rr:
        rows = {int(r["seed"]): r for r in tsv_mg if r["exp_type"] == exp}
        summary["mg"][f"full1e6/{exp}"] = dict(
            draws=[r["draw"] for r in rr], authors_code_ours_hw=ms([r["reward"][-1] for r in rr]),
            curves=[r["reward"] for r in rr], tsv_same_draws=ms([rows[r["draw"]]["reward"][-1] for r in rr]),
            tsv_curves=[rows[r["draw"]]["reward"] for r in rr])
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
json.dump(refs, open(f"{OUT}/authors_code_runs.json", "w"), indent=1)


def f(d, k="mean"):
    return "-" if not d or d.get(k) is None else f"{d[k]:+.3f}" + (f" +- {d['se']:.3f}" if k == "mean" and d.get("se") else "")


for task, d in summary.items():
    for k, v in d.items():
        print(task, k, {kk: (f(vv) if isinstance(vv, dict) and "mean" in vv else vv) for kk, vv in v.items()
                        if kk not in ("curves", "tsv_curves")})
