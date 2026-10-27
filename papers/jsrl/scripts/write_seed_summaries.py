"""Consolidate per-seed headline scores into runs/seed_<n>/metrics.json (required layout) from every runs/<stage>/<Game>/seed_<n>/metrics.json.
Each entry keeps the graded scalars (R0, EARLY, FINAL, greedy FINAL, stage reached, guide share) and the path of the full file (101 evaluation points per arm)."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.eval import scores

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
for p in sorted(glob.glob(os.path.join(REPO, "runs", "*", "*", "seed_*", "metrics.json"))):
    parts = p.split(os.sep)
    stage, game, seed = parts[-4], parts[-3], int(parts[-2].split("_")[1])
    if stage in ("dev",):
        continue
    cell = json.load(open(p)).get(game)
    if cell is None:
        continue
    n = cell["meta"]["n_eval_points"]
    d = out.setdefault(seed, {}).setdefault(stage, {}).setdefault(game, {})
    d["steps"] = cell["meta"]["steps"]; d["guide_true_mean"] = cell["meta"]["guide"]["true_mean"]; d["full_file"] = os.path.relpath(p, REPO)
    for arm, r in cell.items():
        if arm == "meta":
            continue
        sc = scores(r, n)
        d[arm] = dict(R0=sc["r0"], EARLY=sc["early"], FINAL=sc["final"], greedy_FINAL=sc["greedy_final"])
        if "stage_reached" in r:
            d[arm].update(stage_reached=r["stage_reached"], first_eval_at_h0=r["first_eval_at_h0"])
        if "guide_steps" in r:
            d[arm]["guide_share"] = r["guide_steps"] / cell["meta"]["steps"]
for seed, v in out.items():
    os.makedirs(os.path.join(REPO, "runs", f"seed_{seed}"), exist_ok=True)
    json.dump(v, open(os.path.join(REPO, "runs", f"seed_{seed}", "metrics.json"), "w"), indent=1)
print("wrote", len(out), "seed summaries")
