"""full-scale cause test C3/C6: the authors' modules (study-era distribution) on gymnasium
HalfCheetah-v4, REC vs one factor (init std 1.0 = A2b; lr 1e-4 = A5a), 400 k steps (truncated protocol):
score = mean of the 4 evaluation points 100 k..400 k (10 episodes each). Ours at the same truncation, from the full-scale
halfcheetah PROXY lanes (points 1..4 of the curve). Usage: uv run python scripts/official_cause_test.py"""
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))  # authors' checkout, see README
OFF = os.path.join(OFFICIAL_ROOT, "runs", "official_loop")


def off(cfg, seed):
    p = os.path.join(OFF, f"halfcheetah_{cfg}_s{seed}_400k.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    c = [r for s, r in d["curve"][1:]]
    return {"curve": [round(x) for x in c], "score": float(np.mean(c[:4])) if len(c) >= 4 else None}


lanes = []
for f in sorted(glob.glob(os.path.join(REPO, "runs", "full", "halfcheetah", "seed_*", "metrics.json"))):
    lanes += json.load(open(f))["lanes"]
ours = {}
for l in lanes:
    ours.setdefault(l["arm"], {})[l["seed"]] = float(np.mean(l["curve"][1:5]))
out = {"official": {}, "ours_trunc400k": {}}
for arm, cfg in [("REC", "rec_pre"), ("A2b_initstd_1", "rec_pre_std1"), ("A5a_lr_1em4", "rec_pre_lr1em4")]:
    out["official"][arm] = {s: off(cfg, s) for s in (0, 1)}
    out["ours_trunc400k"][arm] = ours[arm]
for arm in ["A2b_initstd_1", "A5a_lr_1em4"]:
    rs = [(out["official"][arm][s], out["official"]["REC"][s]) for s in (0, 1)]
    rs = [(a["score"], r["score"]) for a, r in rs if a and r and a["score"] and r["score"]]
    o = np.array([ours[arm][s] / ours["REC"][s] for s in sorted(ours["REC"])])
    out[arm] = {"official_per_seed_ratio": [a / r for a, r in rs],
                "official_ratio_of_means": (np.mean([a for a, _ in rs]) / np.mean([r for _, r in rs])) if rs else None,
                "ours_trunc400k_ratio_of_means": float(np.mean([ours[arm][s] for s in ours["REC"]]) / np.mean(list(ours["REC"].values()))),
                "ours_trunc400k_per_seed_ratio": o.round(3).tolist()}
json.dump(out, open(os.path.join(REPO, "runs", "full", "official_cause_test.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
