"""Write runs/seed_<n>/metrics.json: for each seed, the final metrics of every (tag, cell, arm, lr) run with that seed
(the per-run files under runs/<tag>/... hold curves and all checkpoints)."""
import glob
import json
import os
from collections import defaultdict

idx = defaultdict(dict)
for f in sorted(glob.glob("runs/*/**/seed_*/metrics.json", recursive=True)):
    if f.startswith("runs/seed_"):
        continue
    m = json.load(open(f))
    seed = m["seed"]
    key = os.path.relpath(os.path.dirname(os.path.dirname(f)), "runs")
    idx[seed][key] = {"final": m["final"], "file": f, **({"train_steps": m["train_steps"]} if "train_steps" in m else {})}
for seed, d in idx.items():
    os.makedirs(f"runs/seed_{seed}", exist_ok=True)
    json.dump({"seed": seed, "runs": d}, open(f"runs/seed_{seed}/metrics.json", "w"), indent=1)
print({s: len(d) for s, d in sorted(idx.items())})
