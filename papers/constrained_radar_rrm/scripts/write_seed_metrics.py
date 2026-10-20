"""Write runs/seed_<n>/metrics.json from a pilot/eval summary.json (per training seed: every learner arm's per-seed
metrics + the baselines on the same draws), with the environment version stamp ."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import CFG_DIR, REPO, sha256_file  # noqa: E402

stage, summ = sys.argv[1], sys.argv[2]
s = json.load(open(summ))
commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
stamp = dict(stage=stage, git_commit=commit, summary=os.path.relpath(summ, REPO),
             sha256={n: sha256_file(os.path.join(CFG_DIR, n)) for n in sorted(os.listdir(CFG_DIR))
                     if n.endswith((".yaml", ".json"))},
             keys="k_valid only (pilot); k_test untouched")
seeds = sorted({p["seed"] for v in s.values() if "per_seed" in v for p in v["per_seed"]})
for sd in seeds:
    out = dict(stamp, seed=sd, learners={}, baselines={k: v for k, v in s.items() if "replica" in v})
    for k, v in s.items():
        if "per_seed" in v:
            for p in v["per_seed"]:
                if p["seed"] == sd:
                    out["learners"][k] = p
    d = os.path.join(REPO, "runs", f"seed_{sd}")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    print("wrote", d)
