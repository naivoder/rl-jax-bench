"""Full-scale DIVERGENT protocol: test the top-ranked cause for each DIVERGENT cell (README Full-scale, <= 15 min each).

    uv run python scripts/divergent.py     -> runs/full/divergent.json

1. SW-UCB on E1 (+27 %): cause 1 = 'the paper's curve is one realisation'. Re-run the headline SW-UCB arm on the
   SAME keys (seeds 0-4 x 2000; bit-identical to runs/full/e1) keeping per-replication curves, and measure how
   often ONE run lands within the read-off precision (+-10) of the paper's R(3000) = 65, R(5000) = 82 and R(T) = 267.
2. SW-UCB on E2b (+64 %, ALT slides): cause 1 = a different window on the slide run; rerun SW-UCB with the T = 1e4
   window tau = 1214 and with the Remark-9 window on E2b (same keys).
"""
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
import src  # noqa: E402,F401
from src import train  # noqa: E402
from src.common import check_device  # noqa: E402

check_device()
out = {}
cfg = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", "e1.yaml")))
recs = train.run_experiment_seeds(cfg, [0, 1, 2, 3, 4], 2000, policies=["swucb"], keep_curves=True)
C = np.concatenate([r["_curves"]["swucb"] for r in recs])  # (10000, 1001), grid every 10 rounds
ref = np.concatenate([json.load(open(os.path.join(ROOT, f"runs/full/e1/seed_{s}/metrics.json")))["policies"]["swucb"]["final_regret"]
                      for s in range(5)])
assert np.allclose(C[:, -1], ref, atol=1e-5), "re-run must reproduce the headline SW-UCB arm"
r3, r5, rT = C[:, 300], C[:, 500], C[:, 1000]
w = lambda x, v: np.abs(x - v) <= 10  # noqa: E731
out["swucb_e1"] = {"n": int(len(rT)), "p_RT_within10": float(w(rT, 267).mean()), "p_R3000_within10": float(w(r3, 65).mean()),
                   "p_R5000_within10": float(w(r5, 82).mean()), "p_all3_within10": float((w(r3, 65) & w(r5, 82) & w(rT, 267)).mean()),
                   "p_RT_le_277": float((rT <= 277).mean()), "p_R5000_le_92": float((r5 <= 92).mean()),
                   "mid_segment_increment_mean": float((r5 - r3).mean()), "mid_segment_increment_paper": 17.0,
                   "mid_segment_increment_pct_of_paper": float(((r5 - r3) <= 17).mean() * 100),
                   "last_segment_increment_mean": float((rT - r5).mean()), "last_segment_increment_paper": 185.0,
                   "last_segment_increment_pct_of_paper": float(((rT - r5) <= 185).mean() * 100),
                   "median_RT_given_R5000_le_92": float(np.median(rT[r5 <= 92])) if (r5 <= 92).any() else None}
print(json.dumps(out["swucb_e1"], indent=1))

cfg2 = yaml.safe_load(open(os.path.join(ROOT, "src", "configs", "e2b.yaml")))
import math  # noqa: E402
cfg2["policies"] = {"swucb": {"xi": 0.5, "tau": 1780},  # headline E2b arm, must reproduce runs/full/e2b
                    "swucb_tau1214": {"algo": "swucb", "xi": 0.5, "tau": 1214},   # cause test: the T = 1e4 window
                    "swucb_tau629": {"algo": "swucb", "xi": 0.5, "tau": int(round(2 * math.sqrt(2e4 * math.log(2e4) / 2)))}}  # Remark 9, U = 2
recs = train.run_experiment_seeds(cfg2, [0, 1, 2, 3, 4], 2000)
m = {k: float(np.mean([r["policies"][k]["final_regret_mean"] for r in recs])) for k in cfg2["policies"]}
head = float(np.mean([json.load(open(os.path.join(ROOT, f"runs/full/e2b/seed_{s}/metrics.json")))["policies"]["swucb"]["final_regret_mean"]
                      for s in range(5)]))
out["swucb_e2b"] = {"means": m, "headline_check": head, "paper": 440, "tau": {k: v.get("tau") for k, v in cfg2["policies"].items()}}
print(json.dumps(out["swucb_e2b"], indent=1))
with open(os.path.join(ROOT, "runs", "full", "divergent.json"), "w") as f:
    json.dump(out, f, indent=1)
