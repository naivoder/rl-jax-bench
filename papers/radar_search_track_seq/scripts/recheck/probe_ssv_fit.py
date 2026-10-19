"""Descriptive: under the single-scan-peak normaliser (p_max = phi_0(0), i.e. p_max,T with T = 0), radial sigma0,
before-push ages, which (gamma, N_SV) is closest to the paper's absolute triple? Also the full-FIFO normaliser (ours)
for contrast. Not a fit used anywhere; it only asks whether ONE reading reproduces all three numbers."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_ssv_norm import run
PAPER = dict(static=-25.2, random=-0.132, coverage=-0.0502)
rng = np.random.default_rng(1)
rows = []
for nsv in (60, 80, 100, 120, 150, 200):
    for gamma in (0.018, 0.021, 0.024, 0.027):
        r = dict(nsv=nsv, gamma=gamma)
        for pol in ("static", "random", "coverage"):
            r[pol] = run(pol, gamma, nsv, "radial", "single", False, rng, n_ep=2)[0]
        r["logerr"] = {k: round(math.log(r[k] / PAPER[k]), 3) for k in PAPER}
        r["rms_logerr"] = round(float(np.sqrt(np.mean([v ** 2 for v in r["logerr"].values()]))), 3)
        rows.append(r); print(json.dumps(r), flush=True)
os.makedirs("runs/recheck/equations", exist_ok=True)
json.dump(rows, open("runs/recheck/equations/probe_ssv_fit.json", "w"), indent=1)
best = min(rows, key=lambda r: r["rms_logerr"]); print("BEST", json.dumps(best))
