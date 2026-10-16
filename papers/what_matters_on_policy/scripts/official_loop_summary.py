"""Summarise the official-module reference loop runs (runs/official_loop/hopper_<cfg>_s<n>.json): per-config score
(mean of the 10 eval points 100k..1M) mean +- SE over seeds, final, non-finite count; NaN scores counted as collapses."""
import glob
import json
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = {}
for f in sorted(glob.glob(os.path.join(REPO, "runs", "official_loop", "hopper_*_s*.json"))):
    m = re.search(r"hopper_(.+)_s(\d+)\.json", f)
    d = json.load(open(f))
    rows.setdefault(m.group(1), []).append((int(m.group(2)), d))
for cfg, lst in rows.items():
    sc = np.array([d["score"] for _, d in lst], float)
    fin = np.array([d["curve"][-1][1] for _, d in lst], float)
    ok = np.isfinite(sc)
    nf = [sum(x["nonfinite"] for x in d["diag"]) for _, d in lst]
    se = sc[ok].std(ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else float("nan")
    print(f"{cfg:10s} seeds={[s for s, _ in lst]} scores={np.round(sc).tolist()} mean={np.nanmean(sc) if ok.any() else float('nan'):.0f}"
          f" se={se:.0f} finals={np.round(fin).tolist()} nonfinite_minibatches={nf} collapsed={int((~ok).sum())}/{len(sc)}"
          f" n_points={[len(d['curve']) for _, d in lst]}")
