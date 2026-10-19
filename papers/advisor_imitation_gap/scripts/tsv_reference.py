"""Reference-only rows from the authors' per-run TSVs for methods we did not run and the MiniGrid
per-run distribution restricted to our hyperparameters. Writes
runs/report/tsv_reference.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import official_tsv, robust_reward_at_k  # noqa: E402

out = {"pd": {}, "mg": {}}
pd = official_tsv("PoisonedDoors")
methods = sorted({r["exp_type"] for r in pd})
for m in methods:
    rows = {int(r["seed"]): float(r["reward"][-1]) for r in pd if r["exp_type"] == m}
    v16 = np.array([rows[d] for d in range(16) if d in rows])
    v50 = np.array(list(rows.values()))
    out["pd"][m] = dict(n_all=len(v50), rr10_all=robust_reward_at_k(v50, 10), mean_all=float(v50.mean()),
                        n_draws0_15=len(v16), rr10_draws0_15=robust_reward_at_k(v16, 10) if len(v16) >= 10 else None,
                        mean_draws0_15=float(v16.mean()))
    print(f"PD  {m:34s} RR@10 all {out['pd'][m]['rr10_all']:+.3f} (n {len(v50)}) | draws 0-15 RR@10 "
          f"{out['pd'][m]['rr10_draws0_15'] if out['pd'][m]['rr10_draws0_15'] is None else round(out['pd'][m]['rr10_draws0_15'], 3)} mean {v16.mean():+.3f}")
mg = official_tsv("AskForHelpLavaCrossingOnce")
for m in sorted({r["exp_type"] for r in mg}):
    R = [r for r in mg if r["exp_type"] == m]
    lr = np.array([float(r["config_kwargs_str"]["LR"]) for r in R])
    al = np.array([float(r["config_kwargs_str"].get("FIXED_ALPHA", np.nan)) for r in R])
    v = np.array([float(r["reward"][-1]) for r in R])
    band = (lr >= 3e-4) & (lr <= 3e-3)
    a8 = band & (al == 8)
    q = lambda x: [float(np.percentile(x, 25)), float(np.median(x)), float(np.percentile(x, 75))] if len(x) else None
    out["mg"][m] = dict(n_all=len(v), rr10_all=robust_reward_at_k(v, 10), max_all=float(v.max()),
                        band_n=int(band.sum()), band_mean=float(v[band].mean()) if band.any() else None, band_q=q(v[band]),
                        band_alpha8_n=int(a8.sum()), band_alpha8_vals=v[a8].round(3).tolist(),
                        band_alpha8_lrs=lr[a8].round(5).tolist())
    print(f"MG  {m:34s} RR@10 all {out['mg'][m]['rr10_all']:+.3f} max {v.max():.3f} | lr band n {band.sum()} mean "
          f"{out['mg'][m]['band_mean']} q {np.round(out['mg'][m]['band_q'], 3).tolist() if band.any() else None}"
          + (f" | band & alpha 8: {v[a8].round(3).tolist()} at lr {lr[a8].round(5).tolist()}" if a8.any() else ""))
os.makedirs("runs/report", exist_ok=True)
json.dump(out, open("runs/report/tsv_reference.json", "w"), indent=1)
