"""Golden check: the primary scene re-run AFTER the fix-stage code changes (ssv_norm, truth_vel_sigma, liveness, BC
scope, selection flag) vs the review run (runs/review/scene_primary.json, itself golden vs full-scale). Every per-episode
value of every fixed policy and seed must agree."""
import json
import numpy as np
R = "runs"
a = json.load(open(f"{R}/review/scene_primary.json"))["arms"]
b = json.load(open(f"{R}/recheck/fixstage/scenes/scene_primary.json"))["arms"]
worst, n = 0.0, 0
for arm in a:
    for s in a[arm]:
        for k, v in a[arm][s]["per_episode"].items():
            x, y = np.asarray(v, float), np.asarray(b[arm][s]["per_episode"][k], float)
            if not np.array_equal(x, y, equal_nan=True):          # bitwise (inf and nan positions included)
                m = np.isfinite(x) & np.isfinite(y)
                rel = float(np.max(np.abs(x[m] - y[m]) / np.maximum(np.abs(x[m]), 1e-300))) if m.any() else float("inf")
                worst = max(worst, rel if np.array_equal(np.isfinite(x), np.isfinite(y)) else float("inf"))
            n += 1
print(f"compared {n} per-episode series over {len(a)} arms; max relative difference {worst:.3g}; bitwise={worst == 0.0}")
