"""Collect every per-arm, per-seed result into the required runs/seed_<n>/metrics.json (one file per seed index)."""
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
for exp_dir in sorted(d for e in ("toys", "small", "full", "full_diag", "full_extra") for d in glob.glob(os.path.join(REPO, "runs", e))):
    exp = os.path.basename(exp_dir)
    for arm_dir in sorted(glob.glob(os.path.join(exp_dir, "*_*"))):
        if not os.path.isdir(arm_dir):
            continue
        arm = os.path.basename(arm_dir)
        for sd in sorted(glob.glob(os.path.join(arm_dir, "seed_*"))):
            n = int(sd.rsplit("_", 1)[1])
            rec = {}
            for f in sorted(glob.glob(os.path.join(sd, "*.json"))):
                rec[os.path.basename(f)[:-5]] = json.load(open(f))
            if exp in ("full", "full_diag") and ("-nobaseline" in arm or "-zerogae" in arm) and "metrics" in rec:
                # recheck: these runs fit V to V + G (or V + A with V == 0), so their value metrics are invalid
                rec["metrics"]["value_metrics_valid"] = False
                rec["metrics"]["value_metrics_note"] = ("value target was V_old + policy advantage (G_t), not V_old + "
                                                        "A^GAE (fixed in 2b0e441); policy/reward unaffected (V unused)")
            out.setdefault(n, {}).setdefault(exp, {})[arm] = rec
for n, d in out.items():
    p = os.path.join(REPO, "runs", f"seed_{n}", "metrics.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({"seed": n, "root_key": f"PRNGKey({n})", "experiments": d}, open(p, "w"))
    print(p, os.path.getsize(p) // 1024, "KiB")
