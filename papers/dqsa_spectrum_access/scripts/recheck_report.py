"""Recheck before/after tables from runs/recheck/rerun/<tag>/seed_*/metrics.json against the same-seed baselines.
Per arm: per-seed final plateau utilisation / strict Pareto / monitor curve, mean +- SE (ddof=1), n, delta vs baseline, paired-by-seed t95 interval
(nominal pairing, README: identical worlds, different action-key consumption) and the declared per-seed REACHES test
(median of the last three MONITOR strict-Pareto fractions >= 0.5, README; read as k of S, descriptive for S < 10)."""
import glob, json, os, sys
import numpy as np
from rl_jax_bench import stats
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs") + os.sep
ARMS = {  # tag -> (baseline dir, label)
    "E0_A1_ep5_carried": ("small/E0_competitive_lr0.0003", "E0 A1 epochs=5 carried Adam"),
    "E0_A2_ep5_fresh": ("small/E0_competitive_lr0.0003", "E0 A2 epochs=5 fresh Adam"),
    "E0_A3_matlab30": ("small/E0_competitive_lr0.0003", "E0 A3 Matlab-default reading (30 ep, fresh, L2 1e-4, lr 1e-3)"),
    "E6_BASE_shaped": (None, "E6 control: v3 per-slot reward at HEAD"),
    "E6_B1_scale_1overN": ("self:E6_BASE_shaped", "E6 B1 reward / N"),
    "E6_C1_episodic": ("self:E6_BASE_shaped", "E6 C1 episodic literal scale"),
    "E6_C2_episodic_scale0.02": ("self:E6_BASE_shaped", "E6 C2 episodic scale 1/T"),
}
def read(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "seed_*", "metrics.json"))):
        m = json.load(open(f)); fin = m["final"]
        mon = [x for x in m.get("monitor", []) if x.get("pareto_strict") is not None]
        l3 = [x["pareto_strict"] for x in mon[-3:]]
        out.append(dict(seed=m["seed"], util=fin["plateau_util"], pareto=fin["pareto_strict"], coll=fin["plateau_coll"], idle=fin["plateau_idle"],
                        mon_util=[round(x["test_util"], 3) for x in mon], reach=(float(np.median(l3)) >= 0.5) if l3 else None,
                        pat7_tol=fin.get("pattern_E7_tol"), pat6_tol=fin.get("pattern_E6_tol")))
    return out
rows = {}
for tag, (base, label) in ARMS.items():
    d = ROOT + "recheck/rerun/" + tag
    r = read(d)
    if not r: continue
    b = read(ROOT + base) if base and not base.startswith("self:") else (read(ROOT + "recheck/rerun/" + base[5:]) if base else [])
    u = [x["util"] for x in r]
    mu, se = stats.mean_se(u)
    rec = dict(tag=tag, label=label, n=len(r), mean=mu, se=se, per_seed=[(x["seed"], round(x["util"], 3), round(x["pareto"], 3)) for x in r],
               reach_k=sum(1 for x in r if x["reach"]), n_util_ge_06=sum(1 for x in r if x["util"] >= 0.6), monitor_util=[x["mon_util"] for x in r])
    if b and len(b) == len(r):
        bu = [x["util"] for x in b]
        rec["baseline_mean"], rec["baseline_se"] = stats.mean_se(bu)
        rec["delta"] = mu - rec["baseline_mean"]
        rec["paired_t95_of_diff"] = stats.t_interval([x - y for x, y in zip(u, bu)])
        rec["baseline_per_seed"] = [(x["seed"], round(x["util"], 3)) for x in b]
    rows[tag] = rec
    print(json.dumps(rec, default=float))
json.dump(rows, open(ROOT + "recheck/rerun/report.json", "w"), indent=1, default=float)
