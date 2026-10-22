"""Amendments 3-4: pooled per-seed 'leaves the plateau' read for E0 (final-iterate plateau util >= 0.6 AND strict Pareto >= 0.5), one-sided Fisher exact
(multi-epoch arm vs single-step control), primary = n = 6 (seed groups 100-102 + 103-105), secondary = n = 9 (+ 106-108)."""
import glob, json, os
from scipy.stats import fisher_exact
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs") + os.sep
GROUPS = {
    "control": ["small/E0_competitive_lr0.0003", "recheck/rerun/E0_E0F_single_step_fresh_seeds", "recheck/rerun/E0_E0G_single_step_fresh_seeds2"],
    "A1 (5 epochs, carried Adam)": ["recheck/rerun/E0_A1_ep5_carried", "recheck/rerun/E0_A1F_ep5_carried_fresh_seeds", "recheck/rerun/E0_A1G_ep5_carried_fresh_seeds2"],
    "A2 (5 epochs, fresh Adam)": ["recheck/rerun/E0_A2_ep5_fresh", "recheck/rerun/E0_A2F_ep5_fresh_fresh_seeds"],
}
def seeds(d):
    out = []
    for f in sorted(glob.glob(R + d + "/seed_*/metrics.json")):
        m = json.load(open(f)); fin = m["final"]
        out.append((m["seed"], fin["plateau_util"], fin["pareto_strict"], fin["plateau_util"] >= 0.6 and fin["pareto_strict"] >= 0.5))
    return out
res = {}
for name, dirs in GROUPS.items():
    per = [seeds(d) for d in dirs]
    res[name] = per
    flat = [x for g in per for x in g]
    print(f"{name}: groups n = {[len(g) for g in per]}; leaves plateau per group = {[sum(x[3] for x in g) for g in per]}; per seed util: "
          + ", ".join(f"{x[0]}:{x[1]:.3f}" for x in flat))
def pooled(name, k):
    g = res[name][:k]
    return sum(x[3] for gg in g for x in gg), sum(len(gg) for gg in g)
for label, k in (("PRIMARY n=6", 2), ("SECONDARY n=9", 3)):
    ck, cn = pooled("control", k)
    for name in ("A1 (5 epochs, carried Adam)", "A2 (5 epochs, fresh Adam)"):
        if len(res[name]) < k: continue
        ak, an = pooled(name, k)
        if an < cn and an != cn: pass
        _, p = fisher_exact([[ak, an - ak], [ck, cn - ck]], alternative="greater")
        print(f"{label}: {name} {ak}/{an} vs control {ck}/{cn}: one-sided Fisher p = {p:.4f}")
