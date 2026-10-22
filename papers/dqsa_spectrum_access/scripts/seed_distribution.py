"""(recheck): DESCRIPTIVE per-seed read next to the declared across-seed mean (never replacing it).  For each run directory:
per-seed final plateau utilisation / strict Pareto / tolerant pattern, the median over seeds, the number of seeds with util >= 0.75 (the paper's
'near-Pareto' level) and the monitor-based 'converged network' proxy = median of the last three MONITOR strict-Pareto fractions (README).
Usage: seed_distribution.py <run_dir> [<run_dir> ...]  -> prints a markdown table and writes <run_dir>/seed_distribution.json"""
import glob, json, os, sys
import numpy as np

def read(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "seed_*", "metrics.json"))):
        m = json.load(open(f)); fin = m["final"]
        mon = [x for x in m.get("monitor", []) if x.get("pareto_strict") is not None]
        last3 = [x["pareto_strict"] for x in mon[-3:]]
        lastu = [x["test_util"] for x in mon[-3:]]
        rows.append(dict(seed=m["seed"], util=fin["plateau_util"], pareto=fin["pareto_strict"], pat7=fin.get("pattern_E7"), pat7_tol=fin.get("pattern_E7_tol"),
                         mon_pareto_med3=float(np.median(last3)) if last3 else None, mon_util_med3=float(np.median(lastu)) if lastu else None))
    return rows

if __name__ == "__main__":
    for d in sys.argv[1:]:
        rows = read(d)
        if not rows:
            print(d, "no seeds"); continue
        u = np.array([r["util"] for r in rows])
        out = dict(run=d, n=len(rows), mean=float(u.mean()), se=float(u.std(ddof=1) / np.sqrt(len(u))) if len(u) > 1 else None, median=float(np.median(u)),
                   n_ge_075=int((u >= 0.75).sum()), n_monitor_pareto_ge_05=int(sum(1 for r in rows if (r["mon_pareto_med3"] or 0) >= 0.5)), seeds=rows)
        json.dump(out, open(os.path.join(d, "seed_distribution.json"), "w"), indent=1)
        print(f"{os.path.basename(d)}: n={out['n']} mean={out['mean']:.3f} median={out['median']:.3f} util>=0.75: {out['n_ge_075']}/{out['n']}  monitor-Pareto>=0.5 (last-3 median): {out['n_monitor_pareto_ge_05']}/{out['n']}")
        for r in rows:
            print("seed %d util %.3f pareto %.3f pat7 %s pat7_tol %s mon_par_med3 %s" % (r["seed"], r["util"], r["pareto"], r["pat7"], r["pat7_tol"], r["mon_pareto_med3"]))
