"""Per-seed table of the diagnostic runs (continuation to 20k, shorter alpha/beta ramp)."""
import glob, json, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
def rows(pat, label):
    for f in sorted(glob.glob(os.path.join(R, pat))):
        d = json.load(open(f)); fi = d["final"]
        pats = {k: round(v, 3) for k, v in fi.items() if k.startswith("pattern") or k.startswith("throughput")}
        print(f"| {label} | {d['seed']} | {d['iterations']} | {fi['plateau_util']:.3f} | {fi['plateau_coll']:.3f} | {fi['plateau_idle']:.3f} | {fi['pareto_strict']:.3f} | {fi['jain_all']:.3f} | {pats or '-'} |")
print("| run | seed | iters | util | coll | idle | Pareto strict | Jain | patterns |\n|---|---|---|---|---|---|---|---|---|")
rows("small/E7_competitive_lr0.0003/seed_*/metrics.json", "E7 ramp 10k, stop 10k")
rows("ext20k/a/E7*/seed_*/metrics.json", "E7 ramp 10k, continued to 20k")
rows("ext20k/b/E7*/seed_*/metrics.json", "E7 ramp 10k, continued to 20k")
rows("diag/E7*ramp3000/seed_*/metrics.json", "E7 ramp 3k, stop 10k")
rows("small/E0_competitive_lr0.0003/seed_*/metrics.json", "E0 ramp 10k, stop 10k")
rows("diag/E0*ramp3000/seed_*/metrics.json", "E0 ramp 3k, stop 10k")
rows("small/E6_sum_rate_lr0.0003/seed_*/metrics.json", "E6 ramp 10k, stop 10k")
rows("diag/E6*ramp3000/seed_*/metrics.json", "E6 ramp 3k, stop 10k")
