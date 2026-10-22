"""Render the Table III markdown tables (text on stdout) from results/e1_report.json: python scripts/tables.py <e1_report.json> CFG-O|CFG-P"""
import json, sys
R = json.load(open(sys.argv[1]))[sys.argv[2]]
print("| Metric (Table III statistic G = time-mean of per-step RMS-GOSPA) | Paper | Ours: G [95 % run-bootstrap CI]; per-run mean GOSPA mean +- SE (n = 50) | Delta % | Judgment | Authors' code (their RNG, n runs) |")
print("|---|---|---|---|---|---|")
for r in R["table3"]:
    if "paper" not in r:
        continue
    off = r.get("official")
    ot = "%.2f (n=%d)%s" % (off["G"], off["n"], "" if off["inside_our_ci"] else " *") if off else "-"
    print("| %s, lambda_C = %g | %.2f | %.2f [%.2f, %.2f]; %.2f +- %.2f | %+.1f | %s | %s |" % (
        r["alg"], r["clutter"], r["paper"], r["G"], r["ci"][0], r["ci"][1], r["gbar_mean"], r["gbar_se"], 100 * (r["G"] - r["paper"]) / r["paper"], r["grade"], ot))
print()
print("(* = the authors' value lies outside our 95 % bootstrap interval)")
