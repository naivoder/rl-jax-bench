"""Table III, the claims and the health counters for CFG-P2 (tag p2, 50 runs), printed as markdown.
Also the authors'-code z-scores (same definition as authors_zscores.py) and the changes against the archived CFG-P (v1, tag e1).
Usage: python scripts/followup/cfg_p2_table.py  -> runs/report/p2_report.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from src import eval as E  # noqa: E402
from src import model as M  # noqa: E402
from rl_jax_bench.stats import mean_se  # noqa: E402

ROOT = os.environ.get("GOSPA_RUNS", "runs")
LAMS = [0.1, 1.0, 2.0]
rep = json.load(open(os.path.join(HERE, "..", "..", "runs", "report", "e1_report.json")))
off = {(r["alg"], r["clutter"]): r["official"] for r in rep["CFG-O"]["table3"] if "official" in r}
rng = np.random.default_rng(0)
algs = [a for a in M.ALG_ORDER if os.path.exists(os.path.join(ROOT, "p2", "seed_49", f"metrics__CFG-P2__{a}.json"))]
print("algorithms available:", algs)
T = E.build_table(ROOT, "p2", "CFG-P2", algs, LAMS, 50)
V1 = {a: E.load_cell(ROOT, "e1", "CFG-P", a, 50) for a in algs}
out = dict(table=[], grades={})
print("| Metric (Table III statistic G) | Paper | Ours CFG-P2: G [95 % run-bootstrap CI]; per-run mean GOSPA mean +- SE (n = 50) | Delta % | Judgment | CFG-P v1 (superseded) G | authors' code (n), z |")
print("|---|---|---|---|---|---|---|")
gr = {"MATCH": 0, "CLOSE": 0, "DIVERGENT": 0}
rels = []
for a in algs:
    for lam in LAMS:
        t = T[a][lam]
        pap = M.PAPER_TABLE3.get(a)
        v1 = E.table3_stat(V1[a][str(lam)]["gospa_sq"])[0]
        gb = mean_se(t["gbar"])
        if pap is None:
            print("| %s, lambda_C = %s | -- | %.2f [%.2f, %.2f]; %.2f +- %.2f | -- | -- | %.2f | -- |" % (a, lam, t["G"], t["ci"][0], t["ci"][1], gb[0], gb[1], v1))
            continue
        pp = pap[M.CLUTTER_RATES.index(lam)]
        rel, g = E.grade(t["G"], pp)
        gr[g] += 1; rels.append(rel)
        zs = "--"
        if (a, lam) in off:
            o = off[(a, lam)]
            gsq = t["cell"]["gospa_sq"]
            Gs = np.array([np.mean(np.sqrt(gsq[rng.integers(0, 50, o["n"])].mean(axis=0))) for _ in range(4000)])
            sd = Gs.std(ddof=1)
            zs = "%.2f (n=%d), z %s" % (o["G"], o["n"], "n/a" if sd < 1e-9 else "%+.2f" % ((o["G"] - t["G"]) / sd))
        print("| %s, lambda_C = %s | %.2f | %.2f [%.2f, %.2f]; %.2f +- %.2f | %+.1f | %s | %.2f | %s |" % (a, lam, pp, t["G"], t["ci"][0], t["ci"][1], gb[0], gb[1], 100 * (t["G"] - pp) / pp, g, v1, zs))
        out["table"].append(dict(alg=a, clutter=lam, G=t["G"], ci=t["ci"], gbar_mean=gb[0], gbar_se=gb[1], paper=pp, rel=rel, grade=g, G_v1=v1, invalid=t["invalid"]))
print("\nSummary CFG-P2: %d MATCH / %d CLOSE / %d DIVERGENT; median |rel| %.1f %%, max %.1f %%" % (gr["MATCH"], gr["CLOSE"], gr["DIVERGENT"], 100 * np.median(rels), 100 * np.max(rels)))
out["summary"] = dict(gr, median_rel=float(np.median(rels)), max_rel=float(np.max(rels)))
need = {"Myopic-GD", "MCTS1-GD", "MCTS2-GD", "MCTS3-GD", "MCTS4-GD", "MCTS1-KLD", "MCTS2-KLD", "MCTS3-KLD", "MCTS4-KLD"}
if need <= set(algs):
    fam = E.families(T, LAMS)
    v = E.verdicts(fam, T, LAMS)
    out["verdicts"] = v
    out["families"] = {k: {kk: vv for kk, vv in d.items() if kk != "diff_per_seed"} for k, d in fam.items()}
    print(json.dumps(v, indent=1, default=float))
    print("\n| test (first arm lower) | lambda_C | mean paired diff of gbar | bootstrap 95 % CI | wins of first arm | p (paired t) | p Holm (33) | significant |\n|---|---|---|---|---|---|---|---|")
    for k, d in fam.items():
        h, rest, lam = k.split("|")
        print("| %s %s | %s | %+.2f | [%+.2f, %+.2f] | %s | %.2g | %.2g | %s |" % (h, rest, lam, d["diff_mean"], d["boot95"][0], d["boot95"][1], 50 - d.get("wins", 0) if d.get("wins") is not None else "-", d["t_p"], d["p_holm"], "yes" if d["sig"] else "no"))
    if "Myopic-KLD" in algs and "MCTS3-GD" in algs:
        out["C5"] = E.c5_check(T, 1.0); print("C5", out["C5"])
    # myopic identity
    for lam in LAMS:
        a, b = T["Myopic-GD"][lam]["cell"]["gospa_sq"], T["Myopic-KLD"][lam]["cell"]["gospa_sq"]
        print("Myopic-GD vs Myopic-KLD at", lam, "dG %.2f" % (T["Myopic-GD"][lam]["G"] - T["Myopic-KLD"][lam]["G"]), "identical runs %.0f %%" % (100 * np.mean(np.all(np.isclose(a, b), axis=1))))
# health
print("\nhealth (invalidating counters, summed over all cells):", {a: {str(l): T[a][l]["invalid"] for l in LAMS if T[a][l]["invalid"]} for a in algs if any(T[a][l]["invalid"] for l in LAMS)})
rg = {a: [int(np.sum(T[a][l]["cell"].get("r_gt_1", [0]))) for l in LAMS] for a in algs}
print("r_gt_1 lane-steps by algorithm (lambda_C 0.1/1/2):", rg)
pn = {a: [int(T[a][l]["cell"]["counters"][:, 5].sum()) for l in LAMS] for a in algs}
print("plan_nan_steps:", pn)
out["r_gt_1"] = rg
json.dump(out, open(os.path.join(HERE, "..", "..", "runs", "report", "p2_report.json"), "w"), indent=1, default=float)
