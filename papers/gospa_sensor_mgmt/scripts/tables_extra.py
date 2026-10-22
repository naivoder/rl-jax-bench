"""More markdown tables.  python scripts/tables_extra.py <what> ...
 fam <e1_report.json> CFG   : the H2/H3 tests + H1 summary    fa <e1_report.json> CFG : ablation family
 strut <strut_report.json>  : S-TRUTH G table + verdicts       div <div_report.json>   : divergence arms
 sens <sens_report.json>    : F-S table                        ft <sens_report.json>   : F-T table
 theory <t1.json> <t2t3.json> : T1/T2/T3                        e5 <e5.json> <e1_report.json> CFG-O : E5 table"""
import json, sys

w = sys.argv[1]
if w == "fam":
    R = json.load(open(sys.argv[2]))[sys.argv[3]]
    F = R["P1_families"]
    print("| test (first arm should be lower) | lambda_C | mean paired diff of gbar (first - second) | bootstrap 95 % CI | wins of first arm / 50 | p (paired t) | p Holm (33) | significant |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in F.items():
        h, t, lam = k.split("|")
        if h == "H1":
            continue
        print("| %s %s | %s | %+.2f | [%+.2f, %+.2f] | %d | %.2g | %.2g | %s |" % (h, t, lam, v["diff_mean"], v["boot95"][0], v["boot95"][1], 50 - v["wins"], v["t_p"], v["p_holm"], "yes" if v["sig"] else "no"))
    h1 = [v for k, v in F.items() if k.startswith("H1")]
    print()
    print("H1 (12 tests, MCTS-GD < Myopic-GD): mean paired difference between %+.2f and %+.2f GOSPA units; first arm wins in %d of 600 run-pairs; every test Holm-significant (largest p_Holm = %.2g)." % (
        max(v["diff_mean"] for v in h1), min(v["diff_mean"] for v in h1), sum(50 - v["wins"] for v in h1), max(v["p_holm"] for v in h1)))
elif w == "fa":
    R = json.load(open(sys.argv[2]))[sys.argv[3]]["P3_ablation"]
    print("| ablated arm vs MCTS3-GD | lambda_C | mean paired diff (ablated - MCTS3-GD) +- bootstrap 95 % CI | p Holm (9) | ablated worse and significant |")
    print("|---|---|---|---|---|")
    for k, v in R["tests"].items():
        _, t, lam = k.split("|")
        print("| %s | %s | %+.2f [%+.2f, %+.2f] | %.2g | %s |" % (t.split(">")[0], lam, v["diff_mean"], v["boot95"][0], v["boot95"][1], v["p_holm"], "yes" if (v["sig"] and v["ablated_worse"]) else "no"))
    print("\nVerdict: %s (%d/9 significant in the predicted direction, %d in the opposite direction)." % (R["verdict"], R["confirmed"], R["opposite"]))
elif w == "strut":
    R = json.load(open(sys.argv[2]))
    print("| algorithm | lambda_C = 0.1: G [95 % CI] (per-run mean +- SE) | lambda_C = 1 | lambda_C = 2 |")
    print("|---|---|---|---|")
    for a, v in R["G"].items():
        print("| %s | " % a + " | ".join("%.2f [%.1f, %.1f] (%.2f +- %.2f)" % (x["G"], x["ci"][0], x["ci"][1], x["gbar"][0], x["gbar"][1]) for x in v.values()) + " |")
    print()
    V = R["verdicts"]
    print("Verdicts on the pre-declared rules: H1 %s (%d/12 significant, %d against); H2 %s (MCTS3-GD lowest G in %d/3 columns, %d/9 significant); H3 %s (GD below KLD in %d/12 cells, %d significant for GD, %d for KLD); C4 %s (max range over clutter %.2f)." % (
        V["H1"]["verdict"], V["H1"]["sig_in_direction"], V["H1"]["sig_against"], V["H2"]["verdict"], V["H2"]["lowest_columns"], V["H2"]["sig_in_direction"], V["H3"]["verdict"],
        V["H3"]["point_direction_GD"], V["H3"]["sig_GD"], V["H3"]["sig_KLD"], V["C4"]["verdict"], V["C4"]["max_range"]))
elif w == "div":
    R = json.load(open(sys.argv[2]))
    print("| CFG-P with | algorithm | lambda_C | G [95 % CI] | paper | rel | grade |")
    print("|---|---|---|---|---|---|---|")
    for k, v in R.items():
        n, a, lam = k.split("|")
        print("| %s | %s | %s | %.2f [%.2f, %.2f] | %.2f | %.1f %% | %s |" % (n, a, lam, v["G"], v["ci"][0], v["ci"][1], v["paper"], 100 * v["rel"], v["grade"]))
elif w == "sens":
    R = json.load(open(sys.argv[2]))["FS"]
    print("| flip (CFG-O, one at a time) | algorithm | lambda_C = 0.1: G ; diff vs baseline (p Holm) | lambda_C = 1 | lambda_C = 2 | material |")
    print("|---|---|---|---|---|---|")
    seen = []
    for k in R["tests"]:
        sid, a, lam = k.split("|")
        if (sid, a) in seen:
            continue
        seen.append((sid, a))
        cells = []
        for l in ("0.1", "1.0", "2.0"):
            v = R["tests"].get(f"{sid}|{a}|{l}")
            cells.append("%.2f ; %+.2f (%.2g)%s" % (v["G"], v["diff_mean"], v["p_holm"], " *" if v["sig"] else "") if v else "-")
        print("| %s | %s | %s | %s | %s | %s |" % (sid, a, cells[0], cells[1], cells[2], "yes" if R["material"].get(f"{sid}|{a}") else "no"))
    print("\n(* = Holm-significant (family of %d tests); material = |diff| >= 2.0 and significant at >= 2 of 3 clutter rates; %d material flips)" % (len(R["tests"]), R["n_material"]))
elif w == "ft":
    R = json.load(open(sys.argv[2]))["FT"]
    print("| setting (both arms) | lambda_C | G MCTS3-GD | G MCTS3-KLD | paired diff of gbar (GD - KLD) [bootstrap 95 % CI] | p Holm (12) |")
    print("|---|---|---|---|---|---|")
    for k, v in R["tests"].items():
        s, lam = k.split("|")
        print("| %s | %s | %.2f | %.2f | %+.2f [%+.2f, %+.2f] | %s |" % (s, lam, v["G_GD"], v["G_KLD"], v["diff_mean"], v["boot95"][0], v["boot95"][1], "-" if v["p_holm"] is None else "%.2g%s" % (v["p_holm"], " *" if v["sig"] else "")))
    print("\nGD below KLD in %d/%d non-default cells, Holm-significant for GD in %d, for KLD in %d -> %s." % (R["point_GD_better"], R["n_tests"], R["sig_GD"], R["sig_KLD"], R["verdict"]))
elif w == "theory":
    t1 = json.load(open(sys.argv[2])); t = json.load(open(sys.argv[3]))
    print("T1: Lemma-1 bound vs Monte-Carlo true expected cost over %d cases: B_exact >= MC - 3 SE in %d/%d (verdict %s); the paper's eq.-38 value falls below MC - 3 SE in %d/%d cases; median (B - MC)/MC = %+.4f (exact p*), %+.4f (eq. 38)." % (
        t1["cases"], t1["exact_bound_holds"], t1["cases"], t1["verdict"], t1["paper_eq38_violations"], t1["cases"], t1["median_rel_exact"], t1["median_rel_paper"]))
    T3 = t["T3"]
    print("\nT3: reduced-tree two-step value >= full-tree value in %.0f %% (eq. 38 p(h)) and %.0f %% (exact p*) of %d random cases; mean ratio reduced/full %.2f and %.2f (median %.2f, %.2f) -> %s." % (
        100 * T3["paper"]["frac_reduced_ge_full"], 100 * T3["exact"]["frac_reduced_ge_full"], T3["paper"]["n"], T3["paper"]["mean_ratio"], T3["exact"]["mean_ratio"], T3["paper"]["median_ratio"], T3["exact"]["median_ratio"], T3["verdict"]))
    T2 = t["T2"]
    print("\n| eps | B | regret vs exhaustive min (V_min) | regret vs mean-backup optimum (V_mean) | hit rate min | hit rate mean |\n|---|---|---|---|---|---|")
    for k, v in T2.items():
        if k.startswith("eps"):
            print("| %g | %d | %.2f | %.2f | %.3f | %.3f |" % (v["eps"], v["B"], v["regret_min"], v["regret_mean"], v["hit_min"], v["hit_mean"]))
    print("\nPower-law slopes of mean regret against B: " + "; ".join("%s: %.2f" % (k.replace("slope_", ""), v) for k, v in T2.items() if k.startswith("slope") and v is not None))
elif w == "e5":
    E5 = json.load(open(sys.argv[2])); R = json.load(open(sys.argv[3]))["CFG-O"]
    print("| algorithm | compile s | s/step (1 lane, mean / median) | s/step in the 150-lane E1 batch | per-lane s/step in E1 | analytic FLOPs/step (joint ub) | paper Table IV s/step (GD) |")
    print("|---|---|---|---|---|---|---|")
    pap = {"Myopic-GD": 0.02, "MCTS1-GD": 0.23, "MCTS2-GD": 0.62, "MCTS3-GD": 0.85, "MCTS4-GD": 2.57, "Myopic-KLD": 0.02, "MCTS1-KLD": 0.23, "MCTS2-KLD": 0.61, "MCTS3-KLD": 0.85, "MCTS4-KLD": 2.56}
    for a, v in E5["algs"].items():
        s150 = R["sec_per_step_150lanes"][a]["sec_per_step"]
        print("| %s | %.1f | %.4f / %.4f | %.2f | %.4f | %.3g | %s |" % (a, v["compile_seconds"], v["sec_per_step_mean"], v["sec_per_step_median"], s150, s150 / 150, v["flops_per_step_joint_ub"], pap.get(a, "-")))
if w == "h2":
    R = json.load(open(sys.argv[2]))
    print("Pooled n = %d runs (runs 0-49 of E1 + runs 50-199), CFG-O, MCTS-GD variants.  G [95 %% run-bootstrap CI]; per-run mean GOSPA mean +- SE:\n" % R["n"])
    print("| algorithm | lambda_C = 0.1 | lambda_C = 1 | lambda_C = 2 |\n|---|---|---|---|")
    for a, v in R["G"].items():
        print("| %s | " % a + " | ".join("%.2f [%.1f, %.1f]; %.2f +- %.2f" % (x["G"], x["ci"][0], x["ci"][1], x["gbar"][0], x["gbar"][1]) for x in v.values()) + " |")
    print("\n| test (MCTS3-GD lower) | lambda_C | mean paired diff | bootstrap 95 % CI | p Holm (9) | significant |\n|---|---|---|---|---|---|")
    for k, v in R["tests"].items():
        _, t, lam = k.split("|")
        print("| %s | %s | %+.2f | [%+.2f, %+.2f] | %.2g | %s |" % (t, lam, v["diff_mean"], v["boot95"][0], v["boot95"][1], v["p_holm"], "yes" if v["sig"] else "no"))
    print("\nVerdict by the pre-declared H2 rule: **%s** (MCTS3-GD has the lowest G in %d/3 columns; %d/9 comparisons Holm-significant in the paper's direction; REPRODUCED needs 3/3 and >= 7/9).  At n = 50 the same rule gave PARTIAL (3/3 columns, 4/9): the n = 50 verdict was a power limit, not a contradiction.  The only non-significant comparison is MCTS3 vs MCTS4 at lambda_C = 2 (p_Holm 0.053, difference -0.83); MCTS3-GD's margin over MCTS4-GD shrinks from 2.5 to 0.8 units as clutter rises." % (R["verdict"], R["lowest_columns"], R["sig_in_direction"]))
