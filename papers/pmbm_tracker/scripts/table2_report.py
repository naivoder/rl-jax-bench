"""Table II sweep report: 15 Table II cells x arms, grades, paired orderings, Holm family, sensitivity rows.
Reads runs/table2_*/seed_*/metrics.json only.  usage: python scripts/table2_report.py [--out OUT.md] [--json OUT.json]
Grade: MATCH <= 5 %, CLOSE <= 20 %, DIVERGENT > 20 % relative to the printed Table II value.  Raises on a missing/invalid run."""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..")]
from src import eval as EV

PDS = (0.95, 0.9, 0.8, 0.7, 0.6); LAMS = (10, 15, 20)
CELLS = [(p, l) for p in PDS for l in LAMS]
# Table II exactly as printed: Est1 Est2 Est3 TOMB MOMB CPHD PHD
T2 = {(0.95, 10): (2.10, 2.10, 2.10, 2.32, 2.10, 2.83, 6.34), (0.95, 15): (2.15, 2.17, 2.15, 2.48, 2.17, 2.97, 6.44), (0.95, 20): (2.26, 2.27, 2.26, 2.61, 2.27, 3.00, 6.51),
      (0.9, 10): (2.23, 2.34, 2.36, 2.65, 2.37, 3.39, 7.05), (0.9, 15): (2.30, 2.42, 2.44, 2.75, 2.45, 3.45, 7.04), (0.9, 20): (2.37, 2.48, 2.50, 2.80, 2.53, 2.57, 7.18),
      (0.8, 10): (2.67, 2.64, 2.66, 2.95, 2.78, 4.19, 8.22), (0.8, 15): (2.80, 2.78, 2.80, 3.15, 2.88, 4.25, 8.23), (0.8, 20): (2.93, 2.90, 2.92, 3.18, 3.00, 4.48, 8.34),
      (0.7, 10): (3.02, 2.99, 3.01, 3.47, 3.15, 4.83, 8.80), (0.7, 15): (3.10, 3.07, 3.09, 3.57, 3.24, 4.99, 8.86), (0.7, 20): (3.29, 3.25, 3.28, 3.67, 3.41, 5.09, 8.87),
      (0.6, 10): (3.42, 3.39, 3.42, 3.81, 3.55, 5.30, 9.09), (0.6, 15): (3.62, 3.60, 3.62, 4.03, 3.72, 5.52, 9.14), (0.6, 20): (3.71, 3.69, 3.71, 4.09, 3.82, 5.61, 9.18)}
COL = dict(est1=0, est2=1, est3=2, tomb=3, momb=4, cphd=5, phd=6)

SKIP = set()    # sensitivity variants not complete at analysis time (set by --skip)

def load(tag, n=None):
    ps = sorted(glob.glob(os.path.join(HERE, "..", "runs", tag, "seed_*", "metrics.json")), key=lambda p: int(p.split("seed_")[-1].split("/")[0]))
    recs = [json.load(open(p)) for p in ps]
    if n is not None and [r["run"] for r in recs] != list(range(n)):
        raise RuntimeError(f"{tag}: runs {[r['run'] for r in recs][:3]}.. != 0..{n - 1} (have {len(recs)})")
    bad = [r["run"] for r in recs if not r.get("valid", True)]
    if bad:
        raise RuntimeError(f"{tag}: invalid runs {bad}")
    return recs

def arm(tag, algo, n):
    return np.array([r["algos"][algo]["ospa2"] for r in load(tag, n)])   # [R,K]

def grade(rel):
    a = abs(rel); return "MATCH" if a <= 0.05 else ("CLOSE" if a <= 0.20 else "DIVERGENT")
G = {"MATCH": "M", "CLOSE": "C", "DIVERGENT": "**D**"}

def fmt(d2, paper):
    c = EV.table_cell(d2, n_boot=1000)
    rel = c["value"] / paper - 1 if paper else None
    return c, rel

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--skip", default="");
    ap.add_argument("--out", default=None); ap.add_argument("--json", default=None); a = ap.parse_args(); n = a.n; SKIP.update(x for x in a.skip.split(",") if x)
    S = {}      # (cell) -> {arm: d2 [R,K]}
    for (p, l) in CELLS:
        s = {}
        for k in (1, 2, 3):
            s[f"est{k}"] = arm(f"table2_pmbm_pd{p}_l{l}", f"pmbm_est{k}", n)
            s[f"est{k}_pw"] = arm(f"table2_pmbmpw_pd{p}_l{l}", f"pmbm_est{k}", n)
        for x in ("tomb", "momb", "tomb_map", "momb_map"):
            s[x] = arm(f"table2_wil_pd{p}_l{l}", x, n)
        s["phd"] = arm(f"table2_cited_pd{p}_l{l}", "phd_vm", n); s["cphd"] = arm(f"table2_cited_pd{p}_l{l}", "cphd", n)   # graded
        for v, tag in (("cited", "table2_cited"), ("repo", "table2_repo"), ("m4c30", "table2_m4c30"), ("m01c100", "table2_m01c100")):
            if v in SKIP: continue
            for al in ("phd", "phd_vm", "cphd"):
                s[f"{al}@{v}"] = arm(f"{tag}_pd{p}_l{l}", al, n)
        S[(p, l)] = s
    out = []; P = out.append; J = {}
    # ---- cell tables
    P("| (p_d, lambda_c) | Est 1 | Est 2 | Est 3 | TOMB | MOMB | CPHD [graded: merge 4, J 100] | PHD [graded: Vo-Ma, merge 4, J 100] |")
    P("|---|---|---|---|---|---|---|---|")
    cnt = {k: {"MATCH": 0, "CLOSE": 0, "DIVERGENT": 0} for k in COL}
    for (p, l) in CELLS:
        row = []; J[f"{p}_{l}"] = {}
        for k, ci in COL.items():
            c, rel = fmt(S[(p, l)][k], T2[(p, l)][ci]); g = grade(rel); cnt[k][g] += 1
            J[f"{p}_{l}"][k] = dict(value=c["value"], se=c["se_delta"], paper=T2[(p, l)][ci], rel=rel, grade=g)
            row.append(f"{c['value']:.3f} +- {c['se_delta']:.3f} ({100 * rel:+.1f} %) {G[g]}")
        P(f"| ({p}, {l}) | " + " | ".join(row) + " |")
    P(""); P("Grade counts over 15 cells (MATCH/CLOSE/DIVERGENT): " + "; ".join(f"{k} {v['MATCH']}/{v['CLOSE']}/{v['DIVERGENT']}" for k, v in cnt.items()))
    J["grade_counts"] = cnt
    # ---- paper values row for reference + sensitivity tables
    def sens_table(title, arms, paper_col):
        P(""); P(f"**{title}**"); P(""); P("| (p_d, lambda_c) | " + " | ".join(arms) + " |"); P("|---|" + "---|" * len(arms))
        cc = {x: {"MATCH": 0, "CLOSE": 0, "DIVERGENT": 0} for x in arms}
        for (p, l) in CELLS:
            row = []
            for x in arms:
                c, rel = fmt(S[(p, l)][x], T2[(p, l)][paper_col[x]]); g = grade(rel); cc[x][g] += 1
                row.append(f"{c['value']:.3f} ({100 * rel:+.1f} %) {G[g]}")
            P(f"| ({p}, {l}) | " + " | ".join(row) + " |")
        P(""); P("counts M/C/D: " + "; ".join(f"{x} {v['MATCH']}/{v['CLOSE']}/{v['DIVERGENT']}" for x, v in cc.items())); return cc
    J["sens_pw"] = sens_table("Sensitivity: PMBM with the authors' HEAD global-weight floor prune_w = 1e-4 (not graded)", ["est1_pw", "est2_pw", "est3_pw"], dict(est1_pw=0, est2_pw=1, est3_pw=2))
    J["sens_map"] = sens_table("Sensitivity: Williams TOMB/MOMB with the ancillary MAP-cardinality rule (not graded)", ["tomb_map", "momb_map"], dict(tomb_map=3, momb_map=4))
    for v, lab in (("repo", "post-paper repo defaults: merge 0.1, cap 30"), ("m4c30", "merge 4, cap 30"), ("m01c100", "merge 0.1, cap 100"), ("cited", "merge 4, cap 100 (graded for phd_vm and cphd)")):
        if v in SKIP: continue
        J[f"sens_{v}"] = sens_table(f"Sensitivity: PHD/CPHD, {lab} (not graded unless stated)", [f"phd@{v}", f"phd_vm@{v}", f"cphd@{v}"], {f"phd@{v}": 6, f"phd_vm@{v}": 6, f"cphd@{v}": 5})
    # ---- paired ordering: Est 1 vs every baseline, per cell
    P(""); P("**Paired ordering: PMBM Est 1 vs each baseline, per cell** (paired over 100 runs on per-run mean squared OSPA; cell = ratio of joint RMS Est1/baseline with bootstrap 95 % CI, wins = runs in which Est 1 is lower, paired-t p; 'Est1<' = Est 1 lower, significant at raw p < 0.05)"); P("")
    bl = ["tomb", "momb", "cphd", "phd"]
    P("| (p_d, lambda_c) | " + " | ".join(f"vs {b.upper()}" for b in bl) + " |"); P("|---|" + "---|" * len(bl))
    PAIR = {}
    for (p, l) in CELLS:
        row = []; PAIR[f"{p}_{l}"] = {}
        for b in bl + ["est2", "est3"]:
            r = EV.paired_test(EV.run_stat(S[(p, l)]["est1"]), EV.run_stat(S[(p, l)][b]), n_boot=2000)
            lo, hi = r["rel_boot95"]
            ratio = np.sqrt(EV.run_stat(S[(p, l)]["est1"]).mean() / EV.run_stat(S[(p, l)][b]).mean())
            d = r["diff_mean"]; sig = r["t_p"] < 0.05
            tag = ("Est1<" if d < 0 else "Est1>") if sig else "tie"
            PAIR[f"{p}_{l}"][b] = dict(ratio_rms=float(ratio), wins_est1=int(r["n"] - r["wins"] - int(np.sum(np.isclose(EV.run_stat(S[(p, l)]["est1"]), EV.run_stat(S[(p, l)][b]))))), t=r["t"], p=r["t_p"], verdict=tag)
            if b in bl:
                w = int(np.sum(EV.run_stat(S[(p, l)]["est1"]) < EV.run_stat(S[(p, l)][b])))
                row.append(f"{ratio:.3f} [{(1 + lo) ** 0.5:.3f}, {(1 + hi) ** 0.5:.3f}], {w}/100, p={r['t_p']:.1e}, {tag}")
        P(f"| ({p}, {l}) | " + " | ".join(row) + " |")
    J["pairs"] = PAIR
    P(""); P("Est 1 vs Est 2 / Est 3 (same filter, three estimators): ")
    P(""); P("| (p_d, lambda_c) | Est1 vs Est2 | Est1 vs Est3 |"); P("|---|---|---|")
    for (p, l) in CELLS:
        row = []
        for b in ("est2", "est3"):
            q = PAIR[f"{p}_{l}"][b]; w = int(np.sum(EV.run_stat(S[(p, l)]["est1"]) < EV.run_stat(S[(p, l)][b])))
            row.append(f"ratio {q['ratio_rms']:.3f}, {w}/100, p={q['p']:.1e}, {q['verdict']}")
        P(f"| ({p}, {l}) | " + " | ".join(row) + " |")
    # ---- Holm family (declared)
    def family(phd_arm, cphd_arm, label):
        tests = {}      # name -> (A, B, claim "A lower", paper_sep)
        for (p, l) in CELLS:
            s = S[(p, l)]; t2 = T2[(p, l)]; best = "est1" if p >= 0.9 else "est2"
            base = dict(tomb=s["tomb"], momb=s["momb"], cphd=s[cphd_arm], phd=s[phd_arm])
            for b in ("tomb", "momb", "cphd", "phd"):
                tests[f"C1 ({p},{l}) {best}<{b}"] = (s[best], base[b], t2[COL[b]] - t2[COL[best]])
            if p >= 0.9:
                for o in ("est2", "est3"):
                    tests[f"C2 ({p},{l}) est1<{o}"] = (s["est1"], s[o], t2[COL[o]] - t2[0])
            else:
                for o in ("est1", "est3"):
                    tests[f"C3 ({p},{l}) est2<{o}"] = (s["est2"], s[o], t2[COL[o]] - t2[1])
            for x, y in (("momb", "tomb"), ("tomb", "cphd"), ("cphd", "phd")):
                tests[f"C4 ({p},{l}) {x}<{y}"] = (base[x], base[y], t2[COL[y]] - t2[COL[x]])
        res = {}
        for nm, (A, B, sep) in tests.items():
            r = EV.paired_test(EV.run_stat(A), EV.run_stat(B), n_boot=2000)
            pv = r["t_p"] if np.isfinite(r["t_p"]) else 1.0     # identical arms (zero paired difference) give nan: no evidence, p = 1
            res[nm] = dict(p=pv, diff=r["diff_mean"], t=r["t"], boot95=r["boot95"], sep=sep)
        H = EV.holm_adjust({k: v["p"] for k, v in res.items()})
        fam = {}
        for nm, v in res.items():
            sig = H[nm]["p_holm"] < 0.05; lower = v["diff"] < 0
            v["p_holm"] = H[nm]["p_holm"]; v["sig_dir"] = "sig-as-claimed" if sig and lower else ("SIG-OPPOSITE" if sig else "n.s.")
            v["required"] = v["sep"] >= 0.05
            fam.setdefault(nm.split()[0], []).append((nm, v))
        P(""); P(f"**Holm family ({label})**: {len(res)} tests, alpha = 0.05"); P("")
        P("| claim | tests | significant in the claimed direction (Holm) | n.s. | significant OPPOSITE | tests the paper separates by >= 0.05 | of those significant as claimed | REPRODUCED |"); P("|---|---|---|---|---|---|---|---|")
        for c, lst in fam.items():
            ok = sum(v["sig_dir"] == "sig-as-claimed" for _, v in lst); ns = sum(v["sig_dir"] == "n.s." for _, v in lst); op = sum(v["sig_dir"] == "SIG-OPPOSITE" for _, v in lst)
            req = [(nm, v) for nm, v in lst if v["required"]]; reqok = sum(v["sig_dir"] == "sig-as-claimed" for _, v in req)
            P(f"| {c} | {len(lst)} | {ok} | {ns} | {op} | {len(req)} | {reqok} | {'YES' if reqok == len(req) and op == 0 else 'NO'} |")
        bad = [(nm, v) for nm, v in res.items() if v["sig_dir"] != "sig-as-claimed" and (v["required"] or v["sig_dir"] == "SIG-OPPOSITE")]
        P(""); P("Tests that are not significant as claimed although the paper separates them by >= 0.05, or that are significant opposite:"); P("")
        for nm, v in bad:
            P(f"* {nm}: mean diff {v['diff']:+.4f}, t = {v['t']:.2f}, raw p = {v['p']:.2g}, Holm p = {v['p_holm']:.2g}, paper sep {v['sep']:+.2f} -> {v['sig_dir']}")
        if not bad: P("* (none)")
        return res
    J["family_graded"] = family("phd", "cphd", "graded arms: PHD = Vo-Ma merge 4 J 100, CPHD merge 4 J 100")
    J["family_repo"] = family("phd@repo", "cphd@repo", "labelled robustness: PHD = GMPHD_estimation2 merge 0.1 cap 30, CPHD merge 0.1 cap 30 (earlier declaration)")
    if a.out: open(a.out, "w").write("\n".join(out) + "\n")
    if a.json: json.dump(J, open(a.json, "w"), indent=1, default=float)
    print("\n".join(out))

if __name__ == "__main__":
    main()
