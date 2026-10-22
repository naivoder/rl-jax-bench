"""Full-scale analysis: reads the e1 (E1 grid, both configs) and e1s (sensitivity / parity arms) shards and prints / saves every result table.
Everything follows the pre-declared rules P1-P10 (README 'Results').  Usage:
    python scripts/report.py --part e1 --out runs/report/e1_report.json     # P1, P2, P3, P7(E2), P8, health
    python scripts/report.py --part sens --out runs/report/sens_report.json # P4, P5
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from src import model as M
from src import eval as E
import arms as ARMS
from rl_jax_bench.stats import paired_compare, holm, mean_se

ROOT = os.environ.get("GOSPA_RUNS", "runs")
CL = list(M.CLUTTER_RATES)
RUNS = 50
OFFSUM = json.load(open(os.path.join(HERE, "..", "results", "official_reference_summary.json")))


def ms(x):
    m, se = mean_se(x)
    return m, (se if se is not None else float("nan"))


def e1_part(out):
    res = {}
    for cfg in ("CFG-O", "CFG-P"):
        R = {}
        T = E.build_table(ROOT, "e1", cfg, M.ALG_ORDER, CL, RUNS)
        # ---- Table III ----
        rows = []
        grades = {"MATCH": 0, "CLOSE": 0, "DIVERGENT": 0}
        rels = []
        for a in M.ALG_ORDER:
            for lam in CL:
                t = T[a][lam]
                gm, gse = ms(t["gbar"])
                row = dict(alg=a, clutter=lam, G=t["G"], G_alt=t["G_alt"], ci=t["ci"], gbar_mean=gm, gbar_se=gse, n=t["n"], invalid=t["invalid"])
                if a in M.PAPER_TABLE3:
                    pp = M.PAPER_TABLE3[a][CL.index(lam)]
                    rel, g = E.grade(t["G"], pp)
                    row.update(paper=pp, rel=rel, grade=g)
                    grades[g] += 1
                    rels.append(rel)
                    ref = OFFSUM.get(f"def/{a}/{lam}")
                    if ref:
                        row["official"] = dict(G=ref["G"], n=ref["n"], ci=ref["ci"], inside_our_ci=bool(t["ci"][0] <= ref["G"] <= t["ci"][1]))
                    row["paper_inside_our_ci"] = bool(t["ci"][0] <= pp <= t["ci"][1])
                rows.append(row)
        R["table3"] = rows
        R["grades"] = grades
        R["median_rel"] = float(np.median(rels))
        R["max_rel"] = float(np.max(rels))
        # ---- P1 family ----
        fam = E.families(T, CL)
        R["P1_families"] = {k: {kk: vv for kk, vv in d.items() if kk not in ("diff_per_seed",)} for k, d in fam.items()}
        R["P1_verdicts"] = E.verdicts(fam, T, CL)
        R["C5"] = {str(lam): E.c5_check(T, lam) for lam in CL}
        # ---- P3 ablation family ----
        tests = {}
        for lam in CL:
            for ab in ("Myopic-GD", "MCTS3-KLD", "RANDOM"):
                tests[f"FA|{ab}>MCTS3-GD|{lam}"] = (T[ab][lam]["gbar"], T["MCTS3-GD"][lam]["gbar"])
        fa, pv = {}, {}
        for k, (a, b) in tests.items():
            fa[k] = paired_compare(a, b, n_boot=10000)
            pv[k] = fa[k]["t_p"]
        h = holm(pv)
        for k in fa:
            fa[k]["p_holm"] = h[k]["p_holm"]
            fa[k]["sig"] = bool(h[k]["p_holm"] < 0.05)
            fa[k]["ablated_worse"] = fa[k]["diff_mean"] > 0
            fa[k].pop("diff_per_seed", None)
        nconf = sum(v["sig"] and v["ablated_worse"] for v in fa.values())
        nopp = sum(v["sig"] and not v["ablated_worse"] for v in fa.values())
        R["P3_ablation"] = dict(tests=fa, confirmed=nconf, opposite=nopp,
                                verdict="CONFIRMED" if (nconf >= 8 and nopp == 0) else "PARTIAL" if nconf >= 5 else "NOT")
        # ---- myopic identity ----
        ident = {}
        for lam in CL:
            a, b = T["Myopic-GD"][lam]["cell"]["gospa_sq"], T["Myopic-KLD"][lam]["cell"]["gospa_sq"]
            ident[str(lam)] = dict(dG=T["Myopic-GD"][lam]["G"] - T["Myopic-KLD"][lam]["G"], frac_identical_runs=float(np.mean([np.allclose(a[i], b[i]) for i in range(len(a))])),
                                   G_min=float(min(np.sqrt(a.mean(0)).mean(), np.sqrt(b.mean(0)).mean())), run_sd_GD=float(T["Myopic-GD"][lam]["gbar"].std(ddof=1)))
        R["myopic_identity"] = ident
        # ---- E2: per-step curves at lambda 1 ----
        e2 = {}
        for a in ("Myopic-GD", "Myopic-KLD", "MCTS1-GD", "MCTS2-GD", "MCTS3-GD", "MCTS4-GD", "MCTS3-KLD"):
            c = T[a][1.0]["cell"]
            cur = E.fig5_curves(c)
            e2[a] = {k: v[::1].round(3).tolist() for k, v in cur.items()}
        R["E2_curves_lam1"] = e2
        # ---- health ----
        health = {}
        for a in M.ALG_ORDER:
            for lam in CL:
                c = T[a][lam]["cell"]["counters"]
                names = sorted(["birth_overflow", "meas_overflow", "est_overflow", "root_overflow", "plan_overflow", "nonfinite_filter", "lbp_nonconverged", "plan_nan_steps"])
                health[f"{a}|{lam}"] = {n: int(c[:, i].sum()) for i, n in enumerate(names)}
        R["health"] = health
        # ---- P8 determinism e1 runs 0-9 vs the 10-run archive (tag small) ----
        det = {}
        try:
            Tg = E.build_table(ROOT, "small", cfg, M.ALG_ORDER, CL, 10)
            for a in M.ALG_ORDER:
                for lam in CL:
                    x = T[a][lam]["cell"]["gospa_sq"][:10]
                    y = Tg[a][lam]["cell"]["gospa_sq"]
                    det[f"{a}|{lam}"] = float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1.0)))
        except Exception as e:  # noqa: BLE001
            det["error"] = repr(e)
        R["P8_determinism_max_rel"] = det
        R["P8_pass"] = bool(det and "error" not in det and max(det.values()) <= 1e-6)
        # ---- timing meta ----
        R["sec_per_step_150lanes"] = {}
        for a in M.ALG_ORDER:
            sh = json.load(open(os.path.join(ROOT, "e1", "seed_0", f"metrics__{cfg}__{a}.json")))["_meta"]
            R["sec_per_step_150lanes"][a] = dict(sec_per_step=sh["sec_per_step"], compile=sh["compile_seconds"])
        res[cfg] = R
    json.dump(res, open(out, "w"), indent=1, default=float)
    return res


def load_arm(alg, base, sets):
    n, safe = ARMS.cfg_name(base, sets)
    c = E.load_cell(ROOT, "e1s", n, alg, RUNS)
    return {lam: c[str(lam)] for lam in CL}


def sens_part(out):
    T0 = E.build_table(ROOT, "e1", "CFG-O", M.ALG_ORDER, CL, RUNS)
    R = {"FS": {}, "FT": {}}
    tests, meta = {}, {}
    for sid, (algs, sets) in ARMS.FS.items():
        for a in algs:
            try:
                arm = load_arm(a, "CFG-O", sets)
            except FileNotFoundError:
                continue
            for lam in CL:
                gb = E.run_scalar(arm[lam]["gospa_sq"])
                tests[f"{sid}|{a}|{lam}"] = (gb, T0[a][lam]["gbar"])
                meta[f"{sid}|{a}|{lam}"] = dict(G=E.table3_stat(arm[lam]["gospa_sq"])[0], G_base=T0[a][lam]["G"], invalid=E.counters_ok(arm[lam]),
                                                counters_plan_nan=int(arm[lam]["counters"][:, 7].sum()))
    fs, pv = {}, {}
    for k, (a, b) in tests.items():
        d = np.asarray(a) - np.asarray(b)
        fs[k] = dict(diff_mean=0.0, t_p=1.0, n=len(d)) if np.allclose(d, 0) else paired_compare(a, b, n_boot=10000)
        pv[k] = fs[k]["t_p"] if np.isfinite(fs[k]["t_p"]) else 1.0
    h = holm(pv)
    for k in fs:
        fs[k]["p_holm"] = h[k]["p_holm"]
        fs[k]["sig"] = bool(h[k]["p_holm"] < 0.05)
        fs[k].pop("diff_per_seed", None)
        fs[k].update(meta[k])
    R["FS"]["tests"] = fs
    mat = {}
    for sid, (algs, sets) in ARMS.FS.items():
        for a in algs:
            ks = [f"{sid}|{a}|{lam}" for lam in CL if f"{sid}|{a}|{lam}" in fs]
            if ks:
                mat[f"{sid}|{a}"] = bool(sum(abs(fs[k]["diff_mean"]) >= 2.0 and fs[k]["sig"] for k in ks) >= 2)
    R["FS"]["material"] = mat
    R["FS"]["n_material"] = int(sum(mat.values()))
    # ---- F-T parity: GD vs KLD under the same setting ----
    ft_tests, ft_meta = {}, {}
    settings = {"default": []}
    settings.update(ARMS.FT)
    for fid, sets in settings.items():
        for lam in CL:
            try:
                if sets:
                    gd = load_arm("MCTS3-GD", "CFG-O", sets)[lam]["gospa_sq"]
                    kl = load_arm("MCTS3-KLD", "CFG-O", sets)[lam]["gospa_sq"]
                else:
                    gd, kl = T0["MCTS3-GD"][lam]["cell"]["gospa_sq"], T0["MCTS3-KLD"][lam]["cell"]["gospa_sq"]
            except FileNotFoundError:
                continue
            ft_tests[f"{fid}|{lam}"] = (E.run_scalar(gd), E.run_scalar(kl))
            ft_meta[f"{fid}|{lam}"] = dict(G_GD=E.table3_stat(gd)[0], G_KLD=E.table3_stat(kl)[0])
    ft, pv = {}, {}
    for k, (a, b) in ft_tests.items():
        ft[k] = paired_compare(a, b, n_boot=10000)
        pv[k] = ft[k]["t_p"]
    h = holm({k: v for k, v in pv.items() if not k.startswith("default")})   # Holm over the 12 non-default tests
    for k in ft:
        ft[k]["p_holm"] = h[k]["p_holm"] if k in h else None
        ft[k]["sig"] = bool(h[k]["p_holm"] < 0.05) if k in h else None
        ft[k].pop("diff_per_seed", None)
        ft[k].update(ft_meta[k])
    nd = {k: v for k, v in ft.items() if not k.startswith("default")}
    R["FT"]["tests"] = ft
    R["FT"]["point_GD_better"] = int(sum(v["diff_mean"] < 0 for v in nd.values()))
    R["FT"]["sig_GD"] = int(sum(bool(v["sig"]) and v["diff_mean"] < 0 for v in nd.values()))
    R["FT"]["sig_KLD"] = int(sum(bool(v["sig"]) and v["diff_mean"] > 0 for v in nd.values()))
    R["FT"]["n_tests"] = len(nd)
    R["FT"]["verdict"] = "H3 ROBUST" if (len(nd) == 12 and R["FT"]["point_GD_better"] >= 11 and R["FT"]["sig_GD"] >= 6) else "NOT ROBUST"
    json.dump(R, open(out, "w"), indent=1, default=float)
    return R


def strut_part(out):
    algs = ["Myopic-GD", "MCTS1-GD", "MCTS2-GD", "MCTS3-GD", "MCTS4-GD", "MCTS1-KLD", "MCTS2-KLD", "MCTS3-KLD", "MCTS4-KLD"]
    T = E.build_table(ROOT, "e1t", "CFG-O", algs, CL, RUNS)
    R = {"G": {a: {str(l): dict(G=T[a][l]["G"], ci=T[a][l]["ci"], gbar=ms(T[a][l]["gbar"]), invalid=T[a][l]["invalid"]) for l in CL} for a in algs}}
    fam = E.families(T, CL)
    R["families"] = {k: {kk: vv for kk, vv in d.items() if kk != "diff_per_seed"} for k, d in fam.items()}
    R["verdicts"] = E.verdicts(fam, T, CL)
    # per-truth means of gbar for MCTS3-GD and Myopic-GD (5 runs per truth)
    pt = {}
    for a in ("Myopic-GD", "MCTS3-GD", "MCTS3-KLD"):
        pt[a] = {str(l): [float(np.mean(T[a][l]["gbar"][np.arange(RUNS) % 10 == j])) for j in range(10)] for l in CL}
    R["per_truth_gbar"] = pt
    json.dump(R, open(out, "w"), indent=1, default=float)
    return R


def div_part(out):
    R = {}
    for name, sets in (("filt_pd=hard", ["filt_pd=hard"]), ("one_sweep_LBP", ["lbp_mode=one_sweep", "lbp_tol=1e-4", "lbp_max_iter=1"]), ("both_hard", ["filt_pd=hard", "truth_pd=hard"])):
        n, safe = ARMS.cfg_name("CFG-P", sets)
        for a in ("Myopic-GD", "MCTS3-GD"):
            try:
                c = E.load_cell(ROOT, "e1d", n, a, RUNS)
            except FileNotFoundError:
                continue
            for lam in CL:
                G, _ = E.table3_stat(c[str(lam)]["gospa_sq"])
                pp = M.PAPER_TABLE3[a][CL.index(lam)]
                rel, g = E.grade(G, pp)
                R[f"{name}|{a}|{lam}"] = dict(G=G, ci=E.boot_G(c[str(lam)]["gospa_sq"], 10000), paper=pp, rel=rel, grade=g, invalid=E.counters_ok(c[str(lam)]))
    json.dump(R, open(out, "w"), indent=1, default=float)
    return R


def h2_part(out):
    """P10: H2 re-tested on the pooled n = 200 runs (e1 runs 0-49 + e2h runs 50-199) for CFG-O."""
    import shutil
    algs = ["MCTS1-GD", "MCTS2-GD", "MCTS3-GD", "MCTS4-GD"]
    for a in algs:                                            # make e2h hold all 200 shards of these cells (runs 0-49 copied from e1)
        for m in range(50):
            dst = os.path.join(ROOT, "e2h", f"seed_{m}", f"metrics__CFG-O__{a}.json")
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(ROOT, "e1", f"seed_{m}", f"metrics__CFG-O__{a}.json"), dst)
    n = 200
    T = {}
    for a in algs:
        c = E.load_cell(ROOT, "e2h", "CFG-O", a, n)
        T[a] = {}
        for lam in CL:
            x = c[str(lam)]
            T[a][lam] = dict(G=E.table3_stat(x["gospa_sq"])[0], ci=E.boot_G(x["gospa_sq"], 10000), gbar=E.run_scalar(x["gospa_sq"]), invalid=E.counters_ok(x), n=int(x["gospa_sq"].shape[0]))
    tests = {}
    for lam in CL:
        for v in ("MCTS1-GD", "MCTS2-GD", "MCTS4-GD"):
            tests[f"H2|MCTS3-GD<{v}|{lam}"] = (T["MCTS3-GD"][lam]["gbar"], T[v][lam]["gbar"])
    res, pv = {}, {}
    for k, (a, b) in tests.items():
        res[k] = paired_compare(a, b, n_boot=10000)
        pv[k] = res[k]["t_p"]
    h = holm(pv)
    for k in res:
        res[k]["p_holm"] = h[k]["p_holm"]; res[k]["sig"] = bool(h[k]["p_holm"] < 0.05); res[k]["better_first"] = res[k]["diff_mean"] < 0
        res[k].pop("diff_per_seed", None)
    lowest = 0
    for lam in CL:
        g = {v: T[v][lam]["G"] for v in algs}
        lowest += int(min(g, key=g.get) == "MCTS3-GD")
    n2 = sum(v["sig"] and v["better_first"] for v in res.values())
    R = dict(n=n, G={a: {str(l): dict(G=T[a][l]["G"], ci=T[a][l]["ci"], gbar=ms(T[a][l]["gbar"]), invalid=T[a][l]["invalid"]) for l in CL} for a in algs}, tests=res,
             lowest_columns=lowest, sig_in_direction=n2,
             verdict="REPRODUCED" if (lowest == 3 and n2 >= 7) else "PARTIAL" if (lowest >= 2 or n2 >= 4) else "NOT")
    json.dump(R, open(out, "w"), indent=1, default=float)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["e1", "sens", "strut", "div", "h2"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = {"e1": e1_part, "sens": sens_part, "strut": strut_part, "div": div_part, "h2": h2_part}[a.part](a.out)
    print("written", a.out)
