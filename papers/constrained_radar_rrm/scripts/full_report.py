"""Full-run report: seed-level statistics from runs/full/eval/*.json (numpy/scipy only, no jax).

Seed unit: learner lane (variant, seed s) value = mean over its 20 replica draws (T_init pooled as
sum/count); baseline pairing-s value = mean over the SAME 20 draws -> 5 paired differences per comparison.
Primary family H1-H9 under A-04 with the declared learner config, Holm over 9; the same
family under A-04a is a labelled sensitivity (own Holm). Writes runs/full/report.json and runs/full/report.md,
and runs/seed_<n>/metrics.json.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

from rl_jax_bench.stats import boot_ci, holm, mean_se, paired_compare, t_interval  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = "full_v2"                       # the re-evaluation (re-digitised replica, v2 keys); "full" = full-run record
EV = os.path.join(REPO, "runs", TAG, "eval")
CATA = -1.0e6                         # a seed below this is 'catastrophic' (its magnitude is numerical)
NP, NR = 5, 20
PAPER_U = {  # Table VII
    "case1": dict(ddpg=-1.90e4, dql=-1.96e4, ea=-2.42e4, da=-1.94e4, oa=-1.82e4),
    "case2": dict(ddpg=-3.54e4, dql=-3.59e4, ea=-4.10e4, da=-3.90e4, oa=-3.59e4),
    "case3f": dict(ddpg=-8.14e3, dql=-7.91e3, ea=-13.7e3, da=-7.96e3, oa=-6.72e3),
    "case3s": dict(ddpg=-8.14e3, dql=-7.91e3, ea=-13.7e3, da=-7.96e3, oa=-6.72e3),
    "case4": dict(ddpg=-5.11e4, dql=-5.06e4, ea=-5.57e4, da=-5.59e4, oa=-5.05e4),
}
PAPER_TINIT = dict(ddpg=6.0, dql=7.83, ea=10.5, da=14.67, oa=20.33)
FIG9 = {  # paper Fig. 9, x 1e4; scan share 0.1..0.9
    "U": dict(ea=[-2.45, -2.65, -3.0, -3.35, -3.8, -4.55, -5.6, -7.65, -12.7],
              da=[-1.95, -2.15, -2.4, -2.7, -3.1, -3.65, -4.55, -6.3, -10.5],
              oa=[-1.8, -2.0, -2.2, -2.5, -2.95, -3.45, -4.4, -6.0, -10.0]),
    "c": dict(ea=[2.4, 2.65, 2.95, 3.3, 3.8, 4.5, 5.6, 7.7, 12.8], da=[1.85, 2.05, 2.35, 2.65, 3.05, 3.6, 4.55, 6.3, 10.5],
              oa=[1.8, 2.0, 2.25, 2.5, 2.95, 3.45, 4.35, 6.0, 10.1]),
    "tinit": dict(ea=[29.5, 10.0, 6.5, 5.0, 6.3, 4.3, 4.3, 4.7, 3.8], da=[18.7, 8.0, 6.3, 8.6, 8.4, 4.8, 4.8, 3.5, 4.2],
                  oa=[13.7, 16.8, 11.2, 6.6, 7.1, 5.6, 3.9, 6.3, 4.8]),
}
FIG9_LEARN = dict(ddpg=dict(U=-1.85e4, c=1.9e4, tinit=6.5), dql=dict(U=-1.95e4, c=1.95e4, tinit=11.7))


def judge(ours, paper, prec=0.0):
    if paper == 0:
        return "n/a", float("nan")
    rel = (ours - paper) / abs(paper)
    if abs(ours - paper) <= prec or abs(rel) <= 0.05:
        return "MATCH", rel
    return ("CLOSE" if abs(rel) <= 0.20 else "DIVERGENT"), rel


def load(p):
    with open(p) as f:
        return json.load(f)


def seedvals_base(m, key, sl="replica"):
    """Baseline per-pairing values [NP] (mean over the 20 draws of each pairing)."""
    x = np.asarray(m[sl][key], float)
    if sl == "spawn":
        return np.full(NP, x.mean())
    return x.reshape(NP, NR).mean(1)


def tinit_base(m, sl="replica"):
    s, n = np.asarray(m[sl]["tinit_sum"], float), np.asarray(m[sl]["tinit_n"], float)
    if sl == "spawn":
        return np.full(NP, s.sum() / max(n.sum(), 1))
    return s.reshape(NP, NR).sum(1) / np.maximum(n.reshape(NP, NR).sum(1), 1)


class Learn:
    def __init__(self, p):
        self.m = load(p)
        self.lanes = [tuple(x) for x in self.m["lanes"]]

    def rows(self, variant):
        idx = [i for i, (v, _) in enumerate(self.lanes) if v == variant]
        seeds = [self.lanes[i][1] for i in idx]
        return idx, seeds

    def vals(self, variant, key, sl="replica"):
        idx, seeds = self.rows(variant)
        x = np.asarray(self.m[sl][key], float)[idx]           # [n, draws]
        order = np.argsort(seeds)
        return x.mean(1)[order]

    def tinit(self, variant, sl="replica"):
        idx, seeds = self.rows(variant)
        s = np.asarray(self.m[sl]["tinit_sum"], float)[idx].sum(1)
        n = np.asarray(self.m[sl]["tinit_n"], float)[idx].sum(1)
        return (s / np.maximum(n, 1))[np.argsort(seeds)]


def one_sided(pc, greater=True):
    t, p = pc["t"], pc["t_p"]
    if not np.isfinite(t):
        return 1.0 if not np.isfinite(p) else p
    return p / 2 if (t > 0) == greater else 1 - p / 2


def ms(x):
    m, se = mean_se(x)
    return dict(mean=m, se=se, t95=t_interval(x), n=len(x), per_seed=np.asarray(x).round(4).tolist())


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if abs(x) >= 1e6:
        return f"{x:.3g}"
    return f"{x:,.{nd}f}"


def main():
    global TAG, EV
    if len(sys.argv) > 1 and sys.argv[1].startswith("--tag="):
        TAG = sys.argv[1].split("=", 1)[1]
        EV = os.path.join(REPO, "runs", TAG, "eval")
    out = dict(meta={}, cases={}, families={}, preds={}, ablation={}, debug={}, comparable={}, c5c7c8={}, fig9={})
    out["meta"]["tag"] = TAG
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out["meta"]["git_commit"] = commit
    calibs = ["A04", "A04a"]
    case_runs = {"case1": "case1", "case2": "case2", "case3f": "case3", "case3s": "case3", "case4": "case1"}
    L = {}
    for p in glob.glob(os.path.join(EV, "learn_*.json")):
        if p.endswith(("_proj.json",)) or os.path.basename(p).startswith("learn_recheck_"):
            continue                                           # projections / the re-evaluation arms: scripts/recheck_report.py
        nm = os.path.basename(p)[6:-5]                         # e.g. ddpg_case1_case4
        L[nm] = Learn(p)
    B = {}
    for p in glob.glob(os.path.join(EV, "base_*.json")):
        _, case, ck = os.path.basename(p)[:-5].split("_")
        B[(case, ck)] = load(p)
    md = []

    # ------------------------------------------------------------------ per-case tables (U, c, T_init, budget)
    for case in ["case1", "case2", "case3f", "case3s", "case4"]:
        for ck in calibs:
            if (case, ck) not in B:
                continue
            b = B[(case, ck)]
            tab = {}
            for kind in b:
                tab[kind] = dict(U=ms(seedvals_base(b[kind], "U")), c=ms(seedvals_base(b[kind], "c")),
                                 scan=ms(seedvals_base(b[kind], "scan_term")), tinit=ms(tinit_base(b[kind])),
                                 sat=ms(seedvals_base(b[kind], "sat_rate")), g=ms(seedvals_base(b[kind], "g_mean")))
                if "spawn" in b[kind]:
                    tab[kind]["spawn_U"] = float(np.mean(b[kind]["spawn"]["U"]))
                    tab[kind]["spawn_tinit"] = float(tinit_base(b[kind], "spawn")[0])
            for algo in ["ddpg", "dql"]:
                for scope in ([case_runs[case]] + (["case1"] if case.startswith("case3") else [])):
                    key = f"{algo}_{scope}_{case}"
                    if key not in L:
                        continue
                    lr = L[key]
                    for v in sorted({x for x, _ in lr.lanes}):
                        vv = v if v != "A04a" else "A04a"
                        if (ck == "A04" and v.startswith("A04a")) or (ck == "A04a" and not v.startswith("A04a")):
                            continue
                        nm = f"{algo}[{v}]" + ("(zero-shot I)" if scope == "case1" and case.startswith("case3") else "")
                        d = dict(U=ms(lr.vals(v, "U")), n_catastrophic=int((lr.vals(v, "U") < CATA).sum()),
                                 c=ms(lr.vals(v, "c")), scan=ms(lr.vals(v, "scan_term")),
                                 tinit=ms(lr.tinit(v)), sat=ms(lr.vals(v, "sat_rate")), g=ms(lr.vals(v, "g_mean")),
                                 lam_multi=ms(lr.vals(v, "lam_multi")), lam_max=ms(lr.vals(v, "lam_max")),
                                 share=ms(lr.vals(v, "track_share")), starve=ms(lr.vals(v, "starve")),
                                 viol_near=ms(lr.vals(v, "viol_near_frac")), g_disc=ms(lr.vals(v, "g_disc")))
                        if "spawn" in lr.m:
                            d["spawn_U"] = ms(lr.vals(v, "U", "spawn"))
                            d["spawn_tinit"] = ms(lr.tinit(v, "spawn"))
                            d["spawn_c"] = ms(lr.vals(v, "c", "spawn"))
                        # paired vs every baseline, gap fraction
                        U_l = lr.vals(v, "U")
                        d["vs"] = {}
                        for kind in b:
                            d["vs"][kind] = paired_compare(U_l, seedvals_base(b[kind], "U"))
                        ua, ud, uo = (seedvals_base(b[k], "U") for k in ("ea", "da", "oa"))
                        best_h = ud if ud.mean() >= ua.mean() else ua
                        G = (U_l - best_h) / (uo - best_h)
                        d["G"] = dict(**ms(G), boot95=boot_ci(G), heuristic="DA" if ud.mean() >= ua.mean() else "EA")
                        if "oa_myopic" in b:
                            um = seedvals_base(b["oa_myopic"], "U")
                            d["G_myopic"] = ms((U_l - best_h) / (um - best_h))
                        tab[nm] = d
            out["cases"][f"{case}|{ck}"] = tab

    # ------------------------------------------------------------------ primary family H1-H9 (+ A-04a sensitivity)
    for ck, var in [("A04", "A04"), ("A04a", "A04a")]:
        fam, pv = {}, {}
        try:
            b1, b2 = B[("case1", ck)], B[("case2", ck)]
            dd1, dq1 = L["ddpg_case1_case1"], L["dql_case1_case1"]
            dd2, dq2 = L["ddpg_case2_case2"], L["dql_case2_case2"]
        except KeyError as e:
            print("missing for family", ck, e)
            continue
        spec = [("H1", dd1.vals(var, "U"), seedvals_base(b1["ea"], "U"), "greater"),
                ("H2", dq1.vals(var, "U"), seedvals_base(b1["ea"], "U"), "greater"),
                ("H3", dd1.vals(var, "U"), seedvals_base(b1["da"], "U"), "greater"),
                ("H4", dq1.vals(var, "U"), seedvals_base(b1["da"], "U"), "greater"),
                ("H5", dd2.vals(var, "U"), seedvals_base(b2["da"], "U"), "greater"),
                ("H6", dq2.vals(var, "U"), seedvals_base(b2["da"], "U"), "greater"),
                ("H7", dd1.tinit(var), tinit_base(b1["oa"]), "less"),
                ("H8", dd1.vals(var, "U"), seedvals_base(b1["oa"], "U"), "two"),
                ("H9", dd2.vals(var, "U"), seedvals_base(b2["oa"], "U"), "two")]
        for hid, a_, b_, side in spec:
            pc = paired_compare(a_, b_)
            p = pc["t_p"] if side == "two" else one_sided(pc, greater=(side == "greater"))
            if not np.isfinite(p):
                p = 1.0
            pv[hid] = p
            fam[hid] = dict(side=side, a=ms(a_), b=ms(b_), **{k: pc[k] for k in
                            ("diff_mean", "diff_se", "t95", "t", "t_p", "boot95", "wins", "rel_improvement", "rel_boot95")},
                            p_test=p)
        hh = holm(pv)
        for hid in fam:
            fam[hid].update(hh[hid])
            side = fam[hid]["side"]
            if side == "two":
                win = "learner" if fam[hid]["diff_mean"] > 0 else "OA"
                fam[hid]["verdict"] = (f"{win} better" if hh[hid]["p_holm"] < 0.05 else "no detectable difference")
            else:
                fam[hid]["verdict"] = "SUPPORTED" if hh[hid]["p_holm"] < 0.05 else "NOT SUPPORTED"
        out["families"][ck] = fam

        # comparable claims C3 (learners ~ OA, Case I) and C6 (DDPG ~ DQL, every case)
        comp = {}

        def comparable(a_, b_):
            pc = paired_compare(a_, b_)
            lo, hi = pc["rel_boot95"]
            v = "SUPPORTED" if (lo >= -0.05 and hi <= 0.05) else ("CONTRADICTED" if (hi < -0.05 or lo > 0.05)
                                                                  else "UNDETERMINED")
            return dict(rel=pc["rel_improvement"], rel_boot95=pc["rel_boot95"], verdict=v)
        comp["C3_ddpg_vs_oa_I"] = comparable(dd1.vals(var, "U"), seedvals_base(b1["oa"], "U"))
        comp["C3_dql_vs_oa_I"] = comparable(dq1.vals(var, "U"), seedvals_base(b1["oa"], "U"))
        for case in ["case1", "case2", "case3f", "case4"]:
            sc = case_runs[case]
            if f"ddpg_{sc}_{case}" in L and f"dql_{sc}_{case}" in L:
                comp[f"C6_ddpg_vs_dql_{case}"] = comparable(L[f"ddpg_{sc}_{case}"].vals(var, "U"),
                                                            L[f"dql_{sc}_{case}"].vals(var, "U"))
        # C5: OA vs DDPG on spawn episodes (Fig. 9 at scan share 0.1)
        oa_sp = seedvals_base(b1["oa"], "U", "spawn")
        comp["C5_oa_minus_ddpg_spawn"] = paired_compare(oa_sp, dd1.vals(var, "U", "spawn"))
        comp["C5_oa_minus_dql_spawn"] = paired_compare(oa_sp, dq1.vals(var, "U", "spawn"))
        comp["C5_test_type"] = ("ONE-SAMPLE t of the 5 learner seed means against OA's mean over the 50 shared spawn "
                                "episodes (oa_sp is a constant vector; OA's episode variance is not in the test)")
        out["comparable"][ck] = comp

        # ------------------------------------------------------------- P1-P6 (Case I replica)
        P = {}
        eq = all(np.array_equal(np.asarray(b1[k]["replica"][f]), np.asarray(b1["ea"]["replica"][f]))
                 for k in ("da", "oa", "oa_myopic") for f in ("tinit_sum", "tinit_n", "nmiss", "never"))
        P["P1"] = dict(identical=bool(eq), verdict="GO" if eq else "KILL")
        shares = {}
        for k in ("ea", "da", "oa", "oa_myopic"):
            shares[k] = float(np.mean(seedvals_base(b1[k], "scan_term") / np.abs(seedvals_base(b1[k], "U"))))
        for nm, lr in (("ddpg", dd1), ("dql", dq1)):
            shares[nm] = float(np.mean(lr.vals(var, "scan_term") / np.abs(lr.vals(var, "U"))))
        P["P2"] = dict(scan_share_of_U=shares, verdict="GO" if max(shares.values()) <= 0.03 else "KILL")
        ce, cd, co = (ms(seedvals_base(b1[k], "c")) for k in ("ea", "da", "oa"))
        holds = co["mean"] <= cd["mean"] <= ce["mean"]
        kill = (co["mean"] - cd["mean"] > 2 * (cd["se"] or 0)) or (co["mean"] - ce["mean"] > 2 * (ce["se"] or 0))
        P["P3"] = dict(c_ea=ce["mean"], c_da=cd["mean"], c_oa=co["mean"], ordering_holds=bool(holds),
                       verdict="GO" if holds else ("KILL" if kill else "neither (ordering fails within 2 SE)"))
        P4, P5, P6 = {}, {}, {}
        cm = ms(seedvals_base(b1["oa_myopic"], "c"))
        for nm, lr in (("ddpg", dd1), ("dql", dq1)):
            gm, vn = lr.vals(var, "g_mean").mean(), lr.vals(var, "viol_near_frac").mean()
            P4[nm] = dict(g_mean=gm, viol_near_frac=vn, verdict="GO" if (gm <= 0.01 and vn >= 0.5) else "KILL")
            lm = lr.vals(var, "lam_multi").mean()
            P5[nm] = dict(lam_multi=lm, verdict="GO" if 3e3 <= lm <= 1e5 else "KILL")
            c_l, sh = lr.vals(var, "c").mean(), lr.vals(var, "track_share").mean()
            go = (c_l >= cm["mean"] - cm["se"]) or (sh > 0.9)
            P6[nm] = dict(c=c_l, c_oa_myopic=cm["mean"], se_myopic=cm["se"], track_share=sh,
                          verdict="GO" if go else "KILL")
        P["P4"], P["P5"], P["P6"] = P4, P5, P6
        out["preds"][ck] = P

    # ------------------------------------------------------------------ ablation + debug variants (Case I, k_test)
    b1 = B.get(("case1", "A04"))
    for algo in ["ddpg", "dql"]:
        key = f"{algo}_case1_case1"
        if key not in L or b1 is None:
            continue
        lr = L[key]
        vs = sorted({x for x, _ in lr.lanes})
        ua, ud, uo = (seedvals_base(b1[k], "U") for k in ("ea", "da", "oa"))
        best_h = ud if ud.mean() >= ua.mean() else ua

        def summ(v):
            return dict(U=ms(lr.vals(v, "U")), sat=ms(lr.vals(v, "sat_rate")), g=ms(lr.vals(v, "g_mean")),
                        lam_max=ms(lr.vals(v, "lam_max")), share=ms(lr.vals(v, "track_share")),
                        tinit=ms(lr.tinit(v)), c=ms(lr.vals(v, "c")),
                        G=ms((lr.vals(v, "U") - best_h) / (uo - best_h)))
        ab = {}
        for v in ("fixlam", "nolam"):
            if v in vs:
                ab[v] = summ(v)
                ab[v]["sat_vs_A04"] = paired_compare(lr.vals(v, "sat_rate"), lr.vals("A04", "sat_rate"))
                ab[v]["sat_lower_p"] = one_sided(ab[v]["sat_vs_A04"], greater=False)
                ab[v]["U_vs_A04"] = paired_compare(lr.vals(v, "U"), lr.vals("A04", "U"))
        if "nolam" in ab:
            ab["rule"] = "SUPPORTS" if ab["nolam"]["sat_lower_p"] < 0.05 else "DOES NOT SUPPORT"
        ab["A04"] = summ("A04")
        out["ablation"][algo] = ab
        dbg = {}
        for v, ref in (("dual50", "A04"), ("raw", "A04"), ("A04a_dual50", "A04a"), ("A04a", "A04")):
            if v not in vs:
                continue
            bref = B.get(("case1", "A04a" if v.startswith("A04a") else "A04"))
            ua_, ud_, uo_ = (seedvals_base(bref[k], "U") for k in ("ea", "da", "oa"))
            bh = ud_ if ud_.mean() >= ua_.mean() else ua_
            pc = paired_compare(lr.vals(v, "U"), lr.vals(ref, "U"))
            G = (lr.vals(v, "U") - bh) / (uo_ - bh)
            p1 = one_sided(pc, True)
            verdict = ("CONFIRMED" if (p1 < 0.05 and G.mean() >= 0) else ("PARTIAL" if p1 < 0.05 else "REFUTED"))
            if v == "A04a":
                verdict = "n/a (different world: compare G, not U)"
            dbg[v] = dict(ref=ref, **summ(v), U_vs_ref=pc, p_one_sided=p1, G_world=ms(G), verdict=verdict)
        out["debug"][algo] = dbg
    # Holm across the declared debug tests (both learners; descriptive, the declared rule has no
    # correction, so its verdicts stand)
    pv = {f"{algo}:{v}": d["p_one_sided"] for algo, dd in out["debug"].items() for v, d in dd.items() if v != "A04a"}
    if pv:
        hh = holm(pv)
        out["debug_holm"] = {k: dict(p=pv[k], p_holm=hh[k]["p_holm"]) for k in pv}

    # ------------------------------------------------------------------ C8 qualitative patterns from traces
    from scipy.stats import spearmanr

    def c8_stats(z, case, z_slow=None):
        tau, act = z["tau"], z["act"] > 0.5                   # [N, T, K]
        multi = (act.sum(-1, keepdims=True) >= 2) & act
        vals = []
        for i in range(tau.shape[0]):
            if case == "case1":
                m = multi[i]
                vals.append(spearmanr(tau[i][m], z["r_true"][i][m])[0] if m.sum() > 10 else np.nan)
            elif case == "case2":
                m = multi[i]
                vals.append(spearmanr(tau[i][m], z["w"][i][m])[0] if m.sum() > 10 else np.nan)
            elif case == "case4":
                ins, outs_ = act[i] & (z["fac_true"][i] > 1), act[i] & (z["fac_true"][i] <= 1)
                vals.append(tau[i][ins].mean() - tau[i][outs_].mean() if ins.any() and outs_.any() else np.nan)
            elif case == "case3":
                f4 = act[i] & (z["tgt"][i] == 2)
                s4 = (z_slow["act"][i] > 0.5) & (z_slow["tgt"][i] == 2)
                vals.append(tau[i][f4].mean() - z_slow["tau"][i][s4].mean() if f4.any() and s4.any() else np.nan)
        v = np.asarray(vals, float)
        ok = v[np.isfinite(v)]
        ti = t_interval(ok) if len(ok) > 1 else [np.nan, np.nan]
        sign = "positive" if ti[0] > 0 else ("negative" if ti[1] < 0 else "undetermined")
        return dict(per_unit=v.round(4).tolist(), mean=float(ok.mean()) if len(ok) else None, t95=ti, sign=sign)

    c8 = {}
    for case, runs_ in (("case1", ["ddpg_case1", "dql_case1"]), ("case2", ["ddpg_case2", "dql_case2"]),
                        ("case4", ["ddpg_case1", "dql_case1"]), ("case3", ["ddpg_case3", "dql_case3"])):
        for run in runs_:
            fp = os.path.join(EV, f"traces_{run}_{'case3f' if case == 'case3' else case}.npz")
            if not os.path.exists(fp):
                continue
            z = dict(np.load(fp))
            zs = dict(np.load(os.path.join(EV, f"traces_{run}_case3s.npz"))) if case == "case3" else None
            lanes = [str(x).split("|") for x in z["lanes"]]
            for v in ("A04", "A04a"):
                idx = [i for i, (vv, _) in enumerate(lanes) if vv == v]
                if not idx:
                    continue
                sub = {k: z[k][idx] for k in z if k != "lanes"}
                subs = {k: zs[k][idx] for k in zs if k != "lanes"} if zs else None
                c8[f"{case}|{run}[{v}]"] = c8_stats(sub, case, subs)
        for kind in ("ea", "da", "oa", "oa_blind"):
            for ck in calibs:
                fp = os.path.join(EV, f"traces_base_{kind}_{'case3f' if case == 'case3' else case}_{ck}.npz")
                if not os.path.exists(fp):
                    continue
                z = dict(np.load(fp))
                zs = dict(np.load(fp.replace("case3f", "case3s"))) if case == "case3" else None
                c8[f"{case}|{kind}[{ck}]"] = c8_stats(z, case, zs)
    out["c5c7c8"]["C8"] = c8

    # ------------------------------------------------------------------ identity: sum_t g_t <= (lam_T - lam_0)/alpha
    d3 = {}
    for fp in sorted(glob.glob(os.path.join(EV, "traces_*_case*_case1.npz"))):
        z = dict(np.load(fp))
        run = os.path.basename(fp)[7:-10]
        alphas = {"dual50": 50.0, "A04a_dual50": 50.0}
        for i, ln in enumerate(z["lanes"]):
            v, sd = str(ln).split("|")
            if v in ("fixlam", "nolam"):
                continue
            al = alphas.get(v, 5000.0)
            g = z["tau"][i].astype(float).sum(-1) / 2.5 - 0.9
            lam = z["lam"][i].astype(float)
            lamT = max(0.0, lam[-1] + al * g[-1])
            bound = (lamT - 5000.0) / (al * len(g))
            d3[f"{run}[{v}]|{sd}"] = dict(mean_g=float(g.mean()), bound=float(bound), lam_T=lamT,
                                          holds=bool(g.mean() <= bound + 1e-3))
    out["d3_identity"] = dict(n=len(d3), n_holds=sum(x["holds"] for x in d3.values()), lanes=d3)

    # ------------------------------------------------------------------ A-18 plateau rule + training lambda trace
    cur = {}
    for cp in sorted(glob.glob(os.path.join(REPO, "runs", "full", "*_case?", "curve.json")) +
                     glob.glob(os.path.join(REPO, "runs", "recheck", "*_case?", "curve.json"))):
        run = os.path.basename(os.path.dirname(cp))
        if "/recheck/" in cp:
            run = "recheck_" + run
        cu = load(os.path.join(os.path.dirname(cp), "config_used.json"))
        lanes = [tuple(x) for x in cu["lanes"]]
        rows = load(cp)
        vrows = [r for r in rows if "val_U" in r]
        per = {}
        for i, (v, sd) in enumerate(lanes):
            vu = np.asarray([r["val_U"][i] for r in vrows], float)
            last = vu[-3:]
            m = last.mean()
            plateau = bool(np.all(np.isfinite(last)) and np.all(np.abs(last - m) <= 0.02 * abs(m)))
            lam_chunk = np.asarray([r["lam"][i] for r in rows], float)
            per.setdefault(v, []).append(dict(seed=sd, plateau=plateau, last3=last.tolist(),
                                              lam_chunkmean_max=float(lam_chunk.max()),
                                              lam_final_chunkmean=float(lam_chunk[-1]),
                                              viol_last=float(rows[-1]["viol"][i]),
                                              train_U_last=float(rows[-1]["U"][i])))
        cur[run] = {v: dict(n_plateau=sum(x["plateau"] for x in xs), n=len(xs),
                            windup_fixed=bool(all(x["lam_chunkmean_max"] < 1e6 for x in xs)), lanes=xs)
                    for v, xs in per.items()}
    out["curves"] = cur

    # ------------------------------------------------------------------ Fig. 9 sweep
    sw = os.path.join(EV, "sweep_fig9.json")
    if os.path.exists(sw):
        s = load(sw)
        f9 = {}
        for ck in calibs:
            for kind in ("ea", "da", "oa"):
                rows = s.get(f"{kind}_{ck}")
                if not rows:
                    continue
                for qn, key, scale in (("U", "U", 1e4), ("c", "c", 1e4)):
                    for set_ in ("spawn", "replica"):
                        ours = [r[set_][key] / scale for r in rows]
                        grades = [judge(o, p, prec=0.1 if qn == "U" else 0.2)[0] for o, p in zip(ours, FIG9[qn][kind])]
                        f9[f"{kind}_{ck}_{qn}_{set_}"] = dict(ours=ours, paper=FIG9[qn][kind], grades=grades)
                for set_ in ("spawn", "replica"):
                    ours = [r[f"{set_}_tinit"] for r in rows]
                    grades = [judge(o, p, prec=0.5)[0] for o, p in zip(ours, FIG9["tinit"][kind])]
                    f9[f"{kind}_{ck}_tinit_{set_}"] = dict(ours=ours, paper=FIG9["tinit"][kind], grades=grades)
        out["fig9"] = f9

    with open(os.path.join(REPO, "runs", TAG, "report.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote report.json")

    # ------------------------------------------------------------------ per-seed metrics.json
    import hashlib
    cfgd = os.path.join(REPO, "src", "configs")
    sha = {n: hashlib.sha256(open(os.path.join(cfgd, n), "rb").read()).hexdigest() for n in sorted(os.listdir(cfgd))
           if n.endswith((".yaml", ".json"))}
    nonfinite = {nm: int(sum((~np.isfinite(np.asarray(lr.m[sl]["U"], float))).sum() for sl in ("replica", "spawn")
                             if sl in lr.m)) for nm, lr in L.items()}
    nonfinite.update({f"base_{c}_{k}": int(sum((~np.isfinite(np.asarray(m[sl]["U"], float))).sum()
                                               for m in b.values() for sl in m)) for (c, k), b in B.items()})
    out["meta"]["nonfinite_U_counts"] = nonfinite
    print("non-finite U values:", sum(nonfinite.values()))
    with open(os.path.join(REPO, "runs", TAG, "report.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    if TAG != "full_v2":
        return                                                 # per-seed files carry the CURRENT (re-evaluation) numbers
    for sd in range(NP):
        rec = dict(stage="full_v2 (the re-evaluation: re-digitised replica, v2 slot keys, OA L_bm "
                         "knowledge)", git_commit=commit, keys="k_test (final evaluation, full-run)", seed=sd, sha256=sha,
                   scenario_note="case4 reuses scenario_case1.json (A-39a); case3 = scenario_case3.json; full-run "
                                 "numbers used scenario_case1_v1.json (runs/full/report.json)",
                   unit="seed-level value = mean over the seed's 20 replica draws (pairing index = seed)",
                   learners={}, baselines={})
        for nm, lr in L.items():
            for i, (v, s_) in enumerate(lr.lanes):
                if s_ != sd:
                    continue
                rec["learners"][f"{nm}[{v}]"] = {k: dict(mean=float(np.mean(lr.m["replica"][k][i])),
                                                         per_draw=np.asarray(lr.m["replica"][k][i]).round(4).tolist())
                                                 for k in ("U", "c", "scan_term", "sat_rate", "g_mean", "track_share",
                                                           "lam_multi", "lam_max", "tinit_sum", "tinit_n", "never")}
                if "spawn" in lr.m:
                    rec["learners"][f"{nm}[{v}]"]["spawn_U_mean"] = float(np.mean(lr.m["spawn"]["U"][i]))
        for (case, ck), b in B.items():
            for kind, m in b.items():
                rec["baselines"][f"{kind}|{case}|{ck}"] = {
                    k: dict(mean=float(np.mean(np.asarray(m["replica"][k])[NR * sd:NR * sd + NR])),
                            per_draw=np.asarray(m["replica"][k])[NR * sd:NR * sd + NR].round(4).tolist())
                    for k in ("U", "c", "scan_term", "sat_rate", "tinit_sum", "tinit_n")}
        d = os.path.join(REPO, "runs", f"seed_{sd}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(rec, f, indent=1)
    print("wrote runs/seed_*/metrics.json")


if __name__ == "__main__":
    main()
