"""sensitivity report: tables and paired tests for the pre-declared sensitivities (see README).
Pure post-processing (no jax).

    uv run python scripts/sensitivity_report.py  -> runs/sensitivity/report.json + runs/sensitivity/report_tables.txt
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402
from report import E5_P, E8_P, E2_P, rel_judge, judge  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "runs", *p)  # noqa: E731
J = lambda *p: json.load(open(R(*p)))  # noqa: E731
LAYOUTS = ("A", "W1", "W0", "OUT", "RUN")
ORDERS = ("official", "alg")
L = []  # output lines


def out(s=""):
    L.append(s)


def a(x):
    return np.asarray(x, float)


def cmp_row(x, y):
    c = paired_compare(x, y)
    return c


def fp(p):
    return f"{p:.2g}"


def main():
    s0 = J("seed_0", "metrics.json")["experiments"]
    div = J("full", "divergence.json")
    abl = J("full", "ablation.json")
    res = {}

    def prim(exp, si, arm, stat, src=None):
        return a((src or s0)[exp][si]["arms"][arm][stat])

    def div_arm(exp, si, method, stat):
        rows = [r for r in div["rows"] if r["tag"] == "DA_alg_order" and r["exp"] == exp and r["method"] == method]
        return a(rows[si][stat])

    # ------------------------------------------------------------------ S-BIN
    have_bins = os.path.exists(R("sensitivity", "bins.json"))
    if have_bins:
        bins = J("sensitivity", "bins.json")["cells"]

        def cell(exp, si, lay, order, stat):
            for c in bins:
                if c["exp"] == exp and c["si"] == si and c["layout"] == lay and c["order"] == order:
                    return a(c[stat])
            raise KeyError((exp, si, lay, order))

        def state_arm(exp, si, order, stat):
            return prim(exp, si, "state_hca", stat) if order == "official" else div_arm(exp, si, "state_hca", stat)

        claims = [("P4", "E3", 0, "final", "R-PG"), ("P8", "E6", 0, "final", "R-PG"), ("P9", "E7", 0, "final", "R-PG"),
                  ("P10", "E8", 0, "avg", "R-PG"), ("sec E3 Return−State", "E3", 0, "final", "R-S"),
                  ("sec E4 ep100 PG−Return", "E4", 0, "ep100", "PG-R"), ("sec E6 State−Return", "E6", 0, "final", "S-R")]
        claims += [(f"sec E5 σ={si} Return−PG", "E5", si, "avg", "R-PG") for si in range(6)]
        claims += [(f"sec E8 ε=0.{si} Return−PG", "E8", si, "avg", "R-PG") for si in range(1, 5)]
        sbin = {}
        out("## S-BIN: Return-HCA ordering claims across 5 bin layouts × 2 update orders (diff mean; Holm over the 10 cells)")
        out("| claim | " + " | ".join(f"{l}/{o[:3]}" for l in LAYOUTS for o in ORDERS) + " | primary sign | all cells Holm<0.05 same sign? | reading |")
        out("|---|" + "---|" * (len(LAYOUTS) * len(ORDERS)) + "---|---|---|")
        for name, exp, si, stat, kind in claims:
            diffs, ps = {}, {}
            for lay in LAYOUTS:
                for order in ORDERS:
                    r = cell(exp, si, lay, order, stat)
                    pg = prim(exp, si, "pg", stat)
                    if kind == "R-PG":
                        x, y = r, pg
                    elif kind == "PG-R":
                        x, y = pg, r
                    elif kind == "R-S":
                        x, y = r, state_arm(exp, si, order, stat)
                    else:
                        x, y = state_arm(exp, si, order, stat), r
                    c = paired_compare(x, y)
                    diffs[(lay, order)] = c
                    ps[f"{lay}/{order}"] = c["t_p"]
            hp = holm(ps)
            prim_sign = np.sign(diffs[("A", "official")]["diff_mean"])
            sig_same = all(hp[f"{l}/{o}"]["p_holm"] < 0.05 and np.sign(diffs[(l, o)]["diff_mean"]) == prim_sign
                           for l in LAYOUTS for o in ORDERS)
            any_opp = any(hp[f"{l}/{o}"]["p_holm"] < 0.05 and np.sign(diffs[(l, o)]["diff_mean"]) == -prim_sign
                          for l in LAYOUTS for o in ORDERS)
            reading = "ROBUST" if sig_same else ("LAYOUT-DEPENDENT (sign flips, sig.)" if any_opp else "LAYOUT-DEPENDENT (n.s. in some cells)")
            cells_txt = []
            for l in LAYOUTS:
                for o in ORDERS:
                    c = diffs[(l, o)]
                    star = "*" if hp[f"{l}/{o}"]["p_holm"] < 0.05 else ""
                    cells_txt.append(f"{c['diff_mean']:+.3f}{star}")
            out(f"| {name} | " + " | ".join(cells_txt) + f" | {'+' if prim_sign > 0 else '−'} | {'yes' if sig_same else 'no'} | {reading} |")
            sbin[name] = {"cells": {f"{l}/{o}": {"diff": diffs[(l, o)]["diff_mean"], "se": diffs[(l, o)]["diff_se"],
                                                 "t_p": diffs[(l, o)]["t_p"], "p_holm": hp[f"{l}/{o}"]["p_holm"],
                                                 "wilcoxon_p": diffs[(l, o)]["wilcoxon_p"]}
                                    for l in LAYOUTS for o in ORDERS}, "reading": reading}
        out("(* = Holm-adjusted p < 0.05 within the claim. Columns: layout/order; off = official pre-update order, alg = Alg. 1/2 order.)")
        out()
        out("### S-BIN: Return-HCA level per layout (mean over 100 runs; official order / alg order)")
        out("| exp | setting | stat | paper | " + " | ".join(LAYOUTS) + " |")
        out("|---|---|---|---|" + "---|" * len(LAYOUTS))
        lev = [("E3", 0, "final", 0.83), ("E4", 0, "ep100", 0.46), ("E4", 0, "final", 0.80)]
        lev += [("E5", si, "avg", E5_P[si][1]) for si in range(6)]
        lev += [("E6", 0, "final", 1.77), ("E7", 0, "final", 1.765)]
        lev += [("E8", si, "avg", E8_P[si][1]) for si in range(5)]
        levels = []
        for exp, si, stat, pv in lev:
            vals = [f"{cell(exp, si, l, 'official', stat).mean():.3f} / {cell(exp, si, l, 'alg', stat).mean():.3f}" for l in LAYOUTS]
            sett = {"E5": f"σ={si}", "E8": f"ε=0.{si}"}.get(exp, "")
            out(f"| {exp} | {sett} | {stat} | {pv} | " + " | ".join(vals) + " |")
            levels.append({"exp": exp, "si": si, "stat": stat, "paper": pv,
                           "means": {f"{l}/{o}": float(cell(exp, si, l, o, stat).mean()) for l in LAYOUTS for o in ORDERS}})
        res["S_BIN"] = {"claims": sbin, "levels": levels}
        out()
    # A-BINS middle-bin share (bin-layout evidence), analytic: DE n = 3, returns +-1 + N(0, 3 sigma^2), centres +-(1 + 2 sqrt3 sigma)
    out("### Bin-layout check: share of Delayed-Effect (n=3) x0 returns that land in the middle A-BINS bin (analytic)")
    out("| σ | centres | P(middle | + branch) = P(middle | − branch) |")
    out("|---|---|---|")
    mid = {}
    for sig in (1.0, 2.0, 3.0, 4.0, 5.0):
        sd = np.sqrt(3) * sig
        c = 1 + 2 * sd
        half = c / 2
        pm = st.norm.cdf((half - 1) / sd) - st.norm.cdf((-half - 1) / sd)
        mid[sig] = float(pm)
        out(f"| {sig:.0f} | [−{c:.2f}, 0, {c:.2f}] | {pm:.3f} |")
    res["F1_middle_bin_share"] = mid
    out()

    # ------------------------------------------------------------------ S-ORDER / S-PG03 / S-PAPERVAR
    tl = J("seed_0", "metrics.json")["pg_tuned_lr"]
    fam = [("P1", "E1", 0, "state_hca", "final"), ("P3", "E3", 0, "state_hca", "final"),
           ("P4", "E3", 0, "return_hca", "final"), ("P5", "E4", 0, "state_hca", "avg"), ("P6", "E5", 5, "state_hca", "avg"),
           ("P7", "E6", 0, "state_hca", "final"), ("P8", "E6", 0, "return_hca", "final"),
           ("P9", "E7", 0, "return_hca", "final"), ("P10", "E8", 0, "return_hca", "avg")]
    e2 = s0["E2"][6]
    orc = e2["oracle"]
    pg_e2 = np.abs(a(e2["arms"]["pg"]["online"]) - orc)

    def fam_table(title, getter, extra_p2=None, key=""):
        out(title)
        out("| # | diff mean ± SE | wins | t p | Holm p | Wilcoxon p | verdict | primary verdict |")
        out("|---|---|---|---|---|---|---|---|")
        cs, ps = {}, {}
        if extra_p2 is not None:
            cs["P2"] = paired_compare(*extra_p2)
            ps["P2"] = cs["P2"]["t_p"]
        for n, exp, si, arm, stat in fam:
            x, y = getter(n, exp, si, arm, stat)
            cs[n] = paired_compare(x, y)
            ps[n] = cs[n]["t_p"]
        hp = holm(ps)
        prim_v = {k: v["verdict"] for k, v in J("full", "report.json")["primary"].items()}
        prim_v = {k.split()[0]: v for k, v in prim_v.items()}
        rows = {}
        for n in sorted(cs, key=lambda k: int(k[1:])):
            c = cs[n]
            v = ("REPRODUCED" if hp[n]["p_holm"] < 0.05 and c["diff_mean"] > 0 else
                 "REVERSED" if hp[n]["p_holm"] < 0.05 else "n.s.")
            out(f"| {n} | {c['diff_mean']:+.4f} ± {c['diff_se']:.4f} | {c['wins']} | {fp(c['t_p'])} | {fp(hp[n]['p_holm'])} | "
                f"{fp(c['wilcoxon_p'])} | {v} | {prim_v[n].split(' (')[0]} |")
            rows[n] = {"diff": c["diff_mean"], "se": c["diff_se"], "t_p": c["t_p"], "p_holm": hp[n]["p_holm"],
                       "wilcoxon_p": c["wilcoxon_p"], "wins": c["wins"], "verdict": v}
        out()
        res[key] = rows

    # S-ORDER: HCA arms in the Alg. 1/2 order; P2 with the estimate taken after the per-rollout update
    e2div = div["e2"][6]
    fam_table("## S-ORDER: primary family with the HCA arms in the literal Alg. 1/2 order (Holm over 10)",
              lambda n, exp, si, arm, stat: (div_arm(exp, si, arm, stat), prim(exp, si, "pg", stat)),
              extra_p2=(pg_e2, np.abs(a(e2div["state_hca_after_update"]) - orc)), key="S_ORDER")
    fam_table("## S-PG03: primary family with PG fixed at lr 0.3 in every panel (second reading of App. E; Holm over 10)",
              lambda n, exp, si, arm, stat: (prim(exp, si, arm, stat), prim(exp, si, "pg_lr0.3", stat)),
              extra_p2=(pg_e2, np.abs(a(e2["arms"]["state_hca"]["online"]) - orc)), key="S_PG03")

    # S-PAPERVAR
    out("## S-PAPERVAR: Alg.-1 (r̂) State-HCA variant vs tuned PG (AB3 runs; Holm over 4)")
    out("| # | State(r̂) mean | official-snippet State mean | diff vs PG ± SE | t p | Holm p | Wilcoxon p | verdict |")
    out("|---|---|---|---|---|---|---|---|")
    pv_cs, pv_ps = {}, {}
    for n, exp, si, stat in (("P1", "E1", 0, "final"), ("P3", "E3", 0, "final"), ("P5", "E4", 0, "avg"), ("P7", "E6", 0, "final")):
        row = [r for r in abl["rows"] if r["tag"] == "AB3_state_variant_paper" and r["exp"] == exp][0]
        x = a(row[stat])
        pv_cs[n] = (paired_compare(x, prim(exp, si, "pg", stat)), x.mean(), prim(exp, si, "state_hca", stat).mean())
        pv_ps[n] = pv_cs[n][0]["t_p"]
    hp = holm(pv_ps)
    res["S_PAPERVAR"] = {}
    for n, (c, m1, m0) in pv_cs.items():
        v = "REPRODUCED" if hp[n]["p_holm"] < 0.05 and c["diff_mean"] > 0 else "REVERSED" if hp[n]["p_holm"] < 0.05 else "n.s."
        out(f"| {n} | {m1:.4f} | {m0:.4f} | {c['diff_mean']:+.4f} ± {c['diff_se']:.4f} | {fp(c['t_p'])} | {fp(hp[n]['p_holm'])} | {fp(c['wilcoxon_p'])} | {v} |")
        res["S_PAPERVAR"][n] = {"diff": c["diff_mean"], "se": c["diff_se"], "t_p": c["t_p"], "p_holm": hp[n]["p_holm"], "verdict": v}
    out()

    # ------------------------------------------------------------------ S-EPS0
    if os.path.exists(R("sensitivity", "eps0.json")):
        e0 = J("sensitivity", "eps0.json")["rows"]
        out("## S-EPS0: E6/E7 at ε = 0 (PG lr re-tuned on seed 999 at ε = 0; Holm over 3)")
        out("| claim | State | Return | PG (lr) | diff ± SE | t p | Holm p | Wilcoxon p | verdict | primary (ε = 0.1) |")
        out("|---|---|---|---|---|---|---|---|---|---|")
        cs, ps = {}, {}
        for n, exp, arm in (("P7", "E6", "state_hca"), ("P8", "E6", "return_hca"), ("P9", "E7", "return_hca")):
            r = [x for x in e0 if x["exp"] == exp][0]
            cs[n] = (paired_compare(a(r["arms"][arm]["final"]), a(r["arms"]["pg"]["final"])), r)
            ps[n] = cs[n][0]["t_p"]
        hp = holm(ps)
        res["S_EPS0"] = {}
        for n, (c, r) in cs.items():
            v = "REPRODUCED" if hp[n]["p_holm"] < 0.05 and c["diff_mean"] > 0 else "REVERSED" if hp[n]["p_holm"] < 0.05 else "n.s."
            out(f"| {n} | {a(r['arms']['state_hca']['final']).mean():.3f} | {a(r['arms']['return_hca']['final']).mean():.3f} | "
                f"{a(r['arms']['pg']['final']).mean():.3f} ({r['pg_tuned_lr']}) | {c['diff_mean']:+.4f} ± {c['diff_se']:.4f} | "
                f"{fp(c['t_p'])} | {fp(hp[n]['p_holm'])} | {fp(c['wilcoxon_p'])} | {v} | REVERSED |")
            res["S_EPS0"][n] = {"diff": c["diff_mean"], "se": c["diff_se"], "t_p": c["t_p"], "p_holm": hp[n]["p_holm"], "verdict": v,
                                "pg_lr": r["pg_tuned_lr"]}
        out("Paper (Fig. 5 left/middle, ε unstated): State 1.845 / 1.495, Return 1.77 / 1.765, PG 1.73 / 1.715.")
        out()

    # ------------------------------------------------------------------ S-SLIDE
    if os.path.exists(R("sensitivity", "slide.json")):
        sl = J("sensitivity", "slide.json")
        out("## S-SLIDE: E3 with sliding-window 3-step returns (PG lr re-tuned on seed 999 under sliding)")
        same_state = float(np.max(np.abs(a(sl["arms"]["state_hca_official"]["final"]) - prim("E3", 0, "state_hca", "final"))))
        same_pg = {lr: float(np.max(np.abs(a(sl["arms"][f"pg_lr{lr}"]["final"]) - prim("E3", 0, f"pg_lr{lr}", "final"))))
                   for lr in (0.1, 0.2, 0.3, 0.4)}
        same_state_avg = float(np.max(np.abs(a(sl["arms"]["state_hca_official"]["avg"]) - prim("E3", 0, "state_hca", "avg"))))
        same_alg = float(np.max(np.abs(a(sl["arms"]["state_hca_alg"]["final"]) - div_arm("E3", 0, "state_hca", "final"))))
        out(f"max |sliding − segment| per-run final: State(off) {same_state:.3g} (avg {same_state_avg:.3g}), State(alg) {same_alg:.3g}, "
            f"PG " + ", ".join(f"lr{k} {v:.3g}" for k, v in same_pg.items()) + f"; PG tuned lr under sliding = {sl['pg_tuned_lr']}")
        cs = {"P3": paired_compare(a(sl["arms"]["state_hca_official"]["final"]), a(sl["arms"]["pg"]["final"])),
              "P4": paired_compare(prim("E3", 0, "return_hca", "final"), a(sl["arms"]["pg"]["final"]))}
        hp = holm({k: v["t_p"] for k, v in cs.items()})
        res["S_SLIDE"] = {"max_abs_diff_vs_segment": {"state_off": same_state, "state_alg": same_alg, "pg": same_pg},
                          "pg_tuned_lr": sl["pg_tuned_lr"]}
        for n, c in cs.items():
            out(f"{n} (sliding): {c['diff_mean']:+.4f} ± {c['diff_se']:.4f}, Holm p {fp(hp[n]['p_holm'])}, wins {c['wins']}")
            res["S_SLIDE"][n] = {"diff": c["diff_mean"], "se": c["diff_se"], "p_holm": hp[n]["p_holm"]}
        out()

    # ------------------------------------------------------------------ S-F3R-V
    if os.path.exists(R("sensitivity", "f3rv.json")):
        fv = J("sensitivity", "f3rv.json")["points"]
        out("## S-F3R-V: E2 PG bars with an online-learned V(x0) baseline vs the batch-mean baseline")
        out("| p_long | paper PG | batch-mean (primary) | online V lr 0.3 | judg. norm./rel. (online 0.3) | online V lr 0.1 | 2p(1−p)·oracle |")
        out("|---|---|---|---|---|---|---|")
        rows = []
        for i, pt in enumerate(fv):
            p = pt["p_long"]
            pv = E2_P["pg"][i]
            m0 = a(s0["E2"][i]["arms"]["pg"]["online"]).mean()
            m3, se3 = mean_se(pt["pg_online_v_lr0.3"])
            m1 = a(pt["pg_online_v_lr0.1"]).mean()
            jn, _ = judge(m3, pv, pt["oracle"], 0.03, 0.02)
            jr = rel_judge(m3, pv)
            out(f"| {p:.2f} | {pv:.2f} | {m0:.3f} | {m3:.3f} ± {se3:.3f} | {jn} / {jr} | {m1:.3f} | {2 * p * (1 - p) * pt['oracle']:.3f} |")
            rows.append({"p_long": p, "paper": pv, "batch": m0, "online_v_0.3": m3, "online_v_0.1": m1, "judg_norm": jn, "judg_rel": jr})
        c = paired_compare(np.abs(a(fv[6]["pg_online_v_lr0.3"]) - fv[6]["oracle"]), np.abs(a(s0["E2"][6]["arms"]["state_hca"]["online"]) - fv[6]["oracle"]))
        out(f"P2 analogue (online-V PG): {c['diff_mean']:+.4f} ± {c['diff_se']:.4f}, p {fp(c['t_p'])}, wins {c['wins']}")
        res["S_F3R_V"] = {"rows": rows, "P2": {"diff": c["diff_mean"], "se": c["diff_se"], "t_p": c["t_p"]}}
        out()

    # ------------------------------------------------------------------ S-AVG
    out("## S-AVG: E5/E8 rows under the curve-mean (A-AVG, primary) and final-value readings")
    out("| exp | setting | arm | paper | ours avg ± SE | judg. norm./rel. (avg) | ours final ± SE | judg. norm./rel. (final) |")
    out("|---|---|---|---|---|---|---|---|")
    savg = []
    for exp, P_, Rf in (("E5", E5_P, lambda si: 1.0), ("E8", E8_P, lambda si: 0.5 - 0.1 * si)):
        for si in range(len(P_)):
            for k, arm in enumerate(("state_hca", "return_hca", "pg")):
                pv = P_[si][k]
                ma, sa = mean_se(prim(exp, si, arm, "avg"))
                mf, sf = mean_se(prim(exp, si, arm, "final"))
                ja = judge(ma, pv, Rf(si), 0.03 if exp == "E5" else 0.01, 0.0)[0] + " / " + rel_judge(ma, pv)
                jf = judge(mf, pv, Rf(si), 0.03 if exp == "E5" else 0.01, 0.0)[0] + " / " + rel_judge(mf, pv)
                sett = f"σ={si}" if exp == "E5" else f"ε=0.{si}"
                out(f"| {exp} | {sett} | {arm} | {pv} | {ma:.3f} ± {sa:.3f} | {ja} | {mf:.3f} ± {sf:.3f} | {jf} |")
                savg.append({"exp": exp, "si": si, "arm": arm, "paper": pv, "avg": ma, "final": mf, "judg_avg": ja, "judg_final": jf})
    out("(The norm. judgments here omit the paper-spread clause, i.e. std_paper = 0, so both readings are judged alike.)")
    res["S_AVG"] = savg
    if os.path.exists(R("sensitivity", "alpha.json")):
        al = J("sensitivity", "alpha.json")["rows"]
        out()
        out("D-ALPHA (oracle continuous h_z, E8 ε=0): " + "; ".join(
            f"α={r['alpha']}: avg {a(r['avg']).mean():.4f}, final {a(r['final']).mean():.4f}" for r in al))
        oh = [r for r in J("full", "oracle_hz.json")["rows"] if r["setting"]["eps"] == 0.0 and r["setting"].get("hidden")][0]
        out(f"(consistency: α=0.3 here vs runs/full/oracle_hz.json avg max|Δ| = {np.max(np.abs(a(al[0]['avg']) - a(oh['avg']))):.3g})")
        res["D_ALPHA"] = [{"alpha": r["alpha"], "avg": float(a(r["avg"]).mean()), "final": float(a(r["final"]).mean())} for r in al]
    out()

    # ------------------------------------------------------------------ S-E1
    out("## S-E1: E1 scored by the fraction of the start-to-optimum gap closed, q = (V − V0)/(−0.1 − V0)")
    out("| arm | ep | paper V (q, V0 = −1.80) | ours V (q ± SE, V0 = −0.963) | Δ% of q | judgment (relative, on q) |")
    out("|---|---|---|---|---|---|")
    paper_e1 = {"state_hca": {20: -0.62, 50: -0.14, 100: -0.10}, "return_hca": {20: -0.83, 50: -0.50, 100: -0.43},
                "pg": {20: -0.87, 50: -0.37, 100: -0.18}}
    v0o = float(prim("E1", 0, "state_hca", "mean_curve")[0])
    se1 = []
    for arm in ("state_hca", "return_hca", "pg"):
        for ep in (20, 50, 100):
            pv = paper_e1[arm][ep]
            qp = (pv + 1.80) / (1.80 - 0.1)
            x = prim("E1", 0, arm, "final" if ep == 100 else f"ep{ep}")
            q = (x - v0o) / (-0.1 - v0o)
            mq, sq = mean_se(q)
            jr = rel_judge(mq, qp)
            out(f"| {arm} | {ep} | {pv:+.2f} ({qp:.3f}) | {x.mean():+.3f} ({mq:.3f} ± {sq:.3f}) | {100 * (mq - qp) / qp:+.0f}% | {jr} |")
            se1.append({"arm": arm, "ep": ep, "paper_V": pv, "paper_q": qp, "ours_V": float(x.mean()), "ours_q": mq, "ours_q_se": sq, "judg_rel": jr})
    out(f"(V0 ours = {v0o:.4f}; ordering at ep100 in V and in q is identical because V0 is common to the arms.)")
    res["S_E1"] = se1
    out()

    # ------------------------------------------------------------------ bimodality check, oracle rows
    x = prim("E4", 0, "state_hca", "avg")
    y = prim("E4", 0, "pg", "avg")
    d = x - y
    out("## Bimodality check: E4 State − PG curve-mean (P5) distribution")
    out(f"State avg: median {np.median(x):.3f}, runs < 0.5: {(x < 0.5).sum()}, < 0: {(x < 0).sum()}; PG avg median {np.median(y):.3f}, "
        f"runs < 0.5: {(y < 0.5).sum()}; diff median {np.median(d):+.4f}, mean {d.mean():+.4f}; "
        f"mean of diff over runs with State avg ≥ 0.5: {d[x >= 0.5].mean():+.4f} (n={int((x >= 0.5).sum())})")
    res["RT4"] = {"state_median": float(np.median(x)), "state_lt_0.5": int((x < 0.5).sum()), "pg_lt_0.5": int((y < 0.5).sum()),
                  "diff_median": float(np.median(d)), "diff_mean": float(d.mean())}
    out()
    oh = J("full", "oracle_hz.json")["rows"]
    out("## Oracle-h_z Return HCA − tuned PG (same seed-0 run keys)")
    out("| exp | setting | stat | oracle mean | PG mean | diff ± SE | t p | Wilcoxon p | wins |")
    out("|---|---|---|---|---|---|---|---|---|")
    rt1 = []
    for r in oh:
        exp = "E8" if r["setting"].get("sigma") == 0.5 else ("E7" if r["setting"].get("hidden") else "E6")
        si = 0 if exp != "E8" else int(round(r["setting"]["eps"] * 10))
        stat = "avg" if exp == "E8" else "final"
        c = paired_compare(a(r[stat]), prim(exp, si, "pg", stat))
        out(f"| {exp} | ε={r['setting']['eps']} | {stat} | {a(r[stat]).mean():.4f} | {prim(exp, si, 'pg', stat).mean():.4f} | "
            f"{c['diff_mean']:+.4f} ± {c['diff_se']:.4f} | {fp(c['t_p'])} | {fp(c['wilcoxon_p'])} | {c['wins']} |")
        rt1.append({"exp": exp, "eps": r["setting"]["eps"], "stat": stat, "diff": c["diff_mean"], "se": c["diff_se"], "t_p": c["t_p"],
                    "wilcoxon_p": c["wilcoxon_p"], "wins": c["wins"]})
    res["RT1"] = rt1
    with open(R("sensitivity", "report.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    open(R("sensitivity", "report_tables.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
