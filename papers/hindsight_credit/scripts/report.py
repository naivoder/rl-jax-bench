"""full-scale report: paper / official snippet / ours tables, judgments, the pre-declared primary test family (Holm),
secondary orderings, replication seeds, ablations, theory and divergence probes. Pure post-processing (no jax).

    uv run python scripts/report.py  -> runs/full/report.json + runs/full/report_tables.txt
    uv run python scripts/report.py --pre-recheck  -> runs/full/report_prerecheck.json (+ _tables): the
        pre-recheck paper read-offs (before the three figure re-reads below)

Judgment rule (declared before any run): normalised error e = |m - p| / R with R = optimum - start of
the task (E2: the oracle at that p_long). MATCH iff e <= 5 % or |m - p| <= delta + 2 * std_paper / sqrt(100);
CLOSE iff e <= 20 %; else DIVERGENT. delta = read-off precision, std_paper = width of the paper's shaded band read off
the figure (coarse, +-50 %; E2: half-width of the black 95 % bar / 1.96 * 10).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "runs", *p)  # noqa: E731
ARMS = ("state_hca", "return_hca", "pg")
SHORT = {"state_hca": "State", "return_hca": "Return", "pg": "PG"}

# (exp, setting index, stat, R, delta, {arm: (paper value, paper std read from band)})
P = [
    ("E1", 0, "ep20", 0.8631, 0.05, {"state_hca": (-0.62, 0.10), "return_hca": (-0.83, 0.10), "pg": (-0.87, 0.10)}),
    ("E1", 0, "ep50", 0.8631, 0.05, {"state_hca": (-0.14, 0.05), "return_hca": (-0.50, 0.07), "pg": (-0.37, 0.07)}),
    ("E1", 0, "final", 0.8631, 0.03, {"state_hca": (-0.10, 0.02), "return_hca": (-0.43, 0.07), "pg": (-0.18, 0.05)}),
    ("E3", 0, "ep100", 1.0, 0.03, {"state_hca": (0.27, 0.12), "return_hca": (0.28, 0.25), "pg": (0.08, 0.30)}),
    ("E3", 0, "final", 1.0, 0.03, {"state_hca": (0.68, 0.10), "return_hca": (0.83, 0.09), "pg": (0.05, 0.58)}),
    ("E4", 0, "ep50", 1.0, 0.03, {"state_hca": (0.62, 0.15), "return_hca": (0.33, 0.25), "pg": (0.40, 0.30)}),
    ("E4", 0, "ep100", 1.0, 0.03, {"state_hca": (0.72, 0.12), "return_hca": (0.46, 0.25), "pg": (0.53, 0.30)}),
    ("E4", 0, "final", 1.0, 0.03, {"state_hca": (0.84, 0.055), "return_hca": (0.80, 0.20), "pg": (0.80, 0.28)}),
]
E5_P = {0: (0.79, 0.80, 0.84), 1: (0.79, 0.80, 0.83), 2: (0.78, 0.72, 0.67), 3: (0.77, 0.56, 0.56), 4: (0.71, 0.54, 0.47),
        5: (0.66, 0.47, 0.25)}
E5_STD = {0: 0.03, 1: 0.08, 2: 0.30, 3: 0.50, 4: 0.55, 5: 0.60}
for si, v in E5_P.items():
    P.append(("E5", si, "avg", 1.0, 0.03, {a: (x, E5_STD[si]) for a, x in zip(ARMS, v)}))
P += [
    ("E6", 0, "final", 0.4, 0.01, {"state_hca": (1.845, 0.25), "return_hca": (1.77, 0.20), "pg": (1.73, 0.13)}),
    ("E7", 0, "final", 0.4, 0.01, {"state_hca": (1.495, 0.12), "return_hca": (1.765, 0.22), "pg": (1.715, 0.13)}),
]
E8_P = {0: (1.58, 1.97, 1.85), 1: (1.56, 1.81, 1.76), 2: (1.535, 1.68, 1.66), 3: (1.50, 1.59, 1.58), 4: (1.515, 1.52, 1.52)}
E8_STD = {"state_hca": 0.25, "return_hca": 0.10, "pg": 0.05}
for si, v in E8_P.items():
    P.append(("E8", si, "avg", 0.5 - 0.1 * si, 0.01, {a: (x, E8_STD[a]) for a, x in zip(ARMS, v)}))
E2_P = {"state_hca": [0.45, 0.62, 0.84, 1.05, 1.56, 1.96, 2.75], "return_hca": [0.21, 0.41, 0.65, 0.89, 1.42, 1.90, 2.78],
        "pg": [0.17, 0.42, 0.69, 0.83, 0.98, 0.82, 0.55]}
E2_ORACLE_P = [1.07, 1.24, 1.45, 1.72, 2.04, 2.46, 2.96]
E2_SE_PAPER = 0.02  # black bars ~ +-0.04 (95 % CI) -> SE ~ 0.02
E2_SE = None  # recheck: per-arm, per-bar SE_paper (None -> the pre-recheck constant E2_SE_PAPER)

# ---------------------------------------------------------------- recheck (2026-10-11) paper re-reads
# Fig. 5 is vector graphics in the PDF; exact means and +-1 std read from the drawing
# (data/fig5_vector.json). Two raster read-offs were re-digitized by hand.
# Fig. 3-right bars are a 95 % CI over the 1000 rollouts of one
# repetition; per-bar half-widths by full line extent (data/fig3r_eb_full.json).
AUDIT_E67 = {"E6": {"state_hca": (1.8458, 0.198), "return_hca": (1.7725, 0.268), "pg": (1.7336, 0.133)},
             "E7": {"state_hca": (1.4945, 0.296), "return_hca": (1.7648, 0.257), "pg": (1.7193, 0.127)}}
AUDIT_E8 = {"state_hca": ([1.5803, 1.5585, 1.5360, 1.4985, 1.5151], [0.354, 0.307, 0.224, 0.148, 0.075]),
            "return_hca": ([1.9715, 1.8142, 1.6805, 1.5878, 1.5234], [0.154, 0.220, 0.217, 0.160, 0.090]),
            "pg": ([1.8529, 1.7591, 1.6577, 1.5774, 1.5213], [0.038, 0.032, 0.040, 0.033, 0.020])}
AUDIT_READS = {("E4", "ep50", "return_hca"): 0.271, ("E1", "ep50", "pg"): -0.338}


def apply_recheck_reads():
    """Replace the pre-recheck paper read-offs in place (``--pre-recheck`` skips this and reproduces the old report)."""
    global E8_P, E2_SE
    for k, (exp, si, stat, Rng, delta, arms) in enumerate(P):
        arms = dict(arms)
        if exp in AUDIT_E67:
            arms = dict(AUDIT_E67[exp])
        elif exp == "E8":
            arms = {a: (AUDIT_E8[a][0][si], AUDIT_E8[a][1][si]) for a in ARMS}
        for a in ARMS:
            if (exp, stat, a) in AUDIT_READS:
                arms[a] = (AUDIT_READS[(exp, stat, a)], arms[a][1])
        P[k] = (exp, si, stat, Rng, delta, arms)
    E8_P = {si: tuple(AUDIT_E8[a][0][si] for a in ARMS) for si in range(5)}
    eb = json.load(open(os.path.join(ROOT, "data", "fig3r_eb_full.json")))
    E2_SE = {arm: [b["half_minus_1px"] / 1.96 for b in eb[key]]
             for arm, key in (("state_hca", "state"), ("return_hca", "return"), ("pg", "pg"))}


def judge(m, p, Rng, delta, se_paper):
    e = abs(m - p) / abs(Rng)
    if e <= 0.05 or abs(m - p) <= delta + 2 * se_paper:
        return "MATCH", e
    return ("CLOSE" if e <= 0.20 else "DIVERGENT"), e


def rel_judge(m, p):
    """Plain relative rule: |m - p| / |p| <= 5 % MATCH, <= 20 % CLOSE, else DIVERGENT."""
    r = abs(m - p) / abs(p)
    return "MATCH" if r <= 0.05 else "CLOSE" if r <= 0.20 else "DIVERGENT"


E1_NOTE = "start level NON-COMPARABLE (ours −0.963, paper −1.80), see S-E1"


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{nd}f}" if x < 0 else f"{x:.{nd}f}"


def load(seed):
    p = R(f"seed_{seed}", "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def official(exp, si, stat, arm):
    """Authors' snippet in the numpy reference driver (100 runs, numpy seed 0): mean, SE (SE only for final/avg)."""
    if arm == "pg":
        return None, None
    if exp == "E2":
        d = json.load(open(R("official", "ref", "E2.json")))
        return mean_se(d["points"][si][arm])
    d = json.load(open(R("official", "ref", f"{exp}.json")))
    m = d["settings"][si]["methods"][arm]
    if stat in ("final", "avg"):
        return mean_se(m[stat])
    return float(m["mean_curve"][int(stat[2:])]), None


def main(pre_recheck: bool = False):
    if not pre_recheck:
        apply_recheck_reads()
    suffix = "_prerecheck" if pre_recheck else ""
    s0 = load(0)
    reps = {s: load(s) for s in (1, 3) if load(s) is not None}
    rows, lines = [], []
    lines.append("| Exp | setting | stat | arm | Paper | Ours (mean ± SE, n=100) | Official snippet (100 runs) | Δ% | e (norm.) | Judgment (pre-declared, norm.) | Judgment (relative) | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for exp, si, stat, Rng, delta, arms in P:
        rec = s0["experiments"][exp][si]
        for arm, (pv, pstd) in arms.items():
            x = np.asarray(rec["arms"][arm][stat])
            m, se = mean_se(x)
            om, ose = official(exp, si, stat, arm)
            j, e = judge(m, pv, Rng, delta, pstd / 10)
            note = f"lr {rec['arms']['pg']['lr']}" if arm == "pg" else ""
            if exp == "E3" and arm == "state_hca" and om is not None:
                note = "official = MC (no bootstrap in snippet); ours = 3-step"
            if exp == "E1":
                note = (note + "; " if note else "") + E1_NOTE
            rep_m = [float(np.mean(r["experiments"][exp][si]["arms"][arm][stat])) for r in reps.values()]
            row = dict(exp=exp, setting=rec["env_kwargs"], si=si, stat=stat, arm=arm, paper=pv, paper_std_read=pstd,
                       ours_mean=m, ours_se=se, ours_std=float(x.std(ddof=1)), official_mean=om, official_se=ose,
                       delta_pct=100 * (m - pv) / abs(pv) if pv != 0 else None, norm_err=e, judgment=j,
                       judgment_rel=rel_judge(m, pv), comparable=exp != "E1", replication_seed_means=rep_m)
            rows.append(row)
            sett = ",".join(f"{k}={v}" for k, v in rec["env_kwargs"].items() if k in ("sigma", "eps") and exp in ("E5", "E8"))
            off = "n/a" if om is None else (f"{om:.3f}" + (f" ± {ose:.3f}" if ose else ""))
            dp = "n/a" if row["delta_pct"] is None else f"{row['delta_pct']:+.0f}%"
            lines.append(f"| {exp} | {sett} | {stat} | {SHORT[arm]} | {pv:.3f} | {m:.3f} ± {se:.3f} | {off} | {dp} | {e:.2f} | {j} | {row['judgment_rel']} | {note} |")
    # E2
    lines2 = ["| p_long | arm | Paper | Ours online (mean ± SE, 100 reps) | Ours final-h | Official snippet | oracle (ours / paper) | e | Judgment (pre-declared, norm.) | Judgment (relative) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for i, pt in enumerate(s0["experiments"]["E2"]):
        orc = pt["oracle"]
        for arm in ARMS:
            x = np.asarray(pt["arms"][arm]["online"])
            m, se = mean_se(x)
            mf = float(np.mean(pt["arms"][arm]["final_h"]))
            om, ose = official("E2", i, None, arm)
            pv = E2_P[arm][i]
            j, e = judge(m, pv, orc, 0.03, E2_SE_PAPER if E2_SE is None else E2_SE[arm][i])
            rows.append(dict(exp="E2", si=i, p_long=pt["p_long"], stat="online", arm=arm, paper=pv, ours_mean=m, ours_se=se,
                             ours_final_h=mf, official_mean=om, official_se=ose, oracle=orc, norm_err=e, judgment=j,
                             judgment_rel=rel_judge(m, pv), comparable=True, delta_pct=100 * (m - pv) / abs(pv)))
            off = "n/a" if om is None else f"{om:.3f} ± {ose:.3f}"
            lines2.append(f"| {pt['p_long']:.2f} | {SHORT[arm]} | {pv:.2f} | {m:.3f} ± {se:.3f} | {mf:.3f} | {off} | "
                          f"{orc:.3f} / {E2_ORACLE_P[i]:.2f} | {e:.2f} | {j} | {rel_judge(m, pv)} |")
        j, e = judge(orc, E2_ORACLE_P[i], orc, 0.03, 0.0)
        rows.append(dict(exp="E2", si=i, p_long=pt["p_long"], stat="oracle", arm="oracle", paper=E2_ORACLE_P[i], ours_mean=orc,
                         norm_err=e, judgment=j))
    # ---------------------------------------------------------- primary family (pre-declared), seed 0
    def arr(exp, si, arm, stat, src=s0):
        return np.asarray(src["experiments"][exp][si]["arms"][arm][stat], float)

    def e2_abs_err_diff(src=s0):
        pt = src["experiments"]["E2"][6]
        o = pt["oracle"]
        return np.abs(np.asarray(pt["arms"]["pg"]["online"]) - o), np.abs(np.asarray(pt["arms"]["state_hca"]["online"]) - o)

    fam = {
        "P1 E1 State-PG ep100": lambda s: (arr("E1", 0, "state_hca", "final", s), arr("E1", 0, "pg", "final", s)),
        "P2 E2 p=0.9 |PG-oracle|-|State-oracle|": lambda s: e2_abs_err_diff(s),
        "P3 E3 State-PG ep500": lambda s: (arr("E3", 0, "state_hca", "final", s), arr("E3", 0, "pg", "final", s)),
        "P4 E3 Return-PG ep500": lambda s: (arr("E3", 0, "return_hca", "final", s), arr("E3", 0, "pg", "final", s)),
        "P5 E4 State-PG curve mean": lambda s: (arr("E4", 0, "state_hca", "avg", s), arr("E4", 0, "pg", "avg", s)),
        "P6 E5 σ=5 State-PG curve mean": lambda s: (arr("E5", 5, "state_hca", "avg", s), arr("E5", 5, "pg", "avg", s)),
        "P7 E6 State-PG ep100": lambda s: (arr("E6", 0, "state_hca", "final", s), arr("E6", 0, "pg", "final", s)),
        "P8 E6 Return-PG ep100": lambda s: (arr("E6", 0, "return_hca", "final", s), arr("E6", 0, "pg", "final", s)),
        "P9 E7 Return-PG ep100": lambda s: (arr("E7", 0, "return_hca", "final", s), arr("E7", 0, "pg", "final", s)),
        "P10 E8 ε=0 Return-PG curve mean": lambda s: (arr("E8", 0, "return_hca", "avg", s), arr("E8", 0, "pg", "avg", s)),
    }
    prim = {}
    for name, f in fam.items():
        a, b = f(s0)
        prim[name] = paired_compare(a, b)
        prim[name]["replication_seed_diffs"] = {str(k): float(np.mean(f(r)[0] - f(r)[1])) for k, r in reps.items()}
    hp = holm({k: v["t_p"] for k, v in prim.items()})
    lines3 = ["| # | claim (paper direction > 0) | diff mean ± SE | 95% t-CI | bootstrap 95% CI | wins/100 | p (t, two-sided) | p Holm | Wilcoxon p | verdict | seeds 1, 3 diff |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in prim.items():
        v.update(hp[k])
        v["verdict"] = "REPRODUCED" if (v["p_holm"] < 0.05 and v["diff_mean"] > 0) else (
            "REVERSED (significant, opposite sign)" if v["p_holm"] < 0.05 else "NOT REPRODUCED (n.s.)")
        v.pop("diff_per_seed", None)
        rs = ", ".join(f"{x:+.3f}" for x in v["replication_seed_diffs"].values())
        lines3.append(f"| {k.split()[0]} | {' '.join(k.split()[1:])} | {v['diff_mean']:+.4f} ± {v['diff_se']:.4f} | "
                      f"[{v['t95'][0]:+.4f}, {v['t95'][1]:+.4f}] | [{v['boot95'][0]:+.4f}, {v['boot95'][1]:+.4f}] | {v['wins']} | "
                      f"{v['t_p']:.2g} | {v['p_holm']:.2g} | {v['wilcoxon_p']:.2g} | {v['verdict']} | {rs} |")
    # ---------------------------------------------------------- secondary orderings (descriptive, uncorrected)
    sec = []
    sec_specs = [("E1", 0, "final", "pg", "return_hca", "paper: PG > Return (−0.18 vs −0.43)"),
                 ("E1", 0, "final", "state_hca", "return_hca", "paper: State > Return"),
                 ("E3", 0, "final", "return_hca", "state_hca", "paper: Return > State (0.83 vs 0.68)"),
                 ("E4", 0, "ep100", "state_hca", "pg", "paper: State > PG at ep100 (0.72 vs 0.53)"),
                 ("E4", 0, "ep100", "pg", "return_hca", "paper: PG > Return at ep100 (0.53 vs 0.46)"),
                 ("E6", 0, "final", "state_hca", "return_hca", "paper: State > Return"),
                 ("E7", 0, "final", "pg", "state_hca", "paper: PG > State (State flat)")]
    for si in range(6):
        sec_specs.append(("E5", si, "avg", "state_hca", "pg", f"paper σ={si}: {E5_P[si][0]} vs {E5_P[si][2]}"))
        sec_specs.append(("E5", si, "avg", "return_hca", "pg", f"paper σ={si}: {E5_P[si][1]} vs {E5_P[si][2]}"))
    for si in range(5):
        sec_specs.append(("E8", si, "avg", "return_hca", "pg", f"paper ε={si/10}: {E8_P[si][1]} vs {E8_P[si][2]}"))
        sec_specs.append(("E8", si, "avg", "pg", "state_hca", f"paper ε={si/10}: {E8_P[si][2]} vs {E8_P[si][0]}"))
    lines4 = ["| Exp | setting | stat | A − B | paper sign | ours diff ± SE | bootstrap 95% CI | p (t) | agrees? |", "|---|---|---|---|---|---|---|---|---|"]
    for exp, si, stat, a, b, note in sec_specs:
        c = paired_compare(arr(exp, si, a, stat), arr(exp, si, b, stat))
        pa = dict(P_lookup(exp, si, stat))
        psign = np.sign(pa[a] - pa[b]) if pa else np.nan
        if psign == 0:  # the paper shows a tie
            agree = "paper tie; ours differs (sig.)" if c["t_p"] < 0.05 else "paper tie; ours n.s."
        else:
            agree = ("yes" if np.sign(c["diff_mean"]) == psign and c["t_p"] < 0.05 else
                     "opposite (sig.)" if c["t_p"] < 0.05 else "n.s.")
        sec.append(dict(exp=exp, si=si, stat=stat, a=a, b=b, note=note, diff=c["diff_mean"], se=c["diff_se"],
                        boot95=c["boot95"], t_p=c["t_p"], paper_sign=float(psign), agrees=agree))
        sett = s0["experiments"][exp][si]["env_kwargs"]
        sett = ",".join(f"{k}={v}" for k, v in sett.items() if k in ("sigma", "eps") and exp in ("E5", "E8"))
        lines4.append(f"| {exp} | {sett} | {stat} | {SHORT[a]} − {SHORT[b]} | {'+' if psign > 0 else '−' if psign < 0 else '0'} | "
                      f"{c['diff_mean']:+.4f} ± {c['diff_se']:.4f} | [{c['boot95'][0]:+.4f}, {c['boot95'][1]:+.4f}] | {c['t_p']:.2g} | {agree} |")
    # E2 secondary: every p, |HCA - oracle| < |PG - oracle|
    for i, pt in enumerate(s0["experiments"]["E2"]):
        o = pt["oracle"]
        for arm in ("state_hca", "return_hca"):
            c = paired_compare(np.abs(np.asarray(pt["arms"]["pg"]["online"]) - o), np.abs(np.asarray(pt["arms"][arm]["online"]) - o))
            agree = "yes" if c["diff_mean"] > 0 and c["t_p"] < 0.05 else ("opposite (sig.)" if c["t_p"] < 0.05 else "n.s.")
            sec.append(dict(exp="E2", p_long=pt["p_long"], a="pg_abs_err", b=f"{arm}_abs_err", diff=c["diff_mean"], se=c["diff_se"],
                            boot95=c["boot95"], t_p=c["t_p"], agrees=agree))
            lines4.append(f"| E2 | p={pt['p_long']:.2f} | online | abs err PG − abs err {SHORT[arm]} | + | {c['diff_mean']:+.4f} ± {c['diff_se']:.4f} | "
                          f"[{c['boot95'][0]:+.4f}, {c['boot95'][1]:+.4f}] | {c['t_p']:.2g} | {agree} |")
    counts, counts_rel, counts_cmp, counts_rel_cmp = {}, {}, {}, {}
    n_flip = 0
    for r in rows:
        if r["exp"] == "E2" and r["arm"] == "oracle":
            continue
        counts[r["judgment"]] = counts.get(r["judgment"], 0) + 1
        counts_rel[r["judgment_rel"]] = counts_rel.get(r["judgment_rel"], 0) + 1
        n_flip += r["judgment"] != r["judgment_rel"]
        if r["comparable"]:
            counts_cmp[r["judgment"]] = counts_cmp.get(r["judgment"], 0) + 1
            counts_rel_cmp[r["judgment_rel"]] = counts_rel_cmp.get(r["judgment_rel"], 0) + 1
    out = {"rows": rows, "primary": prim, "secondary": sec, "judgment_counts": counts,
           "judgment_counts_relative": counts_rel, "rows_changing_judgment_between_rules": n_flip,
           "judgment_counts_excl_E1": counts_cmp, "judgment_counts_relative_excl_E1": counts_rel_cmp,
           "replication_seeds": list(reps), "provenance": s0["provenance"], "pg_tuned_lr": s0["pg_tuned_lr"]}
    with open(R("full", f"report{suffix}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    txt = "\n".join(["## main table", *lines, "", "## E2", *lines2, "", "## primary", *lines3, "", "## secondary", *lines4,
                     "", f"counts {counts}", f"counts relative {counts_rel}", f"rows flipping between rules {n_flip}",
                     f"counts excl. E1 {counts_cmp}", f"counts relative excl. E1 {counts_rel_cmp}"])
    open(R("full", f"report_tables{suffix}.txt"), "w").write(txt + "\n")
    print(txt)


def P_lookup(exp, si, stat):
    if exp == "E5":
        return dict(zip(ARMS, E5_P[si]))
    if exp == "E8":
        return dict(zip(ARMS, E8_P[si]))
    for e, s, st, _, _, arms in P:
        if e == exp and s == si and st == stat:
            return {a: v[0] for a, v in arms.items()}
    return {}


if __name__ == "__main__":
    main(pre_recheck="--pre-recheck" in sys.argv)
