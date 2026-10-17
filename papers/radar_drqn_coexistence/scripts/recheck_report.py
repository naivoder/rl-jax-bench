"""Recheck grading (the README pre-declaration, rules 1-4). Seed is the unit.
Families (each its own Holm over P1-P8, R_eq3, rule the README): SGD {dqn_sgd, ddrqn_sgd}, SGD-k15
{dqn_sgd_k15, ddrqn_sgd_k15}, cadence {dqn, ddrqn_upd10}; PI/SAA/pi* from runs/recheck baselines. Also: H-a (E1, E2), the
E2 learning-validity check, levels + Table II M.C. metrics, and descriptive before/after vs the shipped full-scale DDRQN (a
different seed pool: Welch, unpaired). Writes runs/recheck_report.json. Arms whose npz is missing (killed by the hard line) are skipped."""
import json
import os
import sys

import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from review_report import RUNS, arm_metrics, base_metrics, desc, fam, ident, summ, t_interval, v  # noqa: E402

recheck, G3 = os.path.join(RUNS, "recheck"), os.path.join(RUNS, "full")
SEEDS = list(range(10))
R = "eq3"
FAMILIES = {"sgd": ("dqn_sgd", "ddrqn_sgd"), "sgd_k15": ("dqn_sgd_k15", "ddrqn_sgd_k15"), "cadence": ("dqn", "ddrqn_upd10")}


def main():
    have = {a for a in ("dqn", "dqn_sgd", "dqn_sgd_k15", "ddrqn_upd10", "ddrqn_sgd", "ddrqn_sgd_k15")
            if os.path.exists(os.path.join(recheck, f"arm_{a}.npz"))}
    L = {a: arm_metrics(recheck, a) for a in have}
    L.update({n: base_metrics(recheck, n) for n in ("pistar", "saa", "pi", "random")})
    keys = [(e, R, s) for e in ("mc_p04", "mc_p02") for s in SEEDS]
    out = {"arms_present": sorted(have), "stream_identity": {n: ident(L[n][1], L["pistar"][1], keys) for n in L}}
    J = lambda e, n: v(L[n][0], e, R, SEEDS, "J_conv")  # noqa: E731
    S = lambda e, n: v(L[n][0], e, R, SEEDS, "sigma_within")  # noqa: E731
    for fname, (dq, dd) in FAMILIES.items():
        if dq not in have or dd not in have:
            out[f"family_{fname}"] = "not run (arm missing)"
            continue
        out[f"family_{fname}"] = fam({
            "P1 DDRQN>PI p0.4": (J("mc_p04", dd), J("mc_p04", "pi")),
            "P2 DQN>PI p0.4": (J("mc_p04", dq), J("mc_p04", "pi")),
            "P3 DDRQN>SAA p0.4": (J("mc_p04", dd), J("mc_p04", "saa")),
            "P4 DQN>SAA p0.4": (J("mc_p04", dq), J("mc_p04", "saa")),
            "P5 sd(DQN)>sd(DDRQN) p0.4": (S("mc_p04", dq), S("mc_p04", dd)),
            "P6 DDRQN>=DQN p0.4": (J("mc_p04", dd), J("mc_p04", dq)),
            "P7 DDRQN>PI p0.2": (J("mc_p02", dd), J("mc_p02", "pi")),
            "P8 DQN>PI p0.2": (J("mc_p02", dq), J("mc_p02", "pi")),
        })
    lv = {}
    for e in ("mc_p04", "mc_p02"):
        for n in sorted(have) + ["pi", "saa", "pistar", "random"]:
            lv[f"{e} {n}"] = {k: summ(v(L[n][0], e, R, SEEDS, k)) for k in
                              ("J", "J_conv", "sigma_within", "coll_pct", "missed_pct", "bw_mhz", "adapt_pct")}
        for n in sorted(have):
            g = J(e, "pistar") - J(e, n)
            ti = t_interval(g)
            lv[f"{e} H_a {n}"] = dict(gap=summ(g), t95=ti, approaches=bool(ti[1] <= 0.02))
    out["levels"] = lv
    out["validity_E2"] = {n: lv[f"mc_p02 H_a {n}"]["approaches"] for n in sorted(have)}
    # descriptive before/after: shipped full-scale DDRQN / DQN (seed_root 20261016) vs recheck arms (20261018); unpaired Welch
    g3 = {a: arm_metrics(G3, a)[0] for a in ("ddrqn", "dqn")}
    ba = {}
    for e in ("mc_p04", "mc_p02"):
        for new, old in (("ddrqn_sgd", "ddrqn"), ("ddrqn_upd10", "ddrqn"), ("ddrqn_sgd_k15", "ddrqn"), ("dqn_sgd", "dqn"),
                         ("dqn_sgd_k15", "dqn"), ("dqn", "dqn")):
            if new not in have:
                continue
            for met in ("J_conv", "sigma_within", "adapt_pct", "coll_pct"):
                a, b = v(L[new][0], e, R, SEEDS, met), v(g3[old], e, R, SEEDS, met)
                w = st.ttest_ind(a, b, equal_var=False)
                ba[f"{e} {new} vs full {old} {met}"] = dict(after=summ(a), before=summ(b), diff=float(a.mean() - b.mean()),
                                                            welch_t=float(w.statistic), welch_p=float(w.pvalue))
    out["before_after_unpaired"] = ba
    # extra paired descriptive: DDRQN variants vs each other on the same recheck streams
    pd = {}
    for e in ("mc_p04", "mc_p02"):
        for a_, b_ in (("ddrqn_sgd", "dqn"), ("ddrqn_upd10", "dqn_sgd"), ("ddrqn_sgd", "ddrqn_upd10"),
                       ("ddrqn_sgd_k15", "ddrqn_sgd"), ("dqn_sgd_k15", "dqn_sgd"), ("dqn_sgd", "dqn")):
            if a_ in have and b_ in have:
                pd[f"{e} {a_} - {b_} J_conv"] = desc(J(e, a_), J(e, b_))
    out["paired_descriptive"] = pd
    with open(os.path.join(RUNS, "recheck_report.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    for fname in FAMILIES:
        x = out[f"family_{fname}"]
        if isinstance(x, str):
            print(fname, x)
            continue
        for k, r in x.items():
            print(f"{fname:8s} {k:28s} {r['diff_mean']:+.4f} [{r['t95'][0]:+.4f},{r['t95'][1]:+.4f}] wins {r.get('wins')} "
                  f"pHolm {r['p_holm']:.3g} -> {r['verdict']}")
    for k, x in lv.items():
        if "H_a" in k:
            print(k, "gap", round(x["gap"]["mean"], 4), "t95", [round(y, 4) for y in x["t95"]], "approaches", x["approaches"])
        else:
            print(k, {m: round(x[m]["mean"], 4) for m in ("J_conv", "sigma_within", "adapt_pct", "coll_pct", "missed_pct", "bw_mhz")})


if __name__ == "__main__":
    main()
