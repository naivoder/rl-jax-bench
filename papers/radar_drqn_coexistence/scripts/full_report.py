"""full-scale grading under the rules declared before the run (see the README).

Reads runs/full (train.py output), writes runs/seed_<n>/metrics.json via eval.evaluate, and runs/full/full_tests.json with:
primary (R_eq3) and secondary (R_lin) Holm families P1-P8, H-a, '>0.8', f, FH rule, paper-number judgments, ablations,
theory checks, frozen-vs-online. Seed is the unit. Usage: python scripts/full_report.py [runs/full]
"""
import json
import os
import sys

import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import eval as EV  # noqa: E402
from rl_jax_bench.stats import holm, mean_se, paired_compare, t_interval  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "runs", "full")
LEARNERS = ["dqn", "ddqn", "ddqn_tr10", "ddrqn"]


def lanes_of(run):
    lj = json.load(open(os.path.join(run, "lanes.json")))
    seeds = sorted({s for _, _, s in lj["ids"]})
    per = {s: json.load(open(os.path.join(run, f"seed_{s}", "metrics.json")))["lanes"] for s in seeds}
    return seeds, per


def vals(per, seeds, key, m, metric):
    return np.array([per[s][key]["metrics"][m][metric] for s in seeds], float)


def one_sided(a, b):
    """H1: mean(a - b) > 0. Returns paired_compare dict + one-sided p."""
    c = paired_compare(a, b, n_boot=10_000)
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        c["p_one"] = 1.0
    else:
        c["p_one"] = c["t_p"] / 2 if c["t"] > 0 else 1 - c["t_p"] / 2
    return c


def family(per, seeds, reading):
    k4, k2 = f"mc_p04/{reading}", f"mc_p02/{reading}"
    J = lambda k, m: vals(per, seeds, k, m, "J_conv")  # noqa: E731
    S = lambda k, m: vals(per, seeds, k, m, "sigma_within")  # noqa: E731
    tests = {
        "P1 DDRQN>PI p0.4": (J(k4, "ddrqn"), J(k4, "pi")),
        "P2 DQN>PI p0.4": (J(k4, "dqn"), J(k4, "pi")),
        "P3 DDRQN>SAA p0.4": (J(k4, "ddrqn"), J(k4, "saa")),
        "P4 DQN>SAA p0.4": (J(k4, "dqn"), J(k4, "saa")),
        "P5 sd(DQN)>sd(DDRQN) p0.4": (S(k4, "dqn"), S(k4, "ddrqn")),
        "P6 DDRQN>=DQN p0.4": (J(k4, "ddrqn"), J(k4, "dqn")),
        "P7 DDRQN>PI p0.2": (J(k2, "ddrqn"), J(k2, "pi")),
        "P8 DQN>PI p0.2": (J(k2, "dqn"), J(k2, "pi")),
    }
    res = {k: one_sided(a, b) for k, (a, b) in tests.items()}
    adj = holm({k: v["p_one"] for k, v in res.items()})
    for k, v in res.items():
        v["p_holm"] = adj[k]["p_holm"]
        if v["p_holm"] < 0.05 and v["diff_mean"] > 0:
            v["verdict"] = "GO"
        elif v["diff_mean"] < 0 and v["t95"][1] < 0:
            v["verdict"] = "REVERSED"
        else:
            v["verdict"] = "KILL"
    return res


def judge(ours, paper, half):
    d = ours - paper
    rel = abs(d) / abs(paper)
    if rel <= 0.05 or abs(d) <= 0.01 + half:
        j = "MATCH"
    elif rel <= 0.20:
        j = "CLOSE"
    else:
        j = "DIVERGENT"
    return dict(paper=paper, ours=float(ours), delta_pct=float(100 * d / paper), judgment=j)


def summ(x):
    m, se = mean_se(x)
    ti = t_interval(x)
    return dict(mean=float(m), se=float(se), t95=[float(ti[0]), float(ti[1])], n=len(x), half=float((ti[1] - ti[0]) / 2))


def main():
    out = EV.evaluate(RUN)                         # writes seed_<n>/metrics.json + summary.json
    seeds, per = lanes_of(RUN)
    R = dict(stream_identity=out["stream_identity"], seeds=seeds)
    # integrity: 70 rbar_c values per method, eval stream sha present
    k0 = per[seeds[0]]["mc_p04/eq3"]
    R["integrity"] = dict(n_rbar=len(k0["metrics"]["ddrqn"]["rbar_c"]), sha=k0["eval_stream_sha256"][:12],
                          methods=sorted(k0["metrics"].keys()))
    R["primary_eq3"] = family(per, seeds, "eq3")
    R["secondary_lin"] = family(per, seeds, "lin")
    # H-a and >0.8
    ha = {}
    for rd in ("eq3", "lin"):
        k = f"mc_p04/{rd}"
        jstar_exact = per[seeds[0]][k]["exact"]["segments"][0]["pistar"]
        for m in LEARNERS:
            g_pair = vals(per, seeds, k, "pistar", "J_conv") - vals(per, seeds, k, m, "J_conv")
            g_ex = jstar_exact - vals(per, seeds, k, m, "J_conv")
            ti, te = t_interval(g_pair), t_interval(g_ex)
            ha[f"{m}/{rd}"] = dict(gap_paired=summ(g_pair), gap_exact=summ(g_ex), approaches=bool(ti[1] <= 0.02),
                                   approaches_exact=bool(te[1] <= 0.02),
                                   J_conv=summ(vals(per, seeds, k, m, "J_conv")))
    R["H_a"] = ha
    R["above_0p8_lin"] = {m: dict(J_conv=summ(vals(per, seeds, "mc_p04/lin", m, "J_conv")),
                                  above=bool(vals(per, seeds, "mc_p04/lin", m, "J_conv").mean() > 0.8))
                          for m in LEARNERS + ["pistar"]}
    # gap fraction
    fr = {}
    for rd in ("eq3", "lin"):
        for m in LEARNERS + ["pi", "random"]:
            f = vals(per, seeds, f"mc_p04/{rd}", m, "f")
            ti = t_interval(f)
            fr[f"E1/{m}/{rd}"] = dict(**summ(f), resolved=bool(ti[1] - ti[0] < 0.5))
            fs = np.array([per[s][f"mc_sched/{rd}"]["metrics"][m]["f_seg"] for s in seeds], dtype=object)
            for kk, p in ((2, 0.4), (3, 0.6), (4, 0.8)):
                x = np.array([v for v in fs[:, kk]], float)
                ti = t_interval(x)
                fr[f"E3seg{p}/{m}/{rd}"] = dict(**summ(x), resolved=bool(ti[1] - ti[0] < 0.5))
    R["f"] = fr
    # FH rule
    fh = {}
    for rd in ("eq3", "lin"):
        for m in LEARNERS + ["pi", "saa", "pistar"]:
            J = vals(per, seeds, f"fh/{rd}", m, "J")
            c = vals(per, seeds, f"fh/{rd}", m, "coll_pct")
            mi = vals(per, seeds, f"fh/{rd}", m, "missed_pct")
            fh[f"{m}/{rd}"] = dict(J=summ(J), coll=summ(c), missed=summ(mi),
                                   bw=summ(vals(per, seeds, f"fh/{rd}", m, "bw_mhz")),
                                   adapt=summ(vals(per, seeds, f"fh/{rd}", m, "adapt_pct")),
                                   clean_seeds=int(((c == 0) & (mi == 0)).sum()),
                                   learns_sweep=bool(J.mean() >= 0.99 and ((c == 0) & (mi == 0)).sum() >= 9))
    R["FH"] = fh
    # paper-number judgments: figure levels (R_lin, inferred reward; eq3 reported too)
    fig = {}
    reads = {  # (experiment, method, metric, paper read)
        "Fig3L pi*": ("mc_p04", "pistar", "J_conv", 0.845), "Fig3L DDRQN": ("mc_p04", "ddrqn", "J_conv", 0.835),
        "Fig3L DQN": ("mc_p04", "dqn", "J", 0.75), "Fig3L PI": ("mc_p04", "pi", "J", 0.71),
        "Fig3L random": ("mc_p04", "random", "J", 0.43),
        "Fig3R pi*": ("mc_p02", "pistar", "J_conv", 0.87), "Fig3R DDRQN": ("mc_p02", "ddrqn", "J_conv", 0.86),
        "Fig3R DQN": ("mc_p02", "dqn", "J", 0.82), "Fig3R PI": ("mc_p02", "pi", "J", 0.80),
        "Fig3R random": ("mc_p02", "random", "J", 0.42),
    }
    for rd in ("lin", "eq3"):
        for name, (e, m, met, pv) in reads.items():
            x = vals(per, seeds, f"{e}/{rd}", m, met)
            s_ = summ(x)
            fig[f"{name}/{rd}"] = dict(**judge(s_["mean"], pv, s_["half"]), se=s_["se"], n=s_["n"], metric=met)
    seg_reads = {"pistar": [0.945, 0.87, 0.85, 0.84, 0.84], "ddrqn": [0.92, 0.86, 0.835, 0.83, 0.83],
                 "dqn": [0.89, 0.87, 0.73, None, 0.83], "pi": [0.945, 0.87, 0.74, 0.62, 0.47],
                 "random": [0.43] * 5}
    for rd in ("lin", "eq3"):
        for m, rr in seg_reads.items():
            js = np.array([per[s][f"mc_sched/{rd}"]["metrics"][m]["J_seg"] for s in seeds])
            for kk, pv in enumerate(rr):
                if pv is None:
                    continue
                s_ = summ(js[:, kk])
                fig[f"Fig4 seg{kk + 1} {m}/{rd}"] = dict(**judge(s_["mean"], pv, s_["half"]), se=s_["se"], n=s_["n"],
                                                        metric="J_seg")
    R["figures"] = fig
    # Table II (E5 = E1 eq3; FH under eq3)
    t2p = {"saa": (80.64, 22.58, 22.58, 21.2), "pi": (55.08, 3.8, 65.8, 20.6), "dqn": (60.0, 0.0, 48.9, 0.0),
           "ddrqn": (63.04, 0.25, 42.6, 11.4)}
    t2fh = {"saa": (64.0, 60.0, 40.0, 100.0), "pi": (80.0, 0.0, 0.0, 100.0), "dqn": (80.0, 0.0, 0.0, 100.0),
            "ddrqn": (80.0, 0.0, 0.0, 100.0)}
    tab = {}
    for blk, key, ref in (("MC", "mc_p04/eq3", t2p), ("FH", "fh/eq3", t2fh)):
        for m, pv in ref.items():
            for met, p in zip(("bw_mhz", "coll_pct", "missed_pct", "adapt_pct"), pv):
                x = vals(per, seeds, key, m, met)
                s_ = summ(x)
                if p == 0.0:
                    j = dict(paper=0.0, ours=s_["mean"], delta_pct=None,
                             judgment="MATCH" if abs(s_["mean"]) < 0.05 else "DIVERGENT")
                else:
                    j = judge(s_["mean"], p, 0.0)
                tab[f"{blk} {m} {met}"] = dict(**j, se=s_["se"], n=s_["n"])
    R["table2"] = tab
    # ablations (descriptive)
    ab = {}
    for rd in ("eq3", "lin"):
        for e in ("mc_p04", "mc_p02"):
            k = f"{e}/{rd}"
            for a, b, lab in (("ddqn", "dqn", "double-Q (MLP)"), ("ddrqn", "ddqn_tr10", "LSTM (same 32x10 traces)"),
                              ("ddqn_tr10", "ddqn", "trace batch 32x10 vs 32 (MLP)"), ("ddrqn", "ddqn", "LSTM+traces vs DDQN")):
                for met in ("J_conv", "sigma_within", "adapt_pct"):
                    c = paired_compare(vals(per, seeds, k, a, met), vals(per, seeds, k, b, met), n_boot=2000)
                    ab[f"{k} {a}-{b} {met}"] = dict(label=lab, **{kk: c[kk] for kk in ("diff_mean", "diff_se", "t95", "wins", "n", "t_p")})
    R["ablations"] = ab
    # theory checks
    th = {}
    for rd in ("eq3", "lin"):
        for e in ("mc_p04", "mc_p02"):
            k = f"{e}/{rd}"
            for m in LEARNERS + ["pi", "saa"]:
                d = vals(per, seeds, k, m, "J_conv") - vals(per, seeds, k, "pistar", "J_conv")
                th[f"T1 {k} {m}-pi*"] = dict(**summ(d), max_seed=float(d.max()))
        d = vals(per, seeds, f"mc_p02/{rd}", "saa", "J") - vals(per, seeds, f"mc_p02/{rd}", "pistar", "J")
        ex = per[seeds[0]][f"mc_p02/{rd}"]["exact"]["segments"][0]
        th[f"T2 p0.2 {rd} SAA-pi*"] = dict(max_abs_diff=float(np.abs(d).max()), exact_gap=ex["pistar"] - ex["saa"])
        js = np.array([per[s][f"mc_sched/{rd}"]["metrics"]["pi"]["J_seg_conv"] for s in seeds])
        th[f"T3 E3 PI J_seg_conv {rd}"] = dict(mean=js.mean(0).tolist(), se=(js.std(0, ddof=1) / np.sqrt(len(js))).tolist(),
                                               frozen_p01_prediction_lin=[0.933, 0.867, 0.733, 0.600, 0.467])
    R["theory"] = th
    # frozen vs online (descriptive)
    fz = {}
    for rd in ("eq3", "lin"):
        for e in ("mc_p04", "mc_p02", "mc_sched", "fh"):
            k = f"{e}/{rd}"
            for m in LEARNERS:
                if m + "_frozen" in per[seeds[0]][k]["metrics"]:
                    c = paired_compare(vals(per, seeds, k, m, "J"), vals(per, seeds, k, m + "_frozen", "J"), n_boot=2000)
                    fz[f"{k} {m} online-frozen"] = {kk: c[kk] for kk in ("diff_mean", "diff_se", "t95", "wins", "n")}
    R["frozen"] = fz
    # all-method level table (both readings, all experiments)
    lv = {}
    for k in per[seeds[0]]:
        for m in per[seeds[0]][k]["metrics"]:
            lv[f"{k} {m}"] = {met: summ(vals(per, seeds, k, m, met)) for met in
                              ("J", "J_conv", "sigma_within", "coll_pct", "missed_pct", "bw_mhz", "adapt_pct")}
    R["levels"] = lv
    with open(os.path.join(RUN, "full_tests.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    # console digest
    for fam in ("primary_eq3", "secondary_lin"):
        print(f"== {fam}")
        for k, v in R[fam].items():
            print(f"  {k:28s} diff={v['diff_mean']:+.4f}±{v['diff_se']:.4f} t95=[{v['t95'][0]:+.4f},{v['t95'][1]:+.4f}] "
                  f"boot=[{v['boot95'][0]:+.4f},{v['boot95'][1]:+.4f}] p1={v['p_one']:.2e} holm={v['p_holm']:.2e} "
                  f"wins={v['wins']}/{v['n']} -> {v['verdict']}")
    print("integrity", R["integrity"], "stream identity", all(R["stream_identity"].values()))


if __name__ == "__main__":
    main()
