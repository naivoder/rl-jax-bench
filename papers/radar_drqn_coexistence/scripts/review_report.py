"""review grading (the README pre-declaration): re-grade at 500 training CPIs, lr sweep,
no act-time reset, constant-B rows,  dual-metric figure table. Seed is the unit.

Usage: python scripts/review_report.py [part ...]   parts: pi500 family500 lr nr e23 fig  (default: every part whose inputs exist)
Writes runs/review_report.json (merged with any previous parts).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import eval as EV  # noqa: E402
from full_report import judge, one_sided, summ  # noqa: E402
from rl_jax_bench.stats import holm, paired_compare, t_interval  # noqa: E402

OUT = os.path.join(RUNS, "review_report.json")


def ids_of(run):
    return [tuple(x) for x in json.load(open(os.path.join(run, "lanes.json")))["ids"]]


def cfg_of(run):
    return json.load(open(os.path.join(run, "lanes.json")))["config"]


def seg_of(e):
    return EV.exact_values(e, "eq3")["seg_cpis"]


def arm_metrics(run, arm):
    """{(e, r, s): lane_metrics} for a learner arm + its eval fingerprints."""
    ids, ntr = ids_of(run), cfg_of(run)["train_cpis"]
    z = np.load(os.path.join(run, f"arm_{arm}.npz"))
    o, fp = z["outs"][:, ntr:], z["fps"][:, ntr:]
    return {k: EV.lane_metrics(o[i], seg_of(k[0])) for i, k in enumerate(ids)}, {k: fp[i] for i, k in enumerate(ids)}


def base_metrics(run, name):
    ids = ids_of(run)
    bz = np.load(os.path.join(run, "baselines.npz"))
    o, fp = bz[f"{name}__outs"], bz[f"{name}__fps"]
    return {k: EV.lane_metrics(o[i], seg_of(k[0])) for i, k in enumerate(ids)}, {k: fp[i] for i, k in enumerate(ids)}


def v(m, e, r, seeds, met):
    return np.array([m[(e, r, s)][met] for s in seeds], float)


def ident(fa, fb, keys):
    return bool(all(np.array_equal(fa[k], fb[k]) for k in keys))


def fam(tests):
    """the README rule over a dict name -> (a, b)."""
    res = {k: one_sided(a, b) for k, (a, b) in tests.items()}
    adj = holm({k: x["p_one"] for k, x in res.items()})
    for k, x in res.items():
        x["p_holm"] = adj[k]["p_holm"]
        x["verdict"] = ("GO" if x["p_holm"] < 0.05 and x["diff_mean"] > 0 else
                        "REVERSED" if x["diff_mean"] < 0 and x["t95"][1] < 0 else "KILL")
        for kk in ("boot_dist",):
            x.pop(kk, None)
    return res


def desc(a, b):
    c = paired_compare(a, b, n_boot=10_000)
    return {k: c[k] for k in ("diff_mean", "diff_se", "t95", "boot95", "wins", "n", "t_p")}


SEEDS = list(range(10))
G3, TC = os.path.join(RUNS, "full"), os.path.join(RUNS, "full_500cpi")
P5, LR, NR = (os.path.join(RUNS, x) for x in ("review_pi500", "review_lr", "review_nr"))
E2R, E3R = os.path.join(RUNS, "review_e2"), os.path.join(RUNS, "review_e3")   # review E23, split into two processes (execution note)


def e23_metrics(arm):
    a, b = arm_metrics(E2R, arm), arm_metrics(E3R, arm)
    return {**a[0], **b[0]}, {**a[1], **b[1]}


def part_pi500():
    """PI / PI-static at 500 training CPIs vs pi* and vs PI at 100 (full-scale), every experiment; constant-B rows."""
    R = {}
    m5 = {n: base_metrics(P5, n) for n in ("pistar", "saa", "constb", "random", "pi", "pi_static")}
    g3 = {n: base_metrics(G3, n) for n in ("pistar", "pi", "pi_static")}
    keys = ids_of(P5)
    R["stream_identity_vs_full_pistar"] = {n: ident(m5[n][1], g3["pistar"][1], keys) for n in m5}
    meta = json.load(open(os.path.join(P5, "baselines_meta.json")))
    meta3 = json.load(open(os.path.join(G3, "baselines_meta.json")))
    for e in ("mc_p04", "mc_p02", "mc_sched", "fh"):
        for r in ("eq3", "lin"):
            met = "J" if e in ("mc_sched", "fh") else "J_conv"
            row = {n: summ(v(m5[n][0], e, r, SEEDS, met)) for n in m5}
            row["metric"] = met
            row["pi500_minus_pistar"] = desc(v(m5["pi"][0], e, r, SEEDS, met), v(m5["pistar"][0], e, r, SEEDS, met))
            row["pi500_minus_pi100"] = desc(v(m5["pi"][0], e, r, SEEDS, met), v(g3["pi"][0], e, r, SEEDS, met))
            row["pistatic500_minus_pistar"] = desc(v(m5["pi_static"][0], e, r, SEEDS, met), v(m5["pistar"][0], e, r, SEEDS, met))
            row["constb_minus_pistar_maxabs"] = float(np.abs(v(m5["constb"][0], e, r, SEEDS, met)
                                                            - v(m5["pistar"][0], e, r, SEEDS, met)).max())
            idx = [keys.index((e, r, s)) for s in SEEDS]
            row["pi500_seen"] = dict(states=float(np.mean([meta["pi_meta"][i]["n_seen_states"] for i in idx])),
                                     sa=float(np.mean([meta["pi_meta"][i]["n_seen_sa"] for i in idx])))
            k3 = ids_of(G3)
            i3 = [k3.index((e, r, s)) for s in SEEDS]
            row["pi100_seen"] = dict(states=float(np.mean([meta3["pi_meta"][i]["n_seen_states"] for i in i3])),
                                     sa=float(np.mean([meta3["pi_meta"][i]["n_seen_sa"] for i in i3])))
            if e == "mc_sched":
                for n in ("pistar", "pi", "constb", "random"):
                    js = np.array([m5[n][0][(e, r, s)]["J_seg"] for s in SEEDS])
                    row[f"J_seg_{n}"] = js.mean(0).tolist()
                    row[f"J_seg_se_{n}"] = (js.std(0, ddof=1) / np.sqrt(len(js))).tolist()
            for n in ("pi", "constb", "saa", "pistar"):
                row[f"table2_{n}"] = {k: summ(v(m5[n][0], e, r, SEEDS, k)) for k in ("bw_mhz", "coll_pct", "missed_pct", "adapt_pct")}
            R[f"{e}/{r}"] = row
    return R


def part_family500():
    """Rule 1: P1-P8 at 500 training CPIs (E1 learners = T-C, E2 learners = review E23, PI/SAA = review PI500)."""
    tcm = {a: arm_metrics(TC, a) for a in ("dqn", "ddrqn")}
    e2m = {a: arm_metrics(E2R, a) for a in ("dqn", "ddrqn")}
    b = {n: base_metrics(P5, n) for n in ("pistar", "saa", "pi")}
    L = {}
    for a in ("dqn", "ddrqn"):
        L[a] = ({**tcm[a][0], **e2m[a][0]}, {**tcm[a][1], **e2m[a][1]})
    L.update(b)
    keys = [(e, r, s) for e in ("mc_p04", "mc_p02") for r in ("eq3", "lin") for s in SEEDS]
    R = {"stream_identity": {n: ident(L[n][1], b["pistar"][1], keys) for n in L}}
    for r in ("eq3", "lin"):
        J = lambda e, n: v(L[n][0], e, r, SEEDS, "J_conv")  # noqa: E731
        S = lambda e, n: v(L[n][0], e, r, SEEDS, "sigma_within")  # noqa: E731
        R[r] = fam({
            "P1 DDRQN>PI p0.4": (J("mc_p04", "ddrqn"), J("mc_p04", "pi")),
            "P2 DQN>PI p0.4": (J("mc_p04", "dqn"), J("mc_p04", "pi")),
            "P3 DDRQN>SAA p0.4": (J("mc_p04", "ddrqn"), J("mc_p04", "saa")),
            "P4 DQN>SAA p0.4": (J("mc_p04", "dqn"), J("mc_p04", "saa")),
            "P5 sd(DQN)>sd(DDRQN) p0.4": (S("mc_p04", "dqn"), S("mc_p04", "ddrqn")),
            "P6 DDRQN>=DQN p0.4": (J("mc_p04", "ddrqn"), J("mc_p04", "dqn")),
            "P7 DDRQN>PI p0.2": (J("mc_p02", "ddrqn"), J("mc_p02", "pi")),
            "P8 DQN>PI p0.2": (J("mc_p02", "dqn"), J("mc_p02", "pi")),
        })
        # H-a at 500 (descriptive here; E1 graded at T-C), levels, Table II M.C. at 500
        lv = {}
        for e in ("mc_p04", "mc_p02"):
            for n in ("dqn", "ddrqn", "pi", "saa", "pistar"):
                lv[f"{e} {n}"] = {k: summ(v(L[n][0], e, r, SEEDS, k)) for k in
                                  ("J", "J_conv", "sigma_within", "coll_pct", "missed_pct", "bw_mhz", "adapt_pct")}
            for n in ("dqn", "ddrqn"):
                g = J(e, "pistar") - J(e, n)
                ti = t_interval(g)
                lv[f"{e} H_a {n}"] = dict(gap=summ(g), approaches=bool(ti[1] <= 0.02))
        R[f"levels_{r}"] = lv
    return R


def part_lr():
    """Rule 2: P5/P6 at lr 1e-3 and 1e-4 on the fresh pool, Holm over 4 per reading."""
    out = EV.evaluate(LR)
    ids = ids_of(LR)
    per = {s: json.load(open(os.path.join(LR, f"seed_{s}", "metrics.json")))["lanes"] for s in SEEDS}
    R = {"stream_identity": out["stream_identity"], "seed_root": cfg_of(LR)["seed_root"], "n_lanes": len(ids)}
    for r in ("eq3", "lin"):
        k = f"mc_p04/{r}"
        J = lambda n: np.array([per[s][k]["metrics"][n]["J_conv"] for s in SEEDS])  # noqa: E731
        S = lambda n: np.array([per[s][k]["metrics"][n]["sigma_within"] for s in SEEDS])  # noqa: E731
        R[r] = fam({"P5(1e-3) sd(DQN)>sd(DDRQN)": (S("dqn"), S("ddrqn")),
                    "P6(1e-3) DDRQN>=DQN": (J("ddrqn"), J("dqn")),
                    "P5(1e-4) sd(DQN)>sd(DDRQN)": (S("dqn_lr1e4"), S("ddrqn_lr1e4")),
                    "P6(1e-4) DDRQN>=DQN": (J("ddrqn_lr1e4"), J("dqn_lr1e4"))})
        lv = {n: {m: summ(np.array([per[s][k]["metrics"][n][m] for s in SEEDS])) for m in
                  ("J", "J_conv", "sigma_within", "adapt_pct", "coll_pct")}
              for n in ("dqn", "dqn_lr1e4", "ddrqn", "ddrqn_lr1e4", "pistar", "saa", "pi", "constb")}
        best = {a: max((a, a + "_lr1e4"), key=lambda n: J(n).mean()) for a in ("dqn", "ddrqn")}
        R[f"desc_{r}"] = dict(levels=lv,
                              ddrqn_1e4_minus_1e3=desc(J("ddrqn_lr1e4"), J("ddrqn")),
                              dqn_1e4_minus_1e3=desc(J("dqn_lr1e4"), J("dqn")),
                              ddrqn_1e4_minus_dqn_1e3=desc(J("ddrqn_lr1e4"), J("dqn")),
                              best=best, best_contrast=desc(J(best["ddrqn"]), J(best["dqn"])),
                              gap_closed_frac=float((J("ddrqn_lr1e4").mean() - J("ddrqn").mean())
                                                    / (J("dqn").mean() - J("ddrqn").mean())),
                              H_a={n: dict(gap=summ(J("pistar") - J(n)),
                                           approaches=bool(t_interval(J("pistar") - J(n))[1] <= 0.02))
                                   for n in ("dqn", "dqn_lr1e4", "ddrqn", "ddrqn_lr1e4")})
    return R


def part_nr():
    """Rule 3: no act-time reset vs shipped DDRQN and vs DQN, full-scale lanes."""
    nr = arm_metrics(NR, "ddrqn_noreset")
    g3 = {n: arm_metrics(G3, n) for n in ("ddrqn", "dqn")}
    pis = base_metrics(G3, "pistar")
    keys = ids_of(NR)
    R = {"stream_identity": ident(nr[1], pis[1], keys)}
    for r in ("eq3", "lin"):
        a, b, q = (v(x[0], "mc_p04", r, SEEDS, "J_conv") for x in (nr, g3["ddrqn"], g3["dqn"]))
        c = one_sided(a, b)
        c.pop("boot_dist", None)
        d = desc(a, q)
        R[r] = dict(variant=summ(a), minus_ddrqn=c, minus_dqn=d,
                    sigma=summ(v(nr[0], "mc_p04", r, SEEDS, "sigma_within")),
                    adapt=summ(v(nr[0], "mc_p04", r, SEEDS, "adapt_pct")),
                    supported=bool(c["p_one"] < 0.05 and c["diff_mean"] > 0),
                    explains=bool(c["p_one"] < 0.05 and c["diff_mean"] > 0 and d["t95"][1] >= 0))
    return R


def part_e23():
    """E3 (Fig. 4) and E2 (Fig. 3R) learner levels at 500 CPIs, J and J_conv; f_seg at p 0.4/0.6/0.8."""
    R = {}
    L = {a: e23_metrics(a) for a in ("dqn", "ddrqn")}
    b = {n: base_metrics(P5, n) for n in ("pistar", "saa", "pi", "constb", "random")}
    L.update(b)
    for r in ("eq3", "lin"):
        for n in ("dqn", "ddrqn", "pi", "pistar", "saa", "constb"):
            js = np.array([L[n][0][("mc_sched", r, s)]["J_seg"] for s in SEEDS])
            jc = np.array([L[n][0][("mc_sched", r, s)]["J_seg_conv"] for s in SEEDS])
            R[f"E3 {n}/{r}"] = dict(J_seg=js.mean(0).tolist(), J_seg_se=(js.std(0, ddof=1) / np.sqrt(10)).tolist(),
                                    J_seg_conv=jc.mean(0).tolist())
        ex = EV.exact_values("mc_sched", r)["segments"]
        for n in ("dqn", "ddrqn", "pi", "constb"):
            fs = {}
            for kk in (2, 3, 4):
                g = ex[kk]["pistar"] - ex[kk]["saa"]
                f = np.array([(L[n][0][("mc_sched", r, s)]["J_seg_conv"][kk] - L["saa"][0][("mc_sched", r, s)]["J_seg_conv"][kk]) / g
                              for s in SEEDS])
                ti = t_interval(f)
                fs[kk] = dict(**summ(f), resolved=bool(ti[1] - ti[0] < 0.5))
            R[f"E3 f_seg {n}/{r}"] = fs
        for n in ("dqn", "ddrqn", "pi", "pistar"):
            R[f"E2 {n}/{r}"] = {m: summ(v(L[n][0], "mc_p02", r, SEEDS, m)) for m in ("J", "J_conv", "sigma_within", "adapt_pct")}
    R["stream_identity"] = {n: ident(L[n][1], b["pistar"][1], [(e, r, s) for e in ("mc_p02", "mc_sched")
                                                                for r in ("eq3", "lin") for s in SEEDS]) for n in L}
    return R


def part_fig():
    """every Fig 3 row on BOTH J (CPIs 1-70) and J_conv (11-70), judged by the README, at 100 CPIs (full-scale)."""
    per = {s: json.load(open(os.path.join(G3, f"seed_{s}", "metrics.json")))["lanes"] for s in SEEDS}
    reads = {"Fig3L pi*": ("mc_p04", "pistar", 0.845), "Fig3L DDRQN": ("mc_p04", "ddrqn", 0.835), "Fig3L DQN": ("mc_p04", "dqn", 0.75),
             "Fig3L PI": ("mc_p04", "pi", 0.71), "Fig3L random": ("mc_p04", "random", 0.43),
             "Fig3R pi*": ("mc_p02", "pistar", 0.87), "Fig3R DDRQN": ("mc_p02", "ddrqn", 0.86), "Fig3R DQN": ("mc_p02", "dqn", 0.82),
             "Fig3R PI": ("mc_p02", "pi", 0.80), "Fig3R random": ("mc_p02", "random", 0.42)}
    R = {}
    for rd in ("lin", "eq3"):
        for name, (e, m, pv) in reads.items():
            row = {}
            for met in ("J", "J_conv"):
                x = np.array([per[s][f"{e}/{rd}"]["metrics"][m][met] for s in SEEDS])
                s_ = summ(x)
                row[met] = dict(**judge(s_["mean"], pv, s_["half"]), se=s_["se"])
            R[f"{name}/{rd}"] = row
    return R


def part_fig500():
    """Fig 3 rows at the paper's 500 training CPIs (E1 learners T-C, E2 learners review E2, baselines review PI500),
    both J and J_conv; Table II M.C. learner rows at 500; Fig 4 rows at 500 if review E3 exists."""
    L = {a: ({**arm_metrics(TC, a)[0], **arm_metrics(E2R, a)[0]}) for a in ("dqn", "ddrqn")}
    for n in ("pistar", "pi", "random"):
        L[n] = base_metrics(P5, n)[0]
    reads = {"Fig3L pi*": ("mc_p04", "pistar", 0.845), "Fig3L DDRQN": ("mc_p04", "ddrqn", 0.835), "Fig3L DQN": ("mc_p04", "dqn", 0.75),
             "Fig3L PI": ("mc_p04", "pi", 0.71), "Fig3L random": ("mc_p04", "random", 0.43),
             "Fig3R pi*": ("mc_p02", "pistar", 0.87), "Fig3R DDRQN": ("mc_p02", "ddrqn", 0.86), "Fig3R DQN": ("mc_p02", "dqn", 0.82),
             "Fig3R PI": ("mc_p02", "pi", 0.80), "Fig3R random": ("mc_p02", "random", 0.42)}
    R = {}
    for rd in ("lin", "eq3"):
        for name, (e, m, pv) in reads.items():
            row = {}
            for met in ("J", "J_conv"):
                s_ = summ(v(L[m], e, rd, SEEDS, met))
                row[met] = dict(**judge(s_["mean"], pv, s_["half"]), se=s_["se"])
            R[f"{name}/{rd}"] = row
    t2 = {"dqn": (60.0, 0.0, 48.9, 0.0), "ddrqn": (63.04, 0.25, 42.6, 11.4), "pi": (55.08, 3.8, 65.8, 20.6)}
    for m, pv in t2.items():
        for met, pp in zip(("bw_mhz", "coll_pct", "missed_pct", "adapt_pct"), pv):
            s_ = summ(v(L[m], "mc_p04", "eq3", SEEDS, met))
            j = (dict(paper=0.0, ours=s_["mean"], delta_pct=None, judgment="MATCH" if abs(s_["mean"]) < 0.05 else "DIVERGENT")
                 if pp == 0.0 else judge(s_["mean"], pp, 0.0))
            R[f"T2 MC {m} {met}"] = dict(**j, se=s_["se"])
    if done(E3R):
        E = {a: arm_metrics(E3R, a)[0] for a in ("dqn", "ddrqn")}
        seg_reads = {"ddrqn": [0.92, 0.86, 0.835, 0.83, 0.83], "dqn": [0.89, 0.87, 0.73, None, 0.83]}
        for rd in ("lin", "eq3"):
            for m, rr in seg_reads.items():
                js = np.array([E[m][("mc_sched", rd, s)]["J_seg"] for s in SEEDS])
                for kk, pv in enumerate(rr):
                    if pv is None:
                        continue
                    s_ = summ(js[:, kk])
                    R[f"Fig4 seg{kk + 1} {m}/{rd}"] = dict(**judge(s_["mean"], pv, s_["half"]), se=s_["se"])
    return R


PARTS = dict(fig500=(part_fig500, [P5, E2R, TC]), pi500=(part_pi500, [P5]), family500=(part_family500, [P5, E2R, TC]), lr=(part_lr, [LR]), nr=(part_nr, [NR]),
             e23=(part_e23, [E2R, E3R, P5]), fig=(part_fig, [G3]))


def done(run):
    return os.path.exists(os.path.join(run, "lanes.json")) and (
        os.path.exists(os.path.join(run, "baselines.npz")) or os.path.exists(os.path.join(run, "arm_ddrqn.npz"))
        or any(f.startswith("arm_") for f in os.listdir(run) if "ddrqn" in f))


def main():
    want = sys.argv[1:] or list(PARTS)
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for p in want:
        fn, deps = PARTS[p]
        if not all(done(d) for d in deps):
            print(f"[WARN] skip {p}: inputs missing")
            continue
        R[p] = fn()
        print(f"[INFO] part {p} done")
    json.dump(R, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
