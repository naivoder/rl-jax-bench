"""Full-stage report (pre-declared in the README).

Primary family (README): one-sided paired t (separate > shared-official), Holm over the tests run (P1 51k, P3 20k,
P5 150k) and the conservative m = 5 version (P2, P4 not run -> p = 1).  Secondary: S-E6 (detach / noleak), S-Len
(L = 5), S-Grad (shared-RNN vs separate-critic-RNN official grad-norm ratio).  Judgements vs the paper read-off, the
authors' logs (same window) and the official code on our CPU (16k-20k only).  Writes runs/full/report.json and the
per-seed aggregate runs/seed_<n>/metrics.json (stage full).
"""
import json
import os
import sys

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_jax_bench.stats import holm, mean_se, paired_compare, t_interval  # noqa: E402

from src.eval import author_final, final_value, judge, judge_detail, load_run  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G2, G3 = os.path.join(REPO, "runs", "small"), os.path.join(REPO, "runs", "full")
LOGP = "pomdp/Pendulum/V/{a}/len-{L}/*/*/*/oar/*/progress.csv"
LOGS = "meta/PointRobotSparse-v0/{a}/len-64/*/*/*/or/*/progress.csv"
RAND = json.load(open(os.path.join(G3, "random_policy.json")))


def finals(run_dir, variant):
    d = load_run(run_dir).get(variant, {})
    return {s: final_value(d[s]) for s in sorted(d)}, d


def one_sided(a, b):
    """paired one-sided t (H1: mean(a - b) > 0) + the shared paired_compare record."""
    r = paired_compare(a, b)
    t = st.ttest_rel(a, b)
    r["p_one_sided"] = float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2)
    return r


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else None


def row(metric, paper, ours_vals, refs, lo, hi, notes=""):
    """one full stage table row; judgement vs the authors' logs (exact, same window) with the paper read-off beside it."""
    m, se = mean_se(ours_vals)
    out = {"metric": metric, "paper": paper, "ours": ours_vals, "ours_mean": m, "ours_se": se, "n": len(ours_vals),
           "refs": {}, "notes": notes}
    for name, vals in refs.items():
        if vals is None or len(vals) == 0:
            continue
        rm, rse = mean_se(vals)
        out["refs"][name] = {"vals": vals, "mean": rm, "se": rse, "sd": sd(vals), "n": len(vals),
                             "delta_pct": 100.0 * (m - rm) / abs(rm),
                             "judge_raw": judge(m, rm, sd(vals)),
                             # review: raw relative error, gap in the reference's sd / se units; the
                             # normalised scale is reported only (it no longer decides a judgement)
                             "detail": judge_detail(m, vals, lo=lo, hi=hi)}
    if isinstance(paper, dict) and "value" in paper:
        out["paper_delta_pct"] = 100.0 * (m - paper["value"]) / abs(paper["value"])
        out["paper_judge"] = judge(m, paper["value"], paper.get("spread"))
        out["paper_raw_rel_pct"] = 100.0 * (m - paper["value"]) / abs(paper["value"])
    return out


def grad_ratio(dsh, dsep, k_sh="off_rnn", k_sep="off_q_rnn"):
    seeds = sorted(set(dsh) & set(dsep))
    med = lambda m, k: float(np.median([t[k] for t in m["train"] if k in t]))  # noqa: E731
    sh = [med(dsh[s], k_sh) for s in seeds]
    sp = [med(dsep[s], k_sep) for s in seeds]
    lr = np.log10(np.array(sh) / np.array(sp))
    t = st.ttest_1samp(lr, 0.0)
    return {"seeds": seeds, "shared_off_rnn_median": sh, "separate_off_q_rnn_median": sp,
            "log10_ratio": lr.tolist(), "ratio_geomean": float(10 ** lr.mean()),
            "log10_ratio_t95": t_interval(lr), "t": float(t.statistic),
            "p_one_sided_ratio_gt_1": float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2)}


def author_grad_medians(rel_glob, col, max_steps, n0, per_iter):
    """per-run median of an authors'-log grad-norm column over training iterations with env steps <= max_steps.
    Training rows carry no env-step stamp: row i (0 = after the warm-up updates) is at n0 + i * per_iter env steps
    (one row per iteration, official learner.log)."""
    import csv
    import glob as _g
    from src.eval import AUTHOR_LOGS
    out = []
    for f in sorted(_g.glob(os.path.join(AUTHOR_LOGS, rel_glob))):
        v = [float(r[col]) for r in csv.DictReader(open(f)) if r.get(col) not in (None, "")]
        k = int((max_steps - n0) // per_iter) + 1
        if v:
            out.append(float(np.median(v[:k])))
    return out


def official_window(arm):
    p = os.path.join(REPO, "runs", "official_ref", "report.json")
    if not os.path.exists(p):
        return None
    a = json.load(open(p)).get(arm, {}).get("agg", {}).get("official_v0_ourhw", {})
    return a.get("window_seeds")


def main():
    rep = {"primary": {}, "rows": [], "secondary": {}}
    lo_p, hi_p = RAND["Pendulum-V"]["mean"], 0.0             # Pendulum: random floor, reward upper bound 0
    lo_s, hi_s = RAND["Semi-Circle"]["mean"], 103.0           # Semi-Circle: random floor, oracle (Fig. 11 read-off)

    # ---------------- P1: Pendulum-V sac-lstm-200-oar, 51k (continued small stage runs)
    sep, dsep = finals(os.path.join(G3, "P1_sac_separate"), "separate")
    off, doff = finals(os.path.join(G3, "P1_sac_shared"), "official")
    det, ddet = finals(os.path.join(G3, "P1_sac_shared"), "detach")
    full = all(m["total_env_steps_run"] == 51000 for m in list(dsep.values()) + list(doff.values()))
    rep["P1_full_horizon"] = bool(full)
    S = sorted(set(sep) & set(off) & set(det))
    a_sep = author_final(LOGP.format(a="sac_lstm", L="-1"))
    a_sh = author_final(LOGP.format(a="sac_lstm_shared", L="-1"))
    rep["primary"]["P1"] = {"horizon": "51k (full)", **one_sided([sep[s] for s in S], [off[s] for s in S])}
    rep["rows"].append(row("P1 Pendulum-V sac-lstm-200-oar SEPARATE, last 20 % of 51k",
                           {"value": -165.0, "spread": 15.0, "src": "Fig. 21 read-off"}, [sep[s] for s in S],
                           {"authors_logs_51k": a_sep}, lo_p, hi_p))
    rep["rows"].append(row("P1 Pendulum-V sac-lstm-200-oar SHARED (official), last 20 % of 51k",
                           {"value": -500.0, "spread": 50.0, "src": "Fig. 21 read-off (final; oscillates -300..-750)"},
                           [off[s] for s in S], {"authors_logs_51k": a_sh}, lo_p, hi_p))
    rep["rows"].append(row("P1 separate - shared (paired), 51k", None,
                           [sep[s] - off[s] for s in S], {}, None, None,
                           notes=f"authors' logs (unpaired) {np.mean(a_sep) - np.mean(a_sh):.1f}"))
    # official code on our CPU: 16k-20k only (same runs' small stage window)
    rep["official_code_16_20k"] = {"separate": official_window("separate"), "shared": official_window("shared")}

    # ---------------- P3 / P5: small stage data (reduced horizon)
    for key, run, w, glob_sep, glob_sh, lo, hi, pv in (
            ("P3", "P_td3", (16000, 20000), LOGP.format(a="td3_lstm", L="-1"), LOGP.format(a="td3_lstm_shared", L="-1"),
             lo_p, hi_p, None),
            ("P5", "S_td3", (120000, 150000), LOGS.format(a="td3_lstm"), LOGS.format(a="td3_lstm_shared"), lo_s, hi_s,
             None)):   # review: Table 7 (94.2 +- 0.6 at 1.5M) is NOT comparable with a 150k run
        s2, _ = finals(os.path.join(G2, f"{run}_separate"), "separate")
        o2, _ = finals(os.path.join(G2, f"{run}_shared"), "official")
        SS = sorted(set(s2) & set(o2))
        rep["primary"][key] = {"horizon": "20k (reduced)" if key == "P3" else "150k (truncated)",
                               **one_sided([s2[s] for s in SS], [o2[s] for s in SS])}
        name = "Pendulum-V td3-lstm-200-oar" if key == "P3" else "Semi-Circle td3-lstm-64-or"
        rep["rows"].append(row(f"{key} {name} SEPARATE, window {w[0] // 1000}k-{w[1] // 1000}k", pv,
                               [s2[s] for s in SS],
                               {"authors_logs_same_window": author_final(glob_sep, window=w),
                                "authors_logs_full_run": author_final(glob_sep)}, lo, hi))
        rep["rows"].append(row(f"{key} {name} SHARED (official), window {w[0] // 1000}k-{w[1] // 1000}k",
                               None,   # Fig. 6 read-off is over the full 1.5M; same-window logs are the reference
                               [o2[s] for s in SS],
                               {"authors_logs_same_window": author_final(glob_sh, window=w),
                                "authors_logs_full_run": author_final(glob_sh)}, lo, hi))
    pr = {k: v["p_one_sided"] for k, v in rep["primary"].items()}
    rep["holm_m3"] = holm(pr)
    rep["holm_m5"] = {k: v for k, v in holm({**pr, "P2": 1.0, "P4": 1.0}).items()}
    rep["primary_pass"] = {k: bool(rep["holm_m3"][k]["p_holm"] < 0.05 and rep["holm_m5"][k]["p_holm"] < 0.05)
                           for k in pr}

    # ---------------- S-E6 (mechanism) on P1 51k, noleak at 20k
    e6 = {"detach_gt_official": one_sided([det[s] for s in S], [off[s] for s in S]),
          "detach_minus_separate": paired_compare([det[s] for s in S], [sep[s] for s in S])}
    nl, _ = finals(os.path.join(G2, "P_sac_shared"), "noleak")
    o20, _ = finals(os.path.join(G2, "P_sac_shared"), "official")
    e6["noleak_minus_official_20k"] = paired_compare([nl[s] for s in sorted(nl)], [o20[s] for s in sorted(nl)])
    e6["holm"] = holm({"detach_gt_official": e6["detach_gt_official"]["p_one_sided"]})
    e6["detach_equivalence_band_25"] = bool(abs(e6["detach_minus_separate"]["t95"][0]) < 25
                                            and abs(e6["detach_minus_separate"]["t95"][1]) < 25)
    rep["secondary"]["S_E6"] = e6
    rep["rows"].append(row("E6 shared-DETACH (ours, not in paper), last 20 % of 51k", None, [det[s] for s in S],
                           {"authors_logs_separate_51k": a_sep}, lo_p, hi_p,
                           notes="compared with the authors' SEPARATE arm: no authors' detach run exists"))

    # ---------------- S-Len (L = 5), if run
    l5s, dl5s = finals(os.path.join(G3, "L5_sac_separate"), "separate")
    l5o, _ = finals(os.path.join(G3, "L5_sac_shared"), "official")
    if l5s and l5o:
        L = sorted(set(l5s) & set(l5o) & set(sep))
        a5s = author_final(LOGP.format(a="sac_lstm", L="5"))
        a5o = author_final(LOGP.format(a="sac_lstm_shared", L="5"))
        rep["secondary"]["S_Len"] = {
            "sep_L200_gt_sep_L5": one_sided([sep[s] for s in L], [l5s[s] for s in L]),
            "shared_L200_minus_shared_L5": paired_compare([off[s] for s in L], [l5o[s] for s in L]),
            "sep_L5_minus_shared_L5": paired_compare([l5s[s] for s in L], [l5o[s] for s in L])}
        rep["secondary"]["S_Len"]["holm"] = holm(
            {"sep_L200_gt_sep_L5": rep["secondary"]["S_Len"]["sep_L200_gt_sep_L5"]["p_one_sided"]})
        rep["rows"].append(row("Len: Pendulum-V sac-lstm-5-oar SEPARATE, last 20 % of 51k", None,
                               [l5s[s] for s in L], {"authors_logs_51k": a5s}, lo_p, hi_p))
        rep["rows"].append(row("Len: Pendulum-V sac-lstm-5-oar SHARED (official), last 20 % of 51k", None,
                               [l5o[s] for s in L], {"authors_logs_51k": a5o}, lo_p, hi_p))
    else:
        rep["secondary"]["S_Len"] = {"missing": True}

    # ---------------- S-Grad: the paper's gradient-explosion explanation
    gr = {"P1_sac_51k": grad_ratio(doff, dsep)}
    for key, run in (("P3_td3_20k", "P_td3"), ("P5_semicircle_150k", "S_td3")):
        _, ds = finals(os.path.join(G2, f"{run}_separate"), "separate")
        _, do = finals(os.path.join(G2, f"{run}_shared"), "official")
        gr[key] = grad_ratio(do, ds)
    gr["holm"] = holm({k: v["p_one_sided_ratio_gt_1"] for k, v in gr.items()})
    for key, gs, gh, T, n0, pi in (
            ("P1_sac_51k", LOGP.format(a="sac_lstm", L="-1"), LOGP.format(a="sac_lstm_shared", L="-1"), 51000, 1000, 200),
            ("P3_td3_20k", LOGP.format(a="td3_lstm", L="-1"), LOGP.format(a="td3_lstm_shared", L="-1"), 20000, 1000, 200),
            ("P5_semicircle_150k", LOGS.format(a="td3_lstm"), LOGS.format(a="td3_lstm_shared"), 150000, 48000, 3000)):
        a_sp = author_grad_medians(gs, "rl_loss/q_rnn_grad_norm", T, n0, pi)
        a_sh = author_grad_medians(gh, "rl_loss/rnn_grad_norm", T, n0, pi)
        if a_sp and a_sh:
            gr[key]["authors_logs"] = {"shared_rnn_medians": a_sh, "separate_q_rnn_medians": a_sp,
                                       "ratio_of_geomeans": float(10 ** (np.mean(np.log10(a_sh))
                                                                         - np.mean(np.log10(a_sp))))}
    rep["secondary"]["S_Grad"] = gr

    with open(os.path.join(G3, "report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)

    # per-seed aggregate in the required layout (extends the small stage file; small stage cells kept)
    for s in range(4):
        p = os.path.join(REPO, "runs", f"seed_{s}", "metrics.json")
        agg = json.load(open(p)) if os.path.exists(p) else {"seed": s, "cells": {}}
        agg["stage"] = "full"
        g3 = {"P1_sac_51k/separate": sep.get(s), "P1_sac_51k/official": off.get(s), "P1_sac_51k/detach": det.get(s),
              "L5_sac_51k/separate": l5s.get(s), "L5_sac_51k/official": l5o.get(s)}
        agg["full_cells"] = {k: v for k, v in g3.items() if v is not None}
        agg["full_curves"] = {}
        for k, d in (("P1_sac_51k/separate", dsep), ("P1_sac_51k/official", doff), ("P1_sac_51k/detach", ddet),
                     ("L5_sac_51k/separate", dl5s)):
            if s in d:
                agg["full_curves"][k] = {"env_steps": d[s]["eval_env_steps"], "returns": d[s]["eval_returns"]}
        with open(p, "w") as f:
            json.dump(agg, f, indent=1)

    # console summary
    for r in rep["rows"]:
        refs = {k: (round(v["mean"], 1), round(v["se"] or 0, 1), v["judge_raw"],
                    round(v["detail"]["raw_rel_pct"], 1), round(v["detail"]["gap_in_ref_sd"] or 0, 2),
                    round(v["detail"]["gap_in_ref_se"] or 0, 2), v["detail"]["match_by_spread_only"])
                for k, v in r["refs"].items()}
        print(f"{r['metric']}: ours {r['ours_mean']:.2f} ± {(r['ours_se'] or 0):.2f} (n={r['n']}) "
              f"paper {r.get('paper_judge', '-')} ({r.get('paper_raw_rel_pct', float('nan')):.1f} %) {refs}")
    for k, v in rep["primary"].items():
        print(k, v["horizon"], f"diff {v['diff_mean']:.1f} ± {v['diff_se']:.1f} p1 {v['p_one_sided']:.2e}",
              "holm3", round(rep["holm_m3"][k]["p_holm"], 5), "holm5", round(rep["holm_m5"][k]["p_holm"], 5),
              "boot", np.round(v["boot95"], 1).tolist(), "wins", v["wins"])
    print("E6", json.dumps({k: (round(v["diff_mean"], 1), round(v["diff_se"], 1), np.round(v["t95"], 1).tolist(),
                                round(v.get("p_one_sided", v["t_p"]), 4))
                            for k, v in e6.items() if isinstance(v, dict) and "diff_mean" in v}))
    if "missing" not in rep["secondary"]["S_Len"]:
        print("Len", json.dumps({k: (round(v["diff_mean"], 1), round(v["diff_se"], 1), np.round(v["t95"], 1).tolist(),
                                     round(v.get("p_one_sided", v["t_p"]), 4))
                                 for k, v in rep["secondary"]["S_Len"].items() if "diff_mean" in v}))
    print("Grad", json.dumps({k: (round(v["ratio_geomean"], 3), round(v["p_one_sided_ratio_gt_1"], 4),
                                  round(v.get("authors_logs", {}).get("ratio_of_geomeans", float("nan")), 3))
                              for k, v in gr.items() if "ratio_geomean" in v}), json.dumps(gr["holm"]))


if __name__ == "__main__":
    main()
