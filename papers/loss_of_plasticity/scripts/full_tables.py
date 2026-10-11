"""full-scale tables and pre-declared tests -> runs/full_tables.json.

    uv run python -m scripts.full_tables [part ...]      # parts: b10 d12 methods act mnist primary (default all)

Paper values are plot reads.  Authors'-code values come from runs/oracle_pipeline/summary.json (their full
pipeline, their streams) and runs/oracle_ref/compare_g2.json (their learner on our streams), both CPU.
Judgement: pre-declared protocol rule; the paper's +-1 s.e. band is approximated by our s.e. at the same n (100 runs), widened by
the read precision (B.10 0.03, D.12 0.01, MNIST 0.1 pp).
"""
import json
import os
import sys

import numpy as np

from scipy import stats as st  # noqa: E402
from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402

from scripts.small_tables import B10, D12  # noqa: E402
from src.eval import RUNS, load, scr_arm_table  # noqa: E402

OUT = os.path.join(RUNS, "full_tables.json")


def judge(ours, paper, spread):
    if paper is None:
        return None
    rel = abs(ours - paper) / abs(paper)
    if abs(ours - paper) <= spread or rel <= 0.05:
        return "MATCH"
    return "CLOSE" if rel <= 0.20 else "DIVERGENT"


def delta_pct(ours, paper):
    return None if paper is None else 100.0 * (ours - paper) / abs(paper)


def one_sided(pc, expect_positive):
    """One-sided p of a paired_compare result for the hypothesis diff > 0 (expect_positive) or diff < 0."""
    p2, t = pc["t_p"], pc["t"]
    good = t > 0 if expect_positive else t < 0
    return p2 / 2 if good else 1 - p2 / 2


def oracle_pipeline():
    p = os.path.join(RUNS, "oracle_pipeline", "summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def oracle_ref():
    p = os.path.join(RUNS, "oracle_ref", "compare_g2.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def per_seed_bins(tag, arm, bin_periods=4, seeds=None):
    runs = load(tag)
    ss = sorted(runs) if seeds is None else seeds
    pp = np.asarray([runs[s]["arms"][arm]["mse_per_10k"] for s in ss])
    P = pp.shape[1] // bin_periods * bin_periods
    return pp, pp[:, :P].reshape(len(ss), -1, bin_periods).mean(-1)


# ------------------------------------------------------------------------------------------------ B.10
def part_b10(out):
    t = scr_arm_table("full_scr_b10")
    lin = scr_arm_table("full_scr_linear")
    lin_best = min(lin, key=lambda a: lin[a]["whole_run"][0])
    op = oracle_pipeline()
    orc = {"relu_0.01": op.get("flip_relu_bp_0.01", {}).get("official"),
           "tanh_0.01": op.get("flip_tanh_bp_0.01", {}).get("official")}
    rows, pv = {}, {}
    for a, (pf, pl) in B10.items():
        v = t[a]
        pc = paired_compare(v["per_seed_last_250k"], v["per_seed_min_bin_first_1M"])
        pv[a] = one_sided(pc, True)
        fb, lb = v["first_bin"], v["last_bin"]
        rows[a] = {"first_bin": fb, "paper_first": pf, "d_first_pct": delta_pct(fb[0], pf),
                   "judge_first": judge(fb[0], pf, 0.03 + fb[1]),
                   "last_bin": lb, "paper_last": pl, "d_last_pct": delta_pct(lb[0], pl),
                   "judge_last": judge(lb[0], pl, 0.03 + lb[1]) if pl else ("MATCH" if lb[0] > 1.2 else "DIVERGENT"),
                   "whole_run": v["whole_run"], "last_250k": v["last_250k"], "min_bin_first_1M": v["min_bin_first_1M"],
                   "lop_diff": pc["diff_mean"], "lop_se": pc["diff_se"], "lop_boot95": pc["boot95"],
                   "lop_p_one_sided": pv[a], "wins": pc["wins"],
                   "last250k_minus_linear_best": v["last_250k"][0] - lin[lin_best]["whole_run"][0],
                   "oracle": ({k: orc[a][k] for k in ("first_bin", "last_bin", "whole_run", "last_250k", "delta_lop")}
                              if orc.get(a) else None)}
    h = holm(pv)
    for a in rows:
        rows[a]["p_holm"] = h[a]["p_holm"]
    out["b10"] = rows
    out["linear"] = {a: lin[a]["whole_run"] for a in lin}
    out["linear_best"] = {"arm": lin_best, "whole_run": lin[lin_best]["whole_run"],
                          "last_250k": lin[lin_best]["last_250k"], "paper": "0.80-0.82"}
    for a, r in rows.items():
        print(f"B10 {a:17s} first {r['first_bin'][0]:.3f}±{r['first_bin'][1]:.3f} vs {r['paper_first']} "
              f"{r['judge_first']:9s} last {r['last_bin'][0]:.3f}±{r['last_bin'][1]:.3f} vs {r['paper_last']} "
              f"{r['judge_last']:9s} LoP {r['lop_diff']:+.3f} pHolm={r['p_holm']:.1e} vsLin {r['last250k_minus_linear_best']:+.3f}")
    print("linear", out["linear_best"])


# ------------------------------------------------------------------------------------------------ D.12
D12_UTILS = ("overall", "mean_corrected_contribution", "contribution", "weight_magnitude", "adaptation", "random")
RHOS = ("1e-05", "0.0001", "0.001", "0.01")


def d12_table(tag, orc_cases):
    t = scr_arm_table(tag)
    oref = oracle_ref()
    rows = {}
    for u in ("bp",) + D12_UTILS:
        for r in (RHOS if u != "bp" else ("bp",)):
            a = "bp" if u == "bp" else f"{u}_{r}"
            p = D12["bp"]["0.0001"] if u == "bp" else D12[u][r]
            v = t[a]["whole_run"]
            o = oref.get(orc_cases.get(a, ""), {}).get("whole_oracle")
            rows[a] = {"ours": v, "paper": p, "d_pct": delta_pct(v[0], p),
                       "judge": judge(v[0], p, 0.01 + v[1]) if p else ("MATCH" if v[0] > 0.9 else "DIVERGENT"),
                       "oracle": o, "oracle_case": orc_cases.get(a), "n_rep": t[a]["n_replacements_mean"]}
    # P1-P6 at rho = 1e-4
    ov = t["overall_0.0001"]["per_seed_whole_run"]
    tests = {"P1 overall<BP": paired_compare(ov, t["bp"]["per_seed_whole_run"])}
    for u, nm in (("random", "P2"), ("weight_magnitude", "P3"), ("contribution", "P4"),
                  ("mean_corrected_contribution", "P5"), ("adaptation", "P6")):
        tests[f"{nm} overall<{u}"] = paired_compare(ov, t[f"{u}_0.0001"]["per_seed_whole_run"])
    return rows, tests


def part_d12(out):
    rs = {"overall_0.0001": "tanh_adam_0.01_cbp1e-4_resample", "adaptation_0.0001": "tanh_adam_0.01_adapt1e-4_resample"}
    rows, tests = d12_table("full_scr_d12_rs", rs)
    out["d12_rs"] = {"rows": rows, "tests": tests}
    for a, r in rows.items():
        print(f"D12rs {a:36s} {r['ours'][0]:.3f}±{r['ours'][1]:.3f} paper {r['paper']} {r['judge']} oracle {r['oracle']}")
    # flip-one 3 M reading: full-scale run if it exists, else the small-scale 10-stream run (budget stop, labelled)
    ftag = "full_scr_d12" if os.path.isdir(os.path.join(RUNS, "full_scr_d12")) and load("full_scr_d12") else "small_scr_d12"
    if load(ftag):
        rows_f, tests_f = d12_table(ftag, {})
        # authors' learner on our flip-one streams ran only 1 M steps: compare over the first 1 M
        oref = oracle_ref()
        first1m = {}
        for a, case in (("bp", "tanh_adam_0.01_bp_flip"), ("overall_0.0001", "tanh_adam_0.01_cbp1e-4_flip")):
            pp, _ = per_seed_bins(ftag, a)
            first1m[a] = {"ours_first_1M": mean_se(pp[:, :100].mean(1)), "oracle_first_1M": oref.get(case, {}).get("whole_oracle")}
        h = holm({k: one_sided(v, False) for k, v in tests_f.items()})
        out["d12_flip"] = {"tag": ftag, "rows": rows_f, "tests": tests_f, "holm_one_sided": h, "first_1M_vs_oracle": first1m}
        for a, r in rows_f.items():
            print(f"D12flip {a:36s} {r['ours'][0]:.3f}±{r['ours'][1]:.3f} paper {r['paper']} {r['judge']}")
        print("D12flip first1M", first1m)


# ------------------------------------------------------------------------------------------------ SCR methods + ablations
def part_methods(out):
    tag = "full_scr_methods"
    runs = load(tag)
    seeds = sorted(runs)
    arms = list(runs[seeds[0]]["arms"])
    W = {a: np.asarray([runs[s]["arms"][a]["mse_per_10k"] for s in seeds]).mean(1) for a in arms}
    sel_idx = np.asarray([s < 50 for s in seeds])
    rep_idx = ~sel_idx
    fam = {"L2": [a for a in arms if a.startswith("l2_")], "S&P": [a for a in arms if a.startswith("snp_")],
           "CBP": [a for a in arms if a.startswith("cbp_")]}
    tab = scr_arm_table(tag)
    res, pv = {}, {}
    for m, cand in fam.items():
        best = min(cand, key=lambda a: W[a][sel_idx].mean())
        pc = paired_compare(W[best][rep_idx], W["bp"][rep_idx])
        pv[m] = pc["t_p"]
        res[m] = {"selected_on_0_49": best, "grid_on_0_49": {a: float(W[a][sel_idx].mean()) for a in cand},
                  "report_50_99": mean_se(W[best][rep_idx]), "bp_50_99": mean_se(W["bp"][rep_idx]), "vs_bp": pc}
    h = holm(pv)
    for m in res:
        res[m]["p_holm"] = h[m]["p_holm"]
    # ordering among selected arms on the report streams
    sel = {m: res[m]["selected_on_0_49"] for m in res}
    order = {"CBP<S&P": paired_compare(W[sel["CBP"]][rep_idx], W[sel["S&P"]][rep_idx]),
             "S&P<L2": paired_compare(W[sel["S&P"]][rep_idx], W[sel["L2"]][rep_idx])}
    op = oracle_pipeline()
    orc = {"bp": op.get("rs_relu_bp_0.01", {}).get("official"), "l2_0.01": op.get("rs_relu_l2_1e-2", {}).get("official"),
           "snp_0.001": op.get("rs_relu_snp_1e-3", {}).get("official"),
           "cbp_0.0001": op.get("rs_relu_cbp_1e-4_default", {}).get("official"),
           "abl_m100_0.0001": op.get("rs_relu_cbp_1e-4_mt100", {}).get("official")}
    all_arms = {a: {"whole_run": tab[a]["whole_run"], "last_250k": tab[a]["last_250k"], "last_bin": tab[a]["last_bin"],
                    "n_rep": tab[a]["n_replacements_mean"],
                    "oracle_whole_run": orc[a]["whole_run"] if orc.get(a) else None} for a in arms}
    # ablations (S-ABL): each vs full CBP at the same rho, all 100 streams
    abl_pairs = {"abl_random_1e-05": "cbp_1e-05", "abl_random_0.0001": "cbp_0.0001", "abl_random_0.001": "cbp_0.001",
                 "abl_contribution_0.0001": "cbp_0.0001", "abl_m0_0.0001": "cbp_0.0001", "abl_m100_0.0001": "cbp_0.0001",
                 "abl_m1000_0.0001": "cbp_0.0001", "abl_eta0_0.0001": "cbp_0.0001", "abl_bt_0.0001": "cbp_0.0001"}
    abl = {a: paired_compare(W[a], W[b]) for a, b in abl_pairs.items() if a in W}
    ha = holm({a: v["t_p"] for a, v in abl.items()})
    for a in abl:
        abl[a]["p_holm"] = ha[a]["p_holm"]
        abl[a]["vs"] = abl_pairs[a]
    # ablation by removal: CBP -> BP on last-250k (loss-of-plasticity magnitude)
    out["methods"] = {"s_scr_m": res, "ordering_50_99": order, "arms": all_arms, "ablation": abl}
    for m, r in res.items():
        print(f"S-SCR-M {m:4s} sel {r['selected_on_0_49']:12s} {r['report_50_99'][0]:.3f}±{r['report_50_99'][1]:.3f} "
              f"vs BP {r['bp_50_99'][0]:.3f} diff {r['vs_bp']['diff_mean']:+.3f} pHolm={r['p_holm']:.1e}")
    for k, v in order.items():
        print(f"order {k} diff {v['diff_mean']:+.4f}±{v['diff_se']:.4f} p={v['t_p']:.2g}")
    for a, v in abl.items():
        print(f"ABL {a:24s} vs {v['vs']:11s} {W[a].mean():.3f} vs {W[v['vs']].mean():.3f} diff {v['diff_mean']:+.4f}"
              f"±{v['diff_se']:.4f} pHolm={v['p_holm']:.2g}")


# ------------------------------------------------------------------------------------------------ all activations (S-ACT)
def part_act(out):
    tag = "full_scr_flip_methods"
    runs = load(tag)
    seeds = sorted(runs)
    arms = list(runs[seeds[0]]["arms"])
    PP = {a: np.asarray([runs[s]["arms"][a]["mse_per_10k"] for s in seeds]) for a in arms}
    L = {a: PP[a][:, -25:].mean(1) for a in arms}
    sel_idx = np.asarray([s < 50 for s in seeds])
    rep_idx = ~sel_idx
    res, pv = {}, {}
    for act in ("tanh", "sigmoid", "elu", "relu", "leaky_relu", "swish"):
        cand = [a for a in arms if a.startswith(f"{act}_cbp_")]
        best = min(cand, key=lambda a: L[a][sel_idx].mean())
        pc = paired_compare(L[best][rep_idx], L[f"{act}_bp"][rep_idx])
        pv[act] = one_sided(pc, False)
        bp_bins = PP[f"{act}_bp"][:, :300].reshape(len(seeds), -1, 4).mean(-1)
        cb_bins = PP[best][:, :300].reshape(len(seeds), -1, 4).mean(-1)
        lop_bp = L[f"{act}_bp"] - bp_bins[:, :25].min(1)
        lop_cb = L[best] - cb_bins[:, :25].min(1)
        res[act] = {"selected_on_0_49": best, "cbp_last250k_50_99": mean_se(L[best][rep_idx]),
                    "bp_last250k_50_99": mean_se(L[f"{act}_bp"][rep_idx]), "vs_bp": pc, "p_one_sided": pv[act],
                    "lop_bp_50_99": mean_se(lop_bp[rep_idx]), "lop_cbp_50_99": mean_se(lop_cb[rep_idx]),
                    "cbp_first_bin": mean_se(cb_bins[rep_idx, 0]), "cbp_last_bin": mean_se(cb_bins[rep_idx, -1]),
                    "bp_last_bin": mean_se(bp_bins[rep_idx, -1])}
    h = holm(pv)
    for act in res:
        res[act]["p_holm"] = h[act]["p_holm"]
    relu_other = {a: {"last_250k": mean_se(L[a]), "whole_run": mean_se(PP[a].mean(1))} for a in arms
                  if a.startswith("relu_l2") or a.startswith("relu_snp") or a.startswith("relu_")}
    out["act"] = {"s_act": res, "relu_arms": relu_other}
    for act, r in res.items():
        print(f"S-ACT {act:10s} sel {r['selected_on_0_49']:22s} CBP {r['cbp_last250k_50_99'][0]:.3f} BP "
              f"{r['bp_last250k_50_99'][0]:.3f} LoP bp {r['lop_bp_50_99'][0]:+.3f} cbp {r['lop_cbp_50_99'][0]:+.3f} "
              f"pHolm={r['p_holm']:.1e}")
    for a, r in relu_other.items():
        print(f"  {a:20s} last250k {r['last_250k'][0]:.3f} whole {r['whole_run'][0]:.3f}")


# ------------------------------------------------------------------------------------------------ MNIST
def part_mnist(out, tag="full_mnist", sel="cbp_1e-5"):
    runs = load(tag)
    seeds = sorted(runs)
    arms = list(runs[seeds[0]]["arms"])
    G = lambda a, k: np.asarray([runs[s]["arms"][a][k] for s in seeds])
    nT = G("bp", "acc").shape[1]
    k10 = min(10, nT)
    last10 = {a: G(a, "acc")[:, -k10:].mean(1) for a in arms}
    tab = {}
    for a in arms:
        acc = G(a, "acc")
        k5 = 5
        roll = np.stack([acc[:, i:i + k5].mean(1) for i in range(nT - k5 + 1)], 1)
        wa = G(a, "wmag_after")
        tab[a] = {"acc_task1": mean_se(acc[:, 0]), "acc_peak": float(acc.mean(0).max()),
                  "acc_peak_task": int(acc.mean(0).argmax()) + 1, "acc_last": mean_se(acc[:, -1]),
                  "acc_last10": mean_se(last10[a]), "acc_all": mean_se(acc.mean(1)),
                  "acc_task15": mean_se(acc[:, min(14, nT - 1)]),
                  "acc_task120": mean_se(acc[:, min(119, nT - 1)]),
                  "delta_peak5_last5": mean_se(roll.max(1) - acc[:, -k5:].mean(1)),
                  "dead_first": mean_se(G(a, "dead_pct")[:, 0]), "dead_last": mean_se(G(a, "dead_pct")[:, -1]),
                  "erank_first": mean_se(G(a, "erank_scaled")[:, 0]), "erank_last": mean_se(G(a, "erank_scaled")[:, -1]),
                  "erank_max": float(G(a, "erank_scaled").mean(0).max()),
                  "wmag_before_1": mean_se(G(a, "wmag_before")[:, 0]), "wmag_after_last": mean_se(wa[:, -1]),
                  "n_rep_per_task": float(G(a, "n_rep").mean()),
                  "acc_curve": acc.mean(0).tolist(), "dead_curve": G(a, "dead_pct").mean(0).tolist(),
                  "erank_curve": G(a, "erank_scaled").mean(0).tolist(), "wmag_curve": wa.mean(0).tolist()}
    prim = {"P7 CBP>BP": paired_compare(last10[sel], last10["bp"]),
            "P8 S&P>BP": paired_compare(last10["snp"], last10["bp"]),
            "P9 L2>BP": paired_compare(last10["l2"], last10["bp"]),
            "P10 CBP vs S&P": paired_compare(last10[sel], last10["snp"])}
    p10_all = {a: paired_compare(last10[a], last10["snp"]) for a in arms if a.startswith("cbp_")}
    # S-MNIST-LoP (BP 0.003 here; 0.01 / 0.001 from small-scale, 15 tasks)
    def lop(acc):
        T = acc.shape[1]
        roll = np.stack([acc[:, i:i + 5].mean(1) for i in range(T - 4)], 1)
        return roll.max(1) - acc[:, -5:].mean(1)
    lops = {f"bp_0.003 (g3, {nT} tasks)": lop(G("bp", "acc"))}
    g2 = load("small_mnist_bp_steps")
    if g2:
        for a in ("bp_0.01", "bp_0.001"):
            lops[f"{a} (g2, 15 tasks, seeds 0-2)"] = lop(np.asarray([g2[s]["arms"][a]["acc"] for s in sorted(g2)]))
    lop_t = {}
    for k, d in lops.items():
        t = st.ttest_1samp(d, 0.0)
        lop_t[k] = {"delta": mean_se(d), "p_one_sided": float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2)}
    hl = holm({k: v["p_one_sided"] for k, v in lop_t.items()})
    for k in lop_t:
        lop_t[k]["p_holm"] = hl[k]["p_holm"]
    # S-DIAG
    dl = lambda a: G(a, "dead_pct")[:, -1]
    el = lambda a: G(a, "erank_scaled")[:, -1]
    wl = lambda a: G(a, "wmag_after")[:, -1]
    diag = {"dead CBP<BP": (paired_compare(dl(sel), dl("bp")), False),
            "erank CBP>BP": (paired_compare(el(sel), el("bp")), True),
            "|w| BP end>start": (paired_compare(wl("bp"), G("bp", "wmag_before")[:, 0]), True),
            "|w| CBP<BP": (paired_compare(wl(sel), wl("bp")), False),
            "dead S&P<BP": (paired_compare(dl("snp"), dl("bp")), False)}
    pd = {k: one_sided(v, pos) for k, (v, pos) in diag.items()}
    hd = holm(pd)
    diag = {k: dict(v, p_one_sided=pd[k], p_holm=hd[k]["p_holm"]) for k, (v, _) in diag.items()}
    # S-MECH (ii): Spearman across all runs at the last task
    acc_all = np.concatenate([last10[a] for a in arms])
    feats = {"dead": np.concatenate([dl(a) for a in arms]), "erank": np.concatenate([el(a) for a in arms]),
             "wmag": np.concatenate([wl(a) for a in arms])}
    sp = {k: st.spearmanr(acc_all, v) for k, v in feats.items()}
    hs = holm({k: float(v.pvalue) for k, v in sp.items()})
    mech = {k: {"spearman": float(v.statistic), "p": float(v.pvalue), "p_holm": hs[k]["p_holm"]} for k, v in sp.items()}
    # S-MECH (iii): within BP, over tasks
    accc = G("bp", "acc").mean(0)
    mech_bp = {k: float(st.spearmanr(accc, G("bp", kk).mean(0)).statistic)
               for k, kk in (("dead", "dead_pct"), ("erank", "erank_scaled"), ("wmag", "wmag_after"))}
    # replacement count identity (S-MECH i)
    reps = {a: {"measured_per_task": tab[a]["n_rep_per_task"],
                "expected_per_task": 3 * 100 * float(a.split("_")[1]) * 60000} for a in arms if a.startswith("cbp_")}
    # paper (digitised Fig. 2b middle, BP width 100) vs our BP at matching tasks; Δ on the same horizon
    pr = json.load(open(os.path.join(os.path.dirname(RUNS), "src", "configs", "paper_reads", "fig2b_w100_digitised.json")))["acc_pct_by_task"]
    pc_ = np.asarray([pr[str(t)] if pr.get(str(t)) is not None else np.nan for t in range(1, 151)]) / 100.0
    accb = G("bp", "acc")
    cmp_ = {}
    for t in (1, 2, 3, 5, 10, 15, 25, 50, 75, 90, 100, 105, 110, 120, 150):
        if t <= nT and not np.isnan(pc_[t - 1]):
            v = mean_se(accb[:, t - 1])
            cmp_[f"task {t}"] = {"paper": float(pc_[t - 1]), "ours": v, "d_pct": delta_pct(v[0], pc_[t - 1]),
                                 "d_pp": 100 * (v[0] - pc_[t - 1]), "judge": judge(v[0], pc_[t - 1], 0.0015 + v[1])}
    pk = float(np.nanmax(pc_[:nT]))
    cmp_["peak"] = {"paper": pk, "paper_task": int(np.nanargmax(pc_[:nT])) + 1, "ours": float(accb.mean(0).max()),
                    "ours_task": int(accb.mean(0).argmax()) + 1, "judge": judge(float(accb.mean(0).max()), pk, 0.0015)}
    pr5 = np.asarray([np.nanmean(pc_[i:i + 5]) for i in range(nT - 4)])
    d_paper = float(np.nanmax(pr5) - np.nanmean(pc_[nT - 5:nT]))
    d_ours = mean_se(lop(accb))
    cmp_["delta_peak5_last5"] = {"paper": d_paper, "ours": d_ours, "d_pct": delta_pct(d_ours[0], d_paper),
                                 "judge": judge(d_ours[0], d_paper, 0.003 + d_ours[1])}
    # first 15 tasks vs the authors' pipeline (their streams, n = 5)
    op = oracle_pipeline()
    f15 = {}
    for a, case in (("bp", "mnist_w100_bp_0.003"), ("cbp_1e-4", "mnist_w100_cbp_1e-4")):
        acc = G(a, "acc")[:, :15]
        o = op.get(case, {}).get("official", {})
        f15[a] = {"acc_task15": mean_se(acc[:, 14]), "oracle_acc_task15": o.get("acc_last"),
                  "delta15": mean_se(lop(acc)), "oracle_delta15": o.get("delta_peak5_minus_last5"),
                  "dead15": mean_se(G(a, "dead_pct")[:, 14]), "oracle_dead15": o.get("dead_pct_last"),
                  "erank15": mean_se(G(a, "erank_scaled")[:, 14]), "oracle_erank15": o.get("erank_last"),
                  "wmag15_after": mean_se(G(a, "wmag_after")[:, 14]), "oracle_wmag15_mean": o.get("wmag_last_task_mean")}
    out["mnist"] = {"paper_compare_bp": cmp_, "first15_vs_oracle": f15, "tag": tag, "n_tasks": nT, "n_seeds": len(seeds), "selected_cbp": sel, "arms": tab,
                    "primary": prim, "p10_every_rho": p10_all, "s_mnist_lop": lop_t, "s_diag": diag,
                    "mech_spearman": mech, "mech_bp_over_tasks": mech_bp, "replacements": reps}
    for a, r in tab.items():
        print(f"MNIST {a:9s} t1 {r['acc_task1'][0]:.4f} peak {r['acc_peak']:.4f}@{r['acc_peak_task']} t15 "
              f"{r['acc_task15'][0]:.4f} t120 {r['acc_task120'][0]:.4f} last {r['acc_last'][0]:.4f} last10 "
              f"{r['acc_last10'][0]:.4f}±{r['acc_last10'][1]:.4f} Δ {100 * r['delta_peak5_last5'][0]:+.2f}pp dead "
              f"{r['dead_last'][0]:.2f} erank {r['erank_last'][0]:.1f} |w| {r['wmag_after_last'][0]:.4f}")
    for k, v in prim.items():
        print(f"{k:16s} diff {100 * v['diff_mean']:+.3f}±{100 * v['diff_se']:.3f} pp p={v['t_p']:.2g} wins {v['wins']}/{v['n']}")
    for k, v in p10_all.items():
        print(f"P10[{k}] diff {100 * v['diff_mean']:+.3f}±{100 * v['diff_se']:.3f} pp p={v['t_p']:.2g}")
    print("LoP", {k: (round(100 * v['delta'][0], 3), v['p_holm']) for k, v in lop_t.items()})
    print("DIAG", {k: (round(v['diff_mean'], 4), v['p_holm']) for k, v in diag.items()})
    print("MECH", mech, mech_bp, reps)


# ------------------------------------------------------------------------------------------------ primary family
def part_primary(out):
    pv = {}
    if "d12_rs" in out:
        for k, v in out["d12_rs"]["tests"].items():
            pv[k] = one_sided(v, False)       # direction declared (overall < BP / < u); report one-sided + two-sided
    if "mnist" in out:
        for k, v in out["mnist"]["primary"].items():
            pv[k] = v["t_p"] if k.startswith("P10") else one_sided(v, True)
    h = holm(pv)
    h2 = holm({k: (out["d12_rs"]["tests"][k]["t_p"] if k in out.get("d12_rs", {}).get("tests", {}) else
                   out["mnist"]["primary"][k]["t_p"]) for k in pv})
    out["primary_family"] = {k: {"p": pv[k], "p_holm": h[k]["p_holm"], "p_two_sided_holm": h2[k]["p_holm"]} for k in pv}
    for k, v in out["primary_family"].items():
        print(f"PRIMARY {k:28s} p={v['p']:.2g} holm={v['p_holm']:.2g} (two-sided holm {v['p_two_sided_holm']:.2g})")


def main(parts):
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    fns = {"b10": part_b10, "d12": part_d12, "methods": part_methods, "act": part_act, "mnist": part_mnist,
           "primary": part_primary}
    for p in parts or list(fns):
        fns[p](out)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1:])
