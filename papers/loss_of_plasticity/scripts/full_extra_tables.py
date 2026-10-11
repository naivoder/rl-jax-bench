"""Follow-up tables (review resolution) -> runs/full_extra_tables.json.

    uv run python -m scripts.full_extra_tables [part ...]
    parts: b10 lop mech d12 d12sens b10pt mnist   (default: every part whose runs exist)

b10     Fig. B.10 first/last bins re-judged against the digitised paper reads (), under three rules: the full-scale
        rule (|d| <= read precision + our s.e. or <= 5 % relative), the strict 5 % relative rule, and the absolute
        difference itself ( /).
lop     the unbiased loss-of-plasticity statistic (): early bin = argmin of the MEAN curve over the first 1 M on
        streams 0-49; last-250k minus that bin, paired, on streams 50-99 (one-sided, Holm within each tag).
mech    S-MECH (ii) without pseudo-replication (): Spearman within each arm (n = 10 seeds) and per seed across arms.
d12     Fig. D.12 on the paper protocol (flip-one, 3 M, 100 streams; full_scr_d12 merged from three stream blocks).
d12sens D.12 sensitivity on streams 0-29 (bias transfer, m = 1000, beta2 = 0.99, convention P) vs the default arms.
b10pt   Fig. B.10 on the paper-text 22-bit problem, streams 0-29 ().
mnist   width-100 L2 / S&P / CBP-bias-transfer sweep (seeds 0-2) and the final-seed arms (-M family).
"""
import json
import os
import sys

import numpy as np

from scipy import stats as st  # noqa: E402
from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402

from scripts.small_tables import B10, D12  # noqa: E402
from scripts.full_tables import d12_table, delta_pct, judge, one_sided  # noqa: E402
from src.eval import RUNS, load, scr_arm_table  # noqa: E402

OUT = os.path.join(RUNS, "full_extra_tables.json")

# Fig. B.10 first points, digitised (: p. 38 at 500 dpi, axes calibrated on the gridlines; confirmed by eye on
# 700-dpi crops in the follow-up). Last points: the original plot reads (re-checked in review; unchanged).
B10_FIRST_DIGITISED = {
    "tanh_0.01": 0.387, "tanh_0.003": 0.497, "tanh_0.001": 0.661,
    "sigmoid_0.01": 0.572, "sigmoid_0.003": 0.745, "sigmoid_0.001": 0.921,
    "elu_0.01": 0.552, "elu_0.003": 0.629, "elu_0.001": 0.753,
    "relu_0.01": 0.607, "relu_0.003": 0.623, "relu_0.001": 0.689,
    "leaky_relu_0.01": 0.625, "leaky_relu_0.003": 0.606, "leaky_relu_0.001": 0.706,
    "swish_0.01": 0.552, "swish_0.003": 0.621, "swish_0.001": 0.740}
READ_PREC_B10 = 0.03


def strict(ours, paper):
    if paper is None:
        return None
    rel = abs(ours - paper) / abs(paper)
    return "MATCH" if rel <= 0.05 else ("CLOSE" if rel <= 0.20 else "DIVERGENT")


def counts(xs):
    return {k: sum(1 for x in xs if x == k) for k in ("MATCH", "CLOSE", "DIVERGENT")}


def bins_of(tag, arm, seeds=None):
    runs = load(tag)
    ss = sorted(runs) if seeds is None else seeds
    pp = np.asarray([runs[s]["arms"][arm]["mse_per_10k"] for s in ss])
    P = pp.shape[1] // 4 * 4
    return pp, pp[:, :P].reshape(len(ss), -1, 4).mean(-1)


# ------------------------------------------------------------------------------------------------ B.10
def part_b10(out):
    t = scr_arm_table("full_scr_b10")
    rows = {}
    for a, (pf_old, pl) in B10.items():
        fb, lb = t[a]["first_bin"], t[a]["last_bin"]
        pf = B10_FIRST_DIGITISED[a]
        rows[a] = {"ours_first": fb, "paper_first_old_read": pf_old, "paper_first": pf,
                   "d_first": fb[0] - pf, "d_first_pct": delta_pct(fb[0], pf),
                   "judge_first_full_rule": judge(fb[0], pf, READ_PREC_B10 + fb[1]),
                   "judge_first_strict5": strict(fb[0], pf),
                   "judge_first_old_read": judge(fb[0], pf_old, READ_PREC_B10 + fb[1]),
                   "ours_last": lb, "paper_last": pl,
                   "d_last": None if pl is None else lb[0] - pl,
                   "judge_last_full_rule": judge(lb[0], pl, READ_PREC_B10 + lb[1]) if pl else ("MATCH" if lb[0] > 1.2 else "DIVERGENT"),
                   "judge_last_strict5": strict(lb[0], pl) if pl else ("MATCH" if lb[0] > 1.2 else "DIVERGENT")}
    summ = {k: counts([r[k] for r in rows.values()]) for k in
            ("judge_first_full_rule", "judge_first_strict5", "judge_first_old_read", "judge_last_full_rule",
             "judge_last_strict5")}
    rel = np.asarray([r["d_first_pct"] for r in rows.values()])
    summ["first_bin_rel_diff_pct"] = {"mean": float(rel.mean()), "min": float(rel.min()), "max": float(rel.max()),
                                      "n_above": int((rel > 0).sum())}
    out["b10"] = {"rows": rows, "summary": summ}
    for a, r in rows.items():
        print(f"B10 {a:17s} first ours {r['ours_first'][0]:.3f} paper {r['paper_first']:.3f} (old {r['paper_first_old_read']}) "
              f"{r['d_first_pct']:+5.1f}% {r['judge_first_full_rule']:9s}/{r['judge_first_strict5']:9s} | last "
              f"{r['ours_last'][0]:.3f} vs {r['paper_last']} {r['judge_last_full_rule']}/{r['judge_last_strict5']}")
    print("B10 summary", json.dumps(summ))


# ------------------------------------------------------------------------------------------------ unbiased LoP
def lop_unbiased(tag, arms=None):
    runs = load(tag)
    seeds = sorted(runs)
    arms = arms or list(runs[seeds[0]]["arms"])
    sel = np.asarray([s < 50 for s in seeds])
    res = {}
    for a in arms:
        pp, bins = bins_of(tag, a, seeds)
        l250 = pp[:, -25:].mean(1)
        j = int(np.argmin(bins[sel, :25].mean(0)))
        pc = paired_compare(l250[~sel], bins[~sel, j])
        res[a] = {"early_bin_index": j, "unbiased": [pc["diff_mean"], pc["diff_se"]], "boot95": pc["boot95"],
                  "p_one_sided": one_sided(pc, True), "wins": pc["wins"], "n": pc["n"],
                  "biased_min_of_bins": mean_se(l250 - bins[:, :25].min(1)),
                  "last_minus_first_bin": mean_se(bins[:, -1] - bins[:, 0])}
    h = holm({a: r["p_one_sided"] for a, r in res.items()})
    for a in res:
        res[a]["p_holm"] = h[a]["p_holm"]
    return res


def part_lop(out):
    b = lop_unbiased("full_scr_b10")
    fm = lop_unbiased("full_scr_flip_methods")
    cbp = {a: v for a, v in fm.items() if "_cbp_" in a}
    bp = {a: v for a, v in fm.items() if a.endswith("_bp")}
    # S-ACT selected rho per activation (full-scale, selected on streams 0-49 by last-250k MSE)
    act_sel = {"tanh": "0.0001", "sigmoid": "1e-05", "elu": "1e-05", "relu": "0.0001", "leaky_relu": "0.0001",
               "swish": "0.0001"}
    sel = {k: cbp[f"{k}_cbp_{r}"] for k, r in act_sel.items()}
    rng = lambda d: [min(v["unbiased"][0] for v in d.values()), max(v["unbiased"][0] for v in d.values())]
    out["lop_unbiased"] = {
        "full_scr_b10": b, "full_scr_flip_methods_cbp": cbp, "full_scr_flip_methods_bp": bp, "s_act_selected": sel,
        "summary": {"b10_range": rng(b), "b10_max_p_holm": max(v["p_holm"] for v in b.values()),
                    "cbp_all_range": rng(cbp), "cbp_selected_range": rng(sel),
                    "cbp_n_p_lt_0.01": sum(1 for v in cbp.values() if v["p_one_sided"] < 0.01), "cbp_n": len(cbp),
                    "flip_bp_range": rng(bp),
                    "b10_last_minus_first_negative": [a for a, v in b.items() if v["last_minus_first_bin"][0] < 0]}}
    for tag, d in (("b10", b), ("flipCBP", cbp), ("flipBP", bp)):
        for a, v in d.items():
            print(f"LoP {tag:7s} {a:22s} bin{v['early_bin_index']:2d} unbiased {v['unbiased'][0]:+.3f}±{v['unbiased'][1]:.3f} "
                  f"pHolm {v['p_holm']:.1g} wins {v['wins']}/{v['n']} | biased {v['biased_min_of_bins'][0]:+.3f} "
                  f"| last-first {v['last_minus_first_bin'][0]:+.3f}")
    print("LoP summary", json.dumps(out["lop_unbiased"]["summary"]))


# ------------------------------------------------------------------------------------------------ S-MECH (ii)
def part_mech(out, tag="full_mnist"):
    runs = load(tag)
    seeds = sorted(runs)
    arms = list(runs[seeds[0]]["arms"])
    G = lambda a, k: np.asarray([runs[s]["arms"][a][k] for s in seeds])
    last10 = {a: G(a, "acc")[:, -10:].mean(1) for a in arms}
    feats = {"dead": lambda a: G(a, "dead_pct")[:, -1], "erank": lambda a: G(a, "erank_scaled")[:, -1],
             "wmag": lambda a: G(a, "wmag_after")[:, -1]}
    within, per_seed = {}, {}
    for k, f in feats.items():
        within[k] = {a: float(st.spearmanr(last10[a], f(a)).statistic) for a in arms}
        X = np.stack([last10[a] for a in arms], 1)            # (S, A)
        Y = np.stack([f(a) for a in arms], 1)
        rs = np.asarray([st.spearmanr(X[i], Y[i]).statistic for i in range(len(seeds))])
        per_seed[k] = {"mean": float(np.nanmean(rs)), "min": float(np.nanmin(rs)), "max": float(np.nanmax(rs)),
                       "n_seeds": len(seeds), "n_arms": len(arms)}
    out["mech_ii"] = {"within_arm_spearman": within, "per_seed_across_arms": per_seed,
                      "note": "descriptive; the pooled 60-run p-values of the full-scale runs are pseudo-replicated"}
    print("MECH within-arm", json.dumps(within))
    print("MECH per-seed", json.dumps(per_seed))


# ------------------------------------------------------------------------------------------------ D.12 paper protocol (/)
def part_d12(out):
    rows, tests = d12_table("full_scr_d12", {})
    h = holm({k: one_sided(v, False) for k, v in tests.items()})
    rows_rs, _ = d12_table("full_scr_d12_rs", {})
    rows_g2, _ = d12_table("small_scr_d12", {})
    n = len(load("full_scr_d12"))
    for a in rows:
        r = rows[a]
        r["strict5"] = strict(r["ours"][0], r["paper"]) if r["paper"] else r["judge"]
        r["d_abs"] = None if r["paper"] is None else r["ours"][0] - r["paper"]
        r["resample_1M"] = rows_rs[a]["ours"]
        r["resample_1M_judge"] = rows_rs[a]["judge"]
        r["small_10_streams"] = rows_g2[a]["ours"]
    out["d12_flip_100"] = {"n_streams": n, "rows": rows, "tests": tests, "holm_one_sided": h,
                           "counts": counts([r["judge"] for r in rows.values()]),
                           "counts_strict5": counts([r["strict5"] for r in rows.values()]),
                           "counts_resample_1M": counts([r["resample_1M_judge"] for r in rows.values()])}
    for a, r in rows.items():
        print(f"D12flip100 {a:36s} {r['ours'][0]:.3f}±{r['ours'][1]:.3f} paper {r['paper']} {r['judge']:9s} "
              f"strict {r['strict5']:9s} | rs1M {r['resample_1M'][0]:.3f} {r['resample_1M_judge']} | g2(10) {r['small_10_streams'][0]:.3f}")
    for k, v in tests.items():
        print(f"{k:40s} {v['diff_mean']:+.4f}±{v['diff_se']:.4f} holm1s {h[k]['p_holm']:.2g} wins(other>overall) {v['n'] - v['wins']}/{v['n']}")
    print("D12 counts", out["d12_flip_100"]["counts"], "strict", out["d12_flip_100"]["counts_strict5"])


# ------------------------------------------------------------------------------------------------ D.12 sensitivity
def whole(tag, arm, seeds):
    runs = load(tag)
    return np.asarray([np.mean(runs[s]["arms"][arm]["mse_per_10k"]) for s in seeds])


def part_d12sens(out):
    seeds = list(range(30))
    ref = "full_scr_d12"
    res = {}
    paper = D12["overall"]
    for tag, pairs in (
            ("full_extra_d12_sens_default", [(f"overall_bt1_{r}", ref, f"overall_{r}") for r in paper] +
             [(f"overall_m1000_{r}", ref, f"overall_{r}") for r in paper] + [("bp", ref, "bp")]),
            ("full_extra_d12_sens_beta2", [(f"overall_{r}", ref, f"overall_{r}") for r in paper] +
             [(f"overall_m1000_{r}", ref, f"overall_{r}") for r in paper] + [("bp", ref, "bp")]),
            ("full_extra_d12_sens_convP", [(f"overall_{r}", ref, f"overall_{r}") for r in paper] + [("bp", ref, "bp")])):
        if not load(tag):
            continue
        res[tag] = {}
        for arm, rtag, rarm in pairs:
            a, b = whole(tag, arm, seeds), whole(rtag, rarm, seeds)
            pc = paired_compare(a, b)
            r = arm.split("_")[-1]
            p = D12["bp"]["0.0001"] if arm == "bp" else paper[r]
            res[tag][arm] = {"ours": mean_se(a), "default_same_streams": mean_se(b), "diff": [pc["diff_mean"], pc["diff_se"]],
                             "boot95": pc["boot95"], "p_two_sided": pc["t_p"], "paper": p,
                             "judge": judge(float(a.mean()), p, 0.01 + mean_se(a)[1])}
            print(f"SENS {tag:22s} {arm:22s} {a.mean():.3f} vs default {b.mean():.3f} diff {pc['diff_mean']:+.4f}±{pc['diff_se']:.4f} "
                  f"p {pc['t_p']:.2g} | paper {p} {res[tag][arm]['judge']}")
    out["d12_sens"] = res


# ------------------------------------------------------------------------------------------------ B.10 paper-text
def part_b10pt(out):
    tag = "full_extra_scr_b10_papertext"
    seeds = list(range(30))
    t = {}
    for a in B10:
        pp, bins = bins_of(tag, a, seeds)
        pp0, bins0 = bins_of("full_scr_b10", a, seeds)
        pf, pl = B10_FIRST_DIGITISED[a], B10[a][1]
        fb, lb, fb0, lb0 = mean_se(bins[:, 0]), mean_se(bins[:, -1]), mean_se(bins0[:, 0]), mean_se(bins0[:, -1])
        t[a] = {"first": fb, "last": lb, "first_official_same_streams": fb0, "last_official_same_streams": lb0,
                "paper_first": pf, "paper_last": pl,
                "judge_first": judge(fb[0], pf, READ_PREC_B10 + fb[1]),
                "judge_first_official": judge(fb0[0], pf, READ_PREC_B10 + fb0[1]),
                "judge_last": judge(lb[0], pl, READ_PREC_B10 + lb[1]) if pl else ("MATCH" if lb[0] > 1.2 else "DIVERGENT"),
                "judge_last_official": judge(lb0[0], pl, READ_PREC_B10 + lb0[1]) if pl else ("MATCH" if lb0[0] > 1.2 else "DIVERGENT"),
                "lop_last250k_minus_first_bin": mean_se(pp[:, -25:].mean(1) - bins[:, 0])}
        print(f"B10pt {a:17s} first {fb[0]:.3f} (official-20bit same streams {fb0[0]:.3f}) paper {pf} {t[a]['judge_first']}/"
              f"{t[a]['judge_first_official']} | last {lb[0]:.3f} ({lb0[0]:.3f}) paper {pl} {t[a]['judge_last']}/{t[a]['judge_last_official']}")
    out["b10_papertext"] = {"rows": t, "n_streams": 30,
                            "counts": {k: counts([r[k] for r in t.values()]) for k in
                                       ("judge_first", "judge_first_official", "judge_last", "judge_last_official")}}
    print("B10pt counts", json.dumps(out["b10_papertext"]["counts"]))


# ------------------------------------------------------------------------------------------------ MNIST ( / /)
def mnist_G(tag):
    runs = load(tag)
    seeds = sorted(runs)
    return runs, seeds, (lambda a, k: np.asarray([runs[s]["arms"][a][k] for s in seeds]))


def part_mnist(out):
    """Width-100 baselines ( / / /).

    Selection evidence: the review's CPU L2 sweep (runs/full_extra_review/rr_mnist_l2_sweep, seeds 0-2, 15 tasks) next to
    the stored small-scale arms on the same seeds.  Held-out check: full_extra_mnist_final15 (seeds 1000-1004, tasks 1-15; L2 1e-3
    and CBP 1e-5 + bias transfer, CPU) paired with the stored full_mnist arms (GPU) on the same seeds and tasks.
   -M (Holm over 3, two-sided): mean online accuracy over tasks 11-15.
    """
    res = {}
    rt = os.path.join("full_extra_review", "rr_mnist_l2_sweep")
    sweep = {}
    for tag in (rt, "small_mnist"):
        runs, seeds, G = mnist_G(tag)
        for a in runs[seeds[0]]["arms"]:
            acc = G(a, "acc")[:, :15]
            sweep[f"{a} ({'review CPU' if tag == rt else 'small-scale GPU'})"] = {
                "seeds": seeds, "mean_acc_15": mean_se(acc.mean(1)), "last5": mean_se(acc[:, -5:].mean(1)),
                "task15": mean_se(acc[:, 14]), "dead_t15": mean_se(G(a, "dead_pct")[:, 14])}
    res["sweep_seeds_0_2"] = sweep
    for k, v in sweep.items():
        print(f"SWEEP {k:28s} mean15 {v['mean_acc_15'][0]:.5f} last5 {v['last5'][0]:.5f} t15 {v['task15'][0]:.4f} dead {v['dead_t15'][0]:.1f}")
    runs, seeds, G = mnist_G("full_extra_mnist_final15")
    g3, s3, G3 = mnist_G("full_mnist")
    idx = [s3.index(s) for s in seeds]
    A = {"l2_1e-3 (CPU)": G("l2_1e-3", "acc")[:, :15], "cbp_bt1_1e-5 (CPU)": G("cbp_bt1_1e-5", "acc")[:, :15]}
    for a in ("bp", "l2", "snp", "cbp_1e-4", "cbp_1e-5", "cbp_1e-6"):
        A[f"{a} (stored GPU)"] = G3(a, "acc")[idx, :15]
    m1115 = {a: v[:, 10:15].mean(1) for a, v in A.items()}
    m115 = {a: v.mean(1) for a, v in A.items()}
    fam = {"M1 L2(1e-3) - CBP(1e-5)": paired_compare(m1115["l2_1e-3 (CPU)"], m1115["cbp_1e-5 (stored GPU)"]),
           "M2 CBP(1e-5, bt) - CBP(1e-5)": paired_compare(m1115["cbp_bt1_1e-5 (CPU)"], m1115["cbp_1e-5 (stored GPU)"]),
           "M3 L2(1e-3) - L2(1e-4)": paired_compare(m1115["l2_1e-3 (CPU)"], m1115["l2 (stored GPU)"])}
    h = holm({k: v["t_p"] for k, v in fam.items()})
    for k in fam:
        fam[k]["p_holm"] = h[k]["p_holm"]
    desc = {"L2(1e-3) - CBP(1e-5), tasks 1-15": paired_compare(m115["l2_1e-3 (CPU)"], m115["cbp_1e-5 (stored GPU)"]),
            "L2(1e-3) - CBP(1e-4), tasks 11-15": paired_compare(m1115["l2_1e-3 (CPU)"], m1115["cbp_1e-4 (stored GPU)"]),
            "L2(1e-3) - S&P(1e-4,1e-5), tasks 11-15": paired_compare(m1115["l2_1e-3 (CPU)"], m1115["snp (stored GPU)"]),
            "L2(1e-3) - BP, tasks 11-15": paired_compare(m1115["l2_1e-3 (CPU)"], m1115["bp (stored GPU)"]),
            "CBP(1e-5,bt) - CBP(1e-5), tasks 1-15": paired_compare(m115["cbp_bt1_1e-5 (CPU)"], m115["cbp_1e-5 (stored GPU)"])}
    gap = (A["l2_1e-3 (CPU)"] - A["cbp_1e-5 (stored GPU)"]).mean(0)
    arms = {a: {"acc_task1": mean_se(v[:, 0]), "mean_1_15": mean_se(m115[a]), "mean_11_15": mean_se(m1115[a]),
                "acc_curve": v.mean(0).tolist()} for a, v in A.items()}
    for a in ("l2_1e-3", "cbp_bt1_1e-5"):
        arms[f"{a} (CPU)"].update({"dead_t15": mean_se(G(a, "dead_pct")[:, 14]), "erank_t15": mean_se(G(a, "erank_scaled")[:, 14]),
                                   "wmag_stepmean_t1": mean_se(G(a, "wmag_mean")[:, 0]),
                                   "wmag_stepmean_t15": mean_se(G(a, "wmag_mean")[:, 14]),
                                   "wmag_after_t15": mean_se(G(a, "wmag_after")[:, 14])})
    res["final15"] = {"seeds": seeds, "arms": arms, "family_R35M": fam, "descriptive": desc,
                      "gap_l2_1e-3_minus_cbp_1e-5_per_task_pp": (100 * gap).tolist()}
    for a, v in arms.items():
        print(f"FINAL15 {a:24s} t1 {v['acc_task1'][0]:.4f} mean1-15 {v['mean_1_15'][0]:.5f} mean11-15 {v['mean_11_15'][0]:.5f}±{v['mean_11_15'][1]:.5f}")
    for k, v in list(fam.items()) + list(desc.items()):
        print(f"{k:42s} {100 * v['diff_mean']:+.3f}±{100 * v['diff_se']:.3f} pp boot {np.round(100 * np.asarray(v['boot95']), 3)} "
              f"p {v['t_p']:.2g} {('holm %.2g' % v['p_holm']) if 'p_holm' in v else ''} wins {v['wins']}/{v['n']}")
    print("gap per task (pp)", np.round(100 * gap, 2))
    print("wmag", {a: arms[a].get("wmag_stepmean_t15") for a in arms if "CPU" in a})
    out["mnist35"] = res


PARTS = {"b10": part_b10, "lop": part_lop, "mech": part_mech, "d12": part_d12, "d12sens": part_d12sens,
         "b10pt": part_b10pt, "mnist": part_mnist}


def main(argv):
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for p in (argv or list(PARTS)):
        try:
            PARTS[p](out)
        except (KeyError, FileNotFoundError, IndexError, ValueError) as e:
            print(f"[WARN] part {p} skipped: {type(e).__name__}: {e}")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1:])
