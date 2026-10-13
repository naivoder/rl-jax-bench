"""Full-fp32 stage report (pre-declared in the README before launch).

Pendulum-V sac-lstm-200-oar, 51k env steps, fresh seeds 8-11, fp32 ('highest') in every arm.
Confirmatory family C (one-sided paired t, Holm m = 3): C1 separate > official, C2 detach > official,
C3 separate > noleak.  Equivalence: detach - separate, TOST alpha 0.05 (paired 90 % t-interval inside +-25).
Level rows vs the authors' logs (raw relative error, gap in the logs' sd / se; src.eval.judge_detail).
Descriptive: separate 16k-20k window vs the official code on our CPU (m7); fp32 (8-11) vs TF32 (0-3) unpaired
(precision sensitivity); S-Grad ratio on the fresh seeds in the official metric and the caption's squared metric.
Writes runs/full_fp32/report.json and runs/seed_<n>/metrics.json for n = 8..11.
"""
import json
import os
import sys

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_jax_bench.stats import holm, mean_se, paired_compare, t_interval  # noqa: E402

from src.eval import author_final, final_value, judge_detail, load_run  # noqa: E402


def perm_p(a, b):
    """exact two-sided permutation test on the difference of means (same as official_reference_report.perm_p)."""
    import itertools
    a, b = np.asarray(a, float), np.asarray(b, float)
    pool, na = np.concatenate([a, b]), len(a)
    obs = abs(a.mean() - b.mean())
    cnt = tot = 0
    for idx in itertools.combinations(range(len(pool)), na):
        m = np.zeros(len(pool), bool)
        m[list(idx)] = True
        cnt += abs(pool[m].mean() - pool[~m].mean()) >= obs - 1e-9
        tot += 1
    return cnt / tot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G3, G3B = os.path.join(REPO, "runs", "full"), os.path.join(REPO, "runs", "full_fp32")
LOGP = "pomdp/Pendulum/V/{a}/len--1/*/*/*/oar/*/progress.csv"
RAND = json.load(open(os.path.join(G3, "random_policy.json")))
BAND = 25.0


def finals(run_dir, variant):
    d = load_run(run_dir).get(variant, {})
    for m in d.values():
        assert m["final_metric_key"] == "final_last20" and m["total_env_steps_run"] == 51000, m["final_metric_key"]
    return {s: final_value(d[s]) for s in sorted(d)}, d


def one_sided(a, b):
    r = paired_compare(a, b)
    t = st.ttest_rel(a, b)
    r["p_one_sided"] = float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2)
    return r


def window(m, lo, hi):
    s, r = np.asarray(m["eval_env_steps"]), np.asarray(m["eval_returns"])
    sel = (s >= lo) & (s <= hi)
    return float(r[sel].mean())


def med(m, k):
    v = [t[k] for t in m["train"] if k in t]
    return float(np.median(v)) if v else None


def unpaired(a, b):
    return {"diff": float(np.mean(a) - np.mean(b)), "welch_p": float(st.ttest_ind(a, b, equal_var=False).pvalue),
            "perm_p": perm_p(a, b), "a": mean_se(a), "b": mean_se(b)}


def main():
    rep = {"seeds": [8, 9, 10, 11], "precision": "highest (fp32), every arm", "horizon": "51k (paper)"}
    sep, dsep = finals(os.path.join(G3B, "P1b_sac_separate"), "separate")
    arms = {"separate": sep}
    dsh = {}
    for v in ("official", "noleak", "detach"):
        arms[v], dsh[v] = finals(os.path.join(G3B, "P1b_sac_shared"), v)
    S = sorted(set.intersection(*[set(a) for a in arms.values()]))
    assert S == rep["seeds"], S
    x = {k: [v[s] for s in S] for k, v in arms.items()}
    rep["per_seed"] = x
    rep["means"] = {k: mean_se(v) for k, v in x.items()}

    # ---- confirmatory family C
    C = {"C1_separate_gt_official": one_sided(x["separate"], x["official"]),
         "C2_detach_gt_official": one_sided(x["detach"], x["official"]),
         "C3_separate_gt_noleak": one_sided(x["separate"], x["noleak"])}
    rep["family_C"] = C
    rep["family_C_holm"] = holm({k: v["p_one_sided"] for k, v in C.items()})
    rep["family_C_pass"] = {k: bool(rep["family_C_holm"][k]["p_holm"] < 0.05) for k in C}
    d = np.array(x["detach"]) - np.array(x["separate"])
    ci90, ci95 = t_interval(d, 0.90), t_interval(d, 0.95)
    rep["equivalence_detach_minus_separate"] = {
        "diff_mean": float(d.mean()), "diff_se": mean_se(d)[1], "t90": ci90, "t95": ci95, "band": BAND,
        "tost_equivalent_at_0.05": bool(-BAND < ci90[0] and ci90[1] < BAND), "per_seed": d.tolist()}

    # ---- level rows vs the authors' logs (51k, last 20 %)
    a_sep = author_final(LOGP.format(a="sac_lstm"))
    a_sh = author_final(LOGP.format(a="sac_lstm_shared"))
    lo, hi = RAND["Pendulum-V"]["mean"], 0.0
    rep["levels"] = {
        "separate_vs_logs_separate": judge_detail(np.mean(x["separate"]), a_sep, lo, hi),
        "official_vs_logs_shared": judge_detail(np.mean(x["official"]), a_sh, lo, hi),
        "noleak_vs_logs_shared": judge_detail(np.mean(x["noleak"]), a_sh, lo, hi),
        "detach_vs_logs_separate": judge_detail(np.mean(x["detach"]), a_sep, lo, hi)}
    dif = np.array(x["separate"]) - np.array(x["official"])
    rep["levels"]["gap_separate_minus_official"] = {
        "ours_paired": mean_se(dif), "logs_unpaired": float(np.mean(a_sep) - np.mean(a_sh)),
        "raw_rel_pct": 100.0 * (dif.mean() - (np.mean(a_sep) - np.mean(a_sh))) / abs(np.mean(a_sep) - np.mean(a_sh))}

    # ---- descriptive: separate 16k-20k window vs the official code on our CPU (review)
    p = os.path.join(REPO, "runs", "official_ref", "report.json")
    off_sep = json.load(open(p))["separate"]["agg"]["official_v0_ourhw"]["window_seeds"]
    off_sh = json.load(open(p))["shared"]["agg"]["official_v0_ourhw"]["window_seeds"]
    w_sep = [window(dsep[s], 16000, 20000) for s in S]
    w_off = [window(dsh["official"][s], 16000, 20000) for s in S]
    a_sep_w = author_final(LOGP.format(a="sac_lstm"), window=(16000, 20000))
    a_sh_w = author_final(LOGP.format(a="sac_lstm_shared"), window=(16000, 20000))
    rep["window_16_20k"] = {"separate_fp32": w_sep, "official_fp32": w_off,
                            "separate_vs_official_code": unpaired(w_sep, off_sep),
                            "separate_vs_logs": unpaired(w_sep, a_sep_w),
                            "shared_official_vs_official_code": unpaired(w_off, off_sh),
                            "shared_official_vs_logs": unpaired(w_off, a_sh_w)}

    # ---- descriptive: precision sensitivity (fp32 seeds 8-11 vs TF32 seeds 0-3; unpaired, different seeds)
    t_sep, _ = load_run(os.path.join(G3, "P1_sac_separate")).get("separate"), None
    t_sh = load_run(os.path.join(G3, "P1_sac_shared"))
    tf = {"separate": [final_value(m) for _, m in sorted(t_sep.items())],
          "official": [final_value(m) for _, m in sorted(t_sh["official"].items())],
          "detach": [final_value(m) for _, m in sorted(t_sh["detach"].items())]}
    rep["precision_fp32_vs_tf32"] = {k: unpaired(x[k], tf[k]) for k in tf}
    rep["precision_fp32_vs_tf32"]["gap"] = unpaired(list(dif), [a - b for a, b in zip(tf["separate"], tf["official"])])

    # ---- S-Grad on the fresh seeds: official unsquared metric and the caption's squared global norm
    g = {}
    for name, k_sh, k_sep in (("official_metric", "off_rnn", "off_q_rnn"), ("squared_global", "off_sq_rnn", "off_sq_q_rnn")):
        sh = [med(dsh["official"][s], k_sh) for s in S]
        sp = [med(dsep[s], k_sep) for s in S]
        if None in sh or None in sp:
            continue
        lr = np.log10(np.array(sh) / np.array(sp))
        t = st.ttest_1samp(lr, 0.0)
        g[name] = {"shared_median": sh, "separate_critic_median": sp, "ratio_geomean": float(10 ** lr.mean()),
                   "log10_ratio_t95": t_interval(lr),
                   "p_one_sided_ratio_gt_1": float(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2)}
        if name == "squared_global":
            g[name]["separate_actor_median"] = [med(dsep[s], "off_sq_pi_rnn") for s in S]
    g["clean_pi_vs_q_into_shared_rnn"] = {v: {"clean_pi_rnn": [med(dsh[v][s], "clean_pi_rnn") for s in S],
                                              "clean_q_rnn": [med(dsh[v][s], "clean_q_rnn") for s in S]}
                                          for v in ("official", "noleak", "detach")}
    rep["S_Grad_fresh_seeds"] = g

    # ---- timing
    rep["timing"] = {"separate": dsep[S[0]]["timing"], "shared": dsh["official"][S[0]]["timing"]}

    with open(os.path.join(G3B, "report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    for s in S:
        pth = os.path.join(REPO, "runs", f"seed_{s}", "metrics.json")
        agg = {"seed": s, "stage": "full_fp32", "metric_key": "final_last20 (51k = paper horizon), fp32",
               "full_fp32_cells": {f"P1b_sac_51k_fp32/{k}": arms[k][s] for k in arms},
               "full_fp32_curves": {f"P1b_sac_51k_fp32/{k}": {"env_steps": (dsep if k == "separate" else dsh[k])[s]["eval_env_steps"],
                                                           "returns": (dsep if k == "separate" else dsh[k])[s]["eval_returns"]}
                                 for k in arms}}
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        json.dump(agg, open(pth, "w"), indent=1)

    # console
    for k, (m, se) in rep["means"].items():
        print(f"{k:9s} {m:9.1f} ± {se:6.1f}  per seed {np.round(x[k], 1).tolist()}")
    for k, v in C.items():
        print(k, f"diff {v['diff_mean']:.1f} ± {v['diff_se']:.1f} t95 {np.round(v['t95'], 1).tolist()} "
                 f"p1 {v['p_one_sided']:.4g} holm {rep['family_C_holm'][k]['p_holm']:.4g} wins {v['wins']}")
    print("equiv", json.dumps(rep["equivalence_detach_minus_separate"], default=float))
    for k, v in rep["levels"].items():
        print(k, json.dumps(v, default=float))
    for k, v in rep["window_16_20k"].items():
        print("16-20k", k, json.dumps(v, default=float) if not isinstance(v, list) else np.round(v, 1).tolist())
    for k, v in rep["precision_fp32_vs_tf32"].items():
        print("prec", k, json.dumps(v, default=float))
    print("grad", json.dumps(g, default=float)[:2000])


if __name__ == "__main__":
    main()
