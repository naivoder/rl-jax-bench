"""Sticky-actions run analysis (the rule was declared BEFORE the run and is repeated here).
H1a_sticky = EARLY(JSRL) - EARLY(PDD), pi_e alone, points 1..10, per seed 20..29 (sticky 0.1, inherited default.yaml).
Reference = the 20 INDEPENDENT inherited per-seed H1a values: the small-scale run seeds 0..9 (runs/small/Breakout) and the fresh-seed run seeds 10..19 (runs/verify/Breakout_r3);
the tolerance/window run reused the small-scale run's seeds and is excluded.  The roll-in-collapse reading of the no-sticky-actions assumption is SUPPORTED iff Welch one-sided p < 0.05 that H1a_sticky > H1a_inherited AND
mean shift >= +0.5; otherwise that assumption is CLEARED as a cause of the early deficit.  Confound declared in advance: sticky actions also weaken the guide
(4.83 vs 6.78) and shorten it (H1 = 58.7).  Everything else (C3-final, FINAL levels, cap fraction, hand-over share) is descriptive.
usage: recheck_sticky.py runs/recheck/fix_sticky/Breakout OUT.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats as sps
from rl_jax_bench import stats as S
from src import eval as E

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def per_seed(d, arms=("JSRL", "PDD")):
    runs, meta = E.load(d, "Breakout"); n = meta["n_eval_points"]; ea, fi = E.windows(n)
    seeds = sorted(runs["JSRL"])
    g = lambda a, k, sl: np.array([np.mean(np.asarray(runs[a][s][k], float)[sl]) for s in seeds])
    return runs, meta, seeds, ea, fi, g


if __name__ == "__main__":
    d, outp = sys.argv[1:3]
    runs, meta, seeds, ea, fi, g = per_seed(d)
    h1a = g("JSRL", "ev_alone", ea) - g("PDD", "ev_alone", ea)
    inh = []
    for dd in ("runs/small/Breakout", "runs/verify/Breakout_r3"):
        r, m, sd, e2, f2, gg = per_seed(os.path.join(REPO, dd))
        inh.append(gg("JSRL", "ev_alone", e2) - gg("PDD", "ev_alone", e2))
    inh = np.concatenate(inh)
    w = sps.ttest_ind(h1a, inh, equal_var=False)
    p1 = w.pvalue / 2 if h1a.mean() > inh.mean() else 1 - w.pvalue / 2
    shift = float(h1a.mean() - inh.mean())
    res = dict(seeds=seeds, H1a_sticky=S.mean_se(h1a), H1a_sticky_wins=int((h1a > 0).sum()), H1a_inherited_20=S.mean_se(inh), shift=shift, welch_p_one_sided=float(p1),
               welch_t=float(w.statistic), supported=bool(p1 < 0.05 and shift >= 0.5),
               verdict="no-sticky roll-in-collapse reading SUPPORTED" if (p1 < 0.05 and shift >= 0.5) else "no-sticky assumption CLEARED as a cause of the early deficit")
    fin = lambda a: g(a, "ev_alone", fi)
    res["C3final_sticky"] = dict(mean_se=S.mean_se(fin("JSRL") - fin("PDD")), wins=int((fin("JSRL") > fin("PDD")).sum()))
    res["FINAL"] = {a: S.mean_se(fin(a)) for a in ("JSRL", "PDD")}
    res["EARLY"] = {a: S.mean_se(g(a, "ev_alone", ea)) for a in ("JSRL", "PDD")}
    res["H1c_combined_vs_PDD"] = S.mean_se(g("JSRL", "ev_thr", ea) - g("PDD", "ev_alone", ea))
    res["mu_g_sticky"] = meta["guide"]["true_mean"]; res["stage_list"] = meta["guide"]["stage_list"]
    res["JSRL_minus_mu_g"] = S.mean_se(fin("JSRL") - meta["guide"]["true_mean"]); res["PDD_minus_mu_g"] = S.mean_se(fin("PDD") - meta["guide"]["true_mean"])
    res["JSRL_stage_at_point10"] = [int(np.asarray(runs["JSRL"][s]["ev_stage"])[10]) for s in seeds]
    res["guide_share"] = float(np.mean([runs["JSRL"][s]["guide_steps"] / meta["steps"] for s in seeds]))
    if "ev_cap" in runs["JSRL"][seeds[0]]:
        res["ev_cap_final_window_mean"] = {a: float(np.mean([np.asarray(runs[a][s]["ev_cap"], float)[fi].mean() for s in seeds])) for a in ("JSRL", "PDD")}
    json.dump(res, open(outp, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
