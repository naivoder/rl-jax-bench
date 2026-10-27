"""Fresh-seed run analysis (verification run on FRESH seeds 10..19; the rule was declared BEFORE launch and this docstring repeats it).

Arms (Breakout, 200 k steps, one process): PDD, JSRL, JSRL+eps_own=1 (eps annealed on pi_e's own steps),
PDD+lam_n=0 and JSRL+lam_n=0 (pure 1-step returns: the 10-step uncorrected returns on guide segments),
JSRL+eps_own=1+lam_n=0 (both).  EARLY = mean over points 1..10 of pi_e alone (32 episodes, eps 0.01); FINAL = points 98..100; mu_g = 6.784.
Implementation-suspect family (Holm over 3; paired by seed; one-sided > 0; GO iff Holm p < 0.05 and the mean difference > 0):
    W1 = EARLY(JSRL+eps_own=1) - EARLY(PDD)
    W2 = EARLY(JSRL+lam_n=0)  - EARLY(PDD+lam_n=0)
    W3 = EARLY(JSRL+eps_own=1+lam_n=0) - EARLY(PDD+lam_n=0)
  A GO would mean that the early pi_e-alone deficit of JSRL (the small-scale run H1a = -1.80, the tolerance/window run -1.75) is at least partly an implementation choice of OURS.
Independent replication of the proxy H1a on fresh seeds: REPLICATED iff the mean of EARLY(JSRL) - EARLY(PDD) is < 0 with two-sided p < 0.05;
  reported together with H1c, H1m, H1p (amendment A1), H2 (FINAL(JSRL) - mu_g) and PDD - mu_g, D-C3final (JSRL - PDD FINAL).
Descriptive (never graded): the warm-start pre-flight at the end of EARLY (`pf_mid_agreement`: greedy agreement of the Q net with the guide's labels on
  2,048 demo states), the guide share of steps, the first evaluation at h = 0, FINAL of every arm.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats as sps
from rl_jax_bench import stats as S
from src import eval as E


def vec(runs, arm, key, sl):
    return np.array([np.mean(np.asarray(runs[arm][s][key], float)[sl]) for s in sorted(runs[arm])])


def main(d, game="Breakout"):
    res = E.analyse(d, game)
    runs, meta = E.load(d, game)
    n = meta["n_eval_points"]; ea, fi = E.windows(n); mu = meta["guide"]["true_mean"]
    out = dict(replication=dict(H1a=res["strict"]["H1a"], H1c=res["primary"]["H1c"], H1m=res["strict"].get("H1m"), H1p=res["strict"].get("H1p"),
                                H2=res["primary"]["H2"], D_C3final=res["secondary"]["D-C3final"], arms=res["arms"]))
    a = res["strict"]["H1a"]
    out["replication"]["H1a_replicated_opposite_sign"] = bool(a["diff_mean"] < 0 and a["p_two_sided"] < 0.05)
    d_pdd = vec(runs, "PDD", "ev_alone", fi) - mu
    out["PDD_minus_mu_g"] = dict(mean=float(d_pdd.mean()), se=S.mean_se(d_pdd)[1], wins=int((d_pdd > 0).sum()), n=len(d_pdd),
                                 p_one_sided=float(sps.ttest_1samp(d_pdd, 0.0).pvalue / 2) if d_pdd.mean() > 0 else None)
    fam, fam_p = {}, {}
    for name, v, c in (("W1", "JSRL+eps_own=1", "PDD"), ("W2", "JSRL+lam_n=0", "PDD+lam_n=0"), ("W3", "JSRL+eps_own=1+lam_n=0", "PDD+lam_n=0")):
        if v in runs and c in runs:
            t = E.paired(vec(runs, v, "ev_alone", ea), vec(runs, c, "ev_alone", ea)); fam[name] = dict(t, variant=v, control=c); fam_p[name] = t["p_one_sided"]
    h = S.holm(fam_p) if fam_p else {}
    for k in fam:
        fam[k]["holm_p"] = h[k]["p_holm"]; fam[k]["GO"] = bool(fam[k]["diff_mean"] > 0 and h[k]["p_holm"] < 0.05)
    out["R3F"] = fam
    desc = {}
    for arm in runs:
        r = runs[arm]; ss = sorted(r)
        desc[arm] = dict(early=S.mean_se(vec(runs, arm, "ev_alone", ea)), final=S.mean_se(vec(runs, arm, "ev_alone", fi)), early_comb=S.mean_se(vec(runs, arm, "ev_thr", ea)),
                         guide_share=S.mean_se(np.array([r[s]["guide_steps"] / meta["steps"] for s in ss])),
                         first_h0=[r[s].get("first_eval_at_h0") for s in ss] if "first_eval_at_h0" in r[ss[0]] else None)
        if "pf_mid_agreement" in r[ss[0]]:
            desc[arm]["pf_mid_agreement"] = S.mean_se(np.array([r[s]["pf_mid_agreement"] for s in ss]))
            desc[arm]["pf_end_agreement"] = S.mean_se(np.array([r[s]["pf_end_agreement"] for s in ss]))
            desc[arm]["pf_mid_q_norm"] = S.mean_se(np.array([r[s]["pf_mid_q_norm"] for s in ss]))
    out["descriptive"] = desc
    json.dump(out, open(os.path.join(d, "..", "analysis_vr3.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/verify/Breakout_r3")
