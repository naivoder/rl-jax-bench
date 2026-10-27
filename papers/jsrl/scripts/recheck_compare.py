"""before/after (written and committed BEFORE the amended-learner run / sticky run numbers were read; all numbers computed from metrics.json).
usage: recheck_compare.py INHERITED_DIR AMENDED_DIR OUT.json [GAME]
INHERITED = the original run (the small-scale run, runs/small/Breakout, seeds 0..9, default.yaml); AMENDED = the same seeds under learner_minatar.yaml
(the amended-learner run).  Cross-program comparison: the two configurations are different programs, so deltas carry 'chaos + batch composition' noise and
are PAIRED BY SEED for the t-test only as a convenience (the seeds share init keys and env keys, not trajectories after the first updates).
Declared rules are NOT replaced: the graded members (H1c, H1a/H1m/H1p, H2, H3, S-C4early/final, S-C2) come from src/eval.analyse unchanged for each
configuration; everything here is descriptive."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import stats as sps
from rl_jax_bench import stats as S
from src import eval as E


def tr_curve(r):
    bs, bc = np.asarray(r["bin_sum"], float), np.asarray(r["bin_cnt"], float)
    return np.where(bc > 0, bs / np.maximum(bc, 1), np.nan)


def pc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    t = sps.ttest_rel(a, b) if len(d) > 1 else None
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None, wins=int((d > 0).sum()), n=len(d),
                p2=float(t.pvalue) if t is not None else None)


def first_step(curve, thr, chunk, start=0):
    idx = np.nonzero(np.asarray(curve)[start:] >= thr)[0]
    return None if len(idx) == 0 else int((idx[0] + start) * chunk)


def summarize(d, game):
    runs, meta = E.load(d, game)
    n = meta["n_eval_points"]; chunk = meta["steps"] // n; mu = meta["guide"]["true_mean"]
    ea, fi = E.windows(n)
    seeds = sorted(runs["JSRL"])
    al = lambda a, sl: np.array([np.mean(np.asarray(runs[a][s]["ev_alone"], float)[sl]) for s in seeds])
    th = lambda a, sl: np.array([np.mean(np.asarray(runs[a][s]["ev_thr"], float)[sl]) for s in seeds])
    trn = lambda a: np.array([np.nanmean(tr_curve(runs[a][s])[0:10]) for s in seeds])
    out = dict(dir=d, n=len(seeds), seeds=seeds, mu_g=mu, game=game, cfg_optim=meta["cfg"]["optim"], sticky=meta.get("sticky", 0.0))
    out["analysis"] = E.analyse(d, game)
    out["arm_scores"] = {a: dict(early=al(a, ea).tolist(), final=al(a, fi).tolist(), r0=[float(np.asarray(runs[a][s]["ev_alone"])[0]) for s in seeds]) for a in runs if all(s in runs[a] for s in seeds)}
    ex = {}
    if "PDD" in runs:
        ex["H1a"] = pc(al("JSRL", ea), al("PDD", ea))
        ex["H1c_combined_vs_PDD"] = pc(th("JSRL", ea), al("PDD", ea))
        ex["H1_reading_C_training_eps"] = pc(trn("JSRL"), trn("PDD"))
        ex["C3final"] = pc(al("JSRL", fi), al("PDD", fi))
        ex["PDD_minus_mu_g"] = pc(al("PDD", fi), np.full(len(seeds), mu))
        ex["JSRL_minus_mu_g"] = pc(al("JSRL", fi), np.full(len(seeds), mu))
    if "DQfD" in runs:
        ex["H3_JSRL_minus_DQfD"] = pc(al("JSRL", fi), al("DQfD", fi))
    if "JSRL-Random" in runs:
        ex["S-C4early_A_alone"] = pc(al("JSRL", ea), al("JSRL-Random", ea))
        ex["C4early_B_combined_vs_Random_alone"] = pc(th("JSRL", ea), al("JSRL-Random", ea))
        ex["C4early_C_training_eps"] = pc(trn("JSRL"), trn("JSRL-Random"))
        ex["C4final"] = pc(al("JSRL", fi), al("JSRL-Random", fi))
    if "BCinit-FT" in runs:
        r0 = np.array(out["arm_scores"]["BCinit-FT"]["r0"])
        ex["C2_persistence_FINAL_minus_R0"] = pc(al("BCinit-FT", fi), r0)
        if "PDD" in runs:
            ex["C2_FINAL_BCinitFT_minus_PDD"] = pc(al("BCinit-FT", fi), al("PDD", fi))
        ex["C2_dip_min_EARLY_minus_R0"] = pc(np.array(out["arm_scores"]["BCinit-FT"].get("early")) * 0 + np.array([np.min(np.asarray(runs["BCinit-FT"][s]["ev_alone"], float)[ea]) for s in seeds]), r0)
    # Small-5 under both readings, from the seed-mean curves
    mc = lambda a, k: np.mean([np.asarray(runs[a][s][k], float) for s in seeds], 0)
    g = {}
    if "PDD" in runs:
        for nm, thr in (("0.5_mu_g", 0.5 * mu), ("1.0_mu_g", mu)):
            p = first_step(mc("PDD", "ev_alone"), thr, chunk, 1)
            ja = first_step(mc("JSRL", "ev_alone"), thr, chunk, 1)
            jc = first_step(mc("JSRL", "ev_thr"), thr, chunk, 1)
            g[nm] = dict(PDD=p, JSRL_alone=ja, JSRL_combined=jc, ratio_strict=(ja / p if ja and p else None), ratio_paper_look=(jc / p if jc and p else None))
    out["Small_5"] = g
    # stage reached / first h=0 and cap fractions
    js = runs["JSRL"]
    out["JSRL_first_eval_at_h0"] = [js[s].get("first_eval_at_h0") for s in seeds]
    out["JSRL_stage_at_point10"] = [int(np.asarray(js[s]["ev_stage"])[10]) for s in seeds]
    out["guide_share"] = {a: float(np.mean([runs[a][s]["guide_steps"] / meta["steps"] for s in seeds])) for a in ("JSRL", "JSRL-Random", "PDD") if a in runs}
    cap = {}
    for a in runs:
        if "ev_cap" in runs[a][seeds[0]]:
            cs = np.array([np.asarray(runs[a][s]["ev_cap"], float) for s in seeds])
            cap[a] = dict(final_window_mean=float(cs[:, fi].mean()), final_window_max=float(cs[:, fi].max()), whole_run_mean=float(cs[:, 1:].mean()), n_runs_ever_capped=int((cs[:, 1:].max(1) > 0).sum()))
    out["ev_cap"] = cap
    # return levels
    out["curve_means_alone"] = {a: {str(p): float(mc(a, "ev_alone")[int(round(p / 100 * n))]) for p in (0, 5, 10, 25, 50, 75, 100)} for a in runs}
    out["extras"] = ex
    return out


if __name__ == "__main__":
    inh, amd, outp = sys.argv[1:4]
    game = sys.argv[4] if len(sys.argv) > 4 else "Breakout"
    I, A = summarize(inh, game), summarize(amd, game)
    res = dict(inherited=I, amended=A, before_after={})
    for arm in A["arm_scores"]:
        if arm in I["arm_scores"] and A["seeds"] == I["seeds"]:
            for m in ("early", "final", "r0"):
                res["before_after"][f"{arm}/{m}"] = dict(inherited=S.mean_se(np.array(I["arm_scores"][arm][m])), amended=S.mean_se(np.array(A["arm_scores"][arm][m])),
                                                         delta=pc(A["arm_scores"][arm][m], I["arm_scores"][arm][m]))
    if A["seeds"] == I["seeds"]:
        for k in ("H1a", "C3final", "PDD_minus_mu_g", "JSRL_minus_mu_g", "H3_JSRL_minus_DQfD", "S-C4early_A_alone", "C4final"):
            if k in I["extras"] and k in A["extras"]:
                # difference-in-differences across the two configurations: (amended stat) - (inherited stat), per seed
                a_s, i_s = A, I
        # diff-in-diff of C3final and H1a per seed
        def per(summ, arm, m): return np.array(summ["arm_scores"][arm][m])
        for nm, (x, y, m) in {"C3final": ("JSRL", "PDD", "final"), "H1a": ("JSRL", "PDD", "early")}.items():
            dA = per(A, x, m) - per(A, y, m); dI = per(I, x, m) - per(I, y, m)
            res["before_after"][f"diff_in_diff/{nm}"] = pc(dA, dI)
    json.dump(res, open(outp, "w"), indent=1, default=float)
    def f(t): return "n/a" if t is None else f"{t['mean']:+.3f} +- {t['se']:.3f} ({t['wins']}/{t['n']}, p2 {t['p2']:.2g})" if t.get("se") is not None else str(t)
    for tag, X in (("INHERITED", I), ("AMENDED", A)):
        print(f"== {tag} ({X['dir']}, n={X['n']}, optim={X['cfg_optim']['lr']}/{X['cfg_optim']['target_update_period']}, sticky {X['sticky']})")
        for k, v in X["extras"].items(): print("  ", k, f(v))
        print("   Small-5:", json.dumps(X["Small_5"])); print("   stage@10:", X["JSRL_stage_at_point10"], "first h0:", X["JSRL_first_eval_at_h0"]); print("   ev_cap:", json.dumps(X["ev_cap"])); print("   guide share:", X["guide_share"])
        print("   curve means:", json.dumps(X["curve_means_alone"]))
        for k in ("H1c", "H2", "H3"):
            if k in X["analysis"]["primary"]: t = X["analysis"]["primary"][k]; print("   primary", k, {a: t.get(a) for a in ("diff_mean", "diff_se", "p_one_sided", "p_two_sided", "wins")})
        for k, t in X["analysis"].get("strict", {}).items(): print("   strict", k, {a: t.get(a) for a in ("diff_mean", "diff_se", "p_one_sided", "wins")})
        for k, t in X["analysis"]["secondary"].items(): print("   secondary", k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in t.items() if a in ("diff_mean", "diff_se", "p", "p_one_sided", "wins", "collapse_mean", "rel_collapse", "r0_mean", "p_one_sided_less")})
    print("== BEFORE/AFTER (amended - inherited, cross-program; paired by seed)")
    for k, v in res["before_after"].items():
        if "delta" in v: print(f"  {k:28s} inh {v['inherited'][0]:7.3f}+-{v['inherited'][1]:.3f}  amd {v['amended'][0]:7.3f}+-{v['amended'][1]:.3f}  delta {f(v['delta'])}")
        else: print(f"  {k:28s} {f(v)}")
