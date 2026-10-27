"""Pre-declared statistics (see the README) from runs/<out>/seed_<n>/metrics.json.

Per seed and arm: R(j), j = 0..100, is the mean raw return of 32 pi_e-ALONE episodes at eps 0.01 (point 0 = before training).
    EARLY = mean R(1..10) (first 10 % of the budget)        FINAL = mean R(98..100) (last 3 %; selection-free, never best-window)
Primary family (Holm over the games flown x 3), one-sided unless stated, alpha = 0.05.  AMENDMENT A1 (written before
any full-scale number): the early test of the paper's sample-efficiency claim (C3 early) is graded on the policy the paper plots, the
COMBINED policy at the current curriculum stage (paper Sec. 4.2: 'the new combined policy is then evaluated over the course of training'), and the
original pi_e-ALONE test stays as a stricter secondary family.
    H1c_g paired, EARLY_comb(JSRL) - EARLY(PDD) > 0      EARLY_comb(JSRL) = mean ev_thr (16 combined-policy episodes at the current stage, eps 0.01)
                                                         over points 1..10; EARLY(PDD) = mean ev_alone over 1..10 (PDD's rolled-out policy IS pi_e)
    H2_g  one-sample, FINAL(JSRL) - mu_g > 0                  (mu_g = the guide's true eps-0.01 mean; the guide SE is added in quadrature
                                                              as a sensitivity row)
    H3_g  paired two-sided, FINAL(JSRL) - FINAL(DQfD)
Strict family (pi_e ALONE; Holm over its members; this is the ORIGINAL H1_g plus two stage-matched statistics):
    H1a_g original H1_g: EARLY(JSRL) - EARLY(PDD) over points 1..10, pi_e alone                          (paired, one-sided > 0)
    H1m_g pi_e alone at the FIRST evaluation at which JSRL's curriculum is at h = 0, minus PDD at the SAME evaluation point (paired by seed)
    H1p_g mean pi_e alone over the 10 evaluations from that point on, minus PDD over the same points (paired by seed)
          (H1m / H1p are defined only for the seeds that reach h = 0; the excluded count is reported and the judgment is withheld if > 20 %)
Secondary (Holm separately): S-C4early, S-C4final (AMENDMENT A2: TOST margin +-0.10 x the mean of the two arms' FINAL pooled; the original
+-0.10 mu_g margin is reported beside it as S-C4final-mu), S-H2rand, S-C2 (BCinit-FT collapse, a DQN analogue of the paper's Fig. 2), S-BC, S-IMIT.
Best-window / max-over-evaluations statistics are never computed here.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
import numpy as np
from scipy import stats as sps
from rl_jax_bench import stats as S


def load(out_dir, game):
    runs, meta = {}, None
    for p in sorted(glob.glob(os.path.join(out_dir, "seed_*", "metrics.json")), key=lambda x: int(x.split("seed_")[1].split("/")[0])):
        seed = int(p.split("seed_")[1].split("/")[0])
        cell = json.load(open(p)).get(game)
        if cell is None:
            continue
        meta = cell["meta"]
        for arm, r in cell.items():
            if arm != "meta":
                runs.setdefault(arm, {})[seed] = r
    return runs, meta


def one_sided(p_two, diff):
    return p_two / 2 if diff > 0 else 1 - p_two / 2


def windows(n_eval, early_frac=0.10, final_frac=0.03):
    ne = max(1, int(round(early_frac * n_eval)))
    nf = max(1, int(round(final_frac * n_eval)))
    return slice(1, 1 + ne), slice(n_eval + 1 - nf, n_eval + 1)


def scores(r, n_eval):
    ea, fi = windows(n_eval)
    x = np.asarray(r["ev_alone"], float)
    thr = np.asarray(r["ev_thr"], float)
    return dict(early=float(x[ea].mean()), final=float(x[fi].mean()), r0=float(x[0]), early5=float(x[1:1 + max(1, n_eval // 20)].mean()),
                min_early=float(x[ea].min()), greedy_final=float(np.asarray(r["ev_greedy"], float)[fi].mean()),
                early_comb=float(thr[ea].mean()))


def first_h0_point(r, n_stages):
    """First evaluation point j at which the curriculum was at its last stage (h = 0) when evaluated (ev_stage[j] = the stage in force at j);
    None if never.  Recomputed from ev_stage (the stored `first_eval_at_h0` is the same number, checked in tests)."""
    hit = np.nonzero(np.asarray(r["ev_stage"]) >= n_stages)[0]
    return int(hit[0]) if len(hit) else None


def stage_matched(runs, n_eval, n_stages, win=10):
    """H1m / H1p inputs: per seed that reached h = 0, JSRL pi_e-alone at the first h = 0 point f and its mean over f..f+win-1, and PDD at the SAME
    points.  Returns dict(seeds, f, jsrl_at, pdd_at, jsrl_win, pdd_win, n_excluded)."""
    seeds = sorted(runs["JSRL"])
    keep, f = [], []
    for s in seeds:
        fj = first_h0_point(runs["JSRL"][s], n_stages)
        if fj is not None:
            keep.append(s); f.append(fj)
    al = lambda a, s: np.asarray(runs[a][s]["ev_alone"], float)
    hi = lambda fj: min(fj + win, n_eval + 1)
    return dict(seeds=keep, f=f, n_excluded=len(seeds) - len(keep),
                jsrl_at=np.array([al("JSRL", s)[fj] for s, fj in zip(keep, f)]), pdd_at=np.array([al("PDD", s)[fj] for s, fj in zip(keep, f)]),
                jsrl_win=np.array([al("JSRL", s)[fj:hi(fj)].mean() for s, fj in zip(keep, f)]),
                pdd_win=np.array([al("PDD", s)[fj:hi(fj)].mean() for s, fj in zip(keep, f)]))


def tost(d, delta, alpha=0.05):
    """Paired TOST for |mean d| < delta: p = max of the two one-sided t-test p-values."""
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return dict(p=0.0 if abs(m) < delta else 1.0, delta=delta)
    p_lo = 1 - sps.t.cdf((m + delta) / se, n - 1)       # H0: mean <= -delta
    p_hi = sps.t.cdf((m - delta) / se, n - 1)           # H0: mean >= +delta
    return dict(p=float(max(p_lo, p_hi)), delta=float(delta), mean=float(m), se=float(se))


def paired(a, b, two_sided=False):
    pc = S.paired_compare(a, b)
    pc = dict(pc)
    pc["p_one_sided"] = one_sided(pc["t_p"], pc["diff_mean"])
    pc["p_two_sided"] = pc["t_p"]
    return pc


def analyse(out_dir, game):
    runs, meta = load(out_dir, game)
    n_eval = meta["n_eval_points"]
    mu_g, se_g = meta["guide"]["true_mean"], meta["guide"]["true_se"]
    per = {}
    for arm, rr in runs.items():
        seeds = sorted(rr)
        sc = [scores(rr[s], n_eval) for s in seeds]
        per[arm] = dict(seeds=seeds, **{k: np.array([x[k] for x in sc]) for k in sc[0]})
        extra = {}
        if "stage_reached" in rr[seeds[0]]:
            extra["stage_reached"] = [rr[s]["stage_reached"] for s in seeds]
            extra["first_eval_at_h0"] = [rr[s]["first_eval_at_h0"] for s in seeds]
        if "guide_steps" in rr[seeds[0]]:
            extra["guide_share"] = [rr[s]["guide_steps"] / meta["steps"] for s in seeds]
        per[arm]["extra"] = extra
    res = dict(game=game, steps=meta["steps"], n_eval=n_eval, mu_g=mu_g, se_g=se_g, guide=meta["guide"]["stage_list"], arms={}, primary={}, secondary={})
    for arm, p in per.items():
        res["arms"][arm] = dict(n=len(p["seeds"]), early=S.mean_se(p["early"]), final=S.mean_se(p["final"]), r0=S.mean_se(p["r0"]),
                                greedy_final=S.mean_se(p["greedy_final"]), early5=S.mean_se(p["early5"]), **p["extra"])
        res["arms"][arm]["curve_mean"] = np.mean([np.asarray(runs[arm][s]["ev_alone"], float) for s in p["seeds"]], axis=0).tolist()
    fam = {}
    res["strict"] = {}
    if "JSRL" in per and "PDD" in per:
        t = paired(per["JSRL"]["early_comb"], per["PDD"]["early"]); res["primary"]["H1c"] = t; fam["H1c"] = t["p_one_sided"]
        sf = {}
        t = paired(per["JSRL"]["early"], per["PDD"]["early"]); res["strict"]["H1a"] = t; sf["H1a"] = t["p_one_sided"]
        n_st = meta["cfg"]["jsrl"]["n_stages"]
        sm = stage_matched(runs, n_eval, n_st)
        res["first_h0_points"] = sm["f"]
        if len(sm["seeds"]) >= 2:
            for k, a, b in (("H1m", sm["jsrl_at"], sm["pdd_at"]), ("H1p", sm["jsrl_win"], sm["pdd_win"])):
                t = dict(paired(a, b)); t["n_excluded_never_h0"] = sm["n_excluded"]
                t["judgment_withheld"] = bool(sm["n_excluded"] > 0.2 * (len(sm["seeds"]) + sm["n_excluded"]))
                res["strict"][k] = t; sf[k] = t["p_one_sided"]
        res["holm_strict_game"] = S.holm(sf)
    if "JSRL" in per:
        d = per["JSRL"]["final"] - mu_g
        tt = sps.ttest_1samp(d, 0.0)
        se_q = float(np.sqrt(d.var(ddof=1) / len(d) + se_g ** 2))
        res["primary"]["H2"] = dict(diff_mean=float(d.mean()), diff_se=S.mean_se(d)[1], t95=S.t_interval(d), boot95=S.boot_ci(d),
                                    p_one_sided=one_sided(float(tt.pvalue), float(d.mean())), wins=int((d > 0).sum()), n=len(d),
                                    sens_se_with_guide=se_q, sens_t_with_guide=float(d.mean() / se_q) if se_q > 0 else None,
                                    kill=bool(S.t_interval(d)[1] < 0))
        fam["H2"] = res["primary"]["H2"]["p_one_sided"]
    if "JSRL" in per and "DQfD" in per:
        t = paired(per["JSRL"]["final"], per["DQfD"]["final"]); res["primary"]["H3"] = t; fam["H3"] = t["p_two_sided"]
    res["holm_game"] = S.holm(fam) if fam else {}
    sec, sp = {}, {}
    if "JSRL" in per and "JSRL-Random" in per:
        t = paired(per["JSRL"]["early"], per["JSRL-Random"]["early"]); sec["S-C4early"] = t; sp["S-C4early"] = t["p_one_sided"]
        d = per["JSRL"]["final"] - per["JSRL-Random"]["final"]
        pooled = float(np.mean(np.concatenate([per["JSRL"]["final"], per["JSRL-Random"]["final"]])))
        sec["S-C4final"] = dict(tost(d, 0.10 * pooled), pooled_final=pooled, diff_per_seed=d.round(3).tolist()); sp["S-C4final"] = sec["S-C4final"]["p"]
        sec["S-C4final-mu"] = tost(d, 0.10 * mu_g)       # the ORIGINAL margin (+-0.10 mu_g), kept for the record; not in the Holm family
    if "JSRL-Random" in per:
        d = per["JSRL-Random"]["final"] - mu_g; tt = sps.ttest_1samp(d, 0.0)
        sec["S-H2rand"] = dict(diff_mean=float(d.mean()), diff_se=S.mean_se(d)[1], p_one_sided=one_sided(float(tt.pvalue), float(d.mean())), wins=int((d > 0).sum()), n=len(d))
        sp["S-H2rand"] = sec["S-H2rand"]["p_one_sided"]
    if "BCinit-FT" in per:
        d = per["BCinit-FT"]["min_early"] - per["BCinit-FT"]["r0"]; tt = sps.ttest_1samp(d, 0.0)
        p_less = tt.pvalue / 2 if d.mean() < 0 else 1 - tt.pvalue / 2
        sec["S-C2"] = dict(collapse_mean=float(d.mean()), collapse_se=S.mean_se(d)[1], r0_mean=float(per["BCinit-FT"]["r0"].mean()),
                           min_early_mean=float(per["BCinit-FT"]["min_early"].mean()), rel_collapse=float(d.mean() / max(per["BCinit-FT"]["r0"].mean(), 1e-9)),
                           p_one_sided_less=float(p_less), n=len(d))
        sp["S-C2"] = sec["S-C2"]["p_one_sided_less"]
        if "JSRL" in per:
            t = paired(per["JSRL"]["final"], per["BCinit-FT"]["final"]); sec["S-BC"] = t; sp["S-BC"] = t["p_two_sided"]
    if "JSRL" in per and "IMIT" in per:
        t = paired(per["JSRL"]["final"], per["IMIT"]["final"]); sec["S-IMIT"] = t; sp["S-IMIT"] = t["p_one_sided"]
    if "JSRL" in per and "PDD" in per:   # C3 final (descriptive)
        sec["D-C3final"] = paired(per["JSRL"]["final"], per["PDD"]["final"])
    res["secondary"] = sec
    res["holm_secondary_game"] = S.holm(sp) if sp else {}
    return res


def holm_across_games(results, family="primary"):
    """Holm over a whole family across the games flown: primary = (games) x (H1c, H2, H3); strict = (games) x (H1a, H1m, H1p) (members
    whose judgment is withheld, or that do not exist, are left out and listed by the caller)."""
    p = {}
    for r in results:
        for k, v in r[family].items():
            if not isinstance(v, dict) or "p_one_sided" not in v:
                continue
            if v.get("judgment_withheld"):
                continue
            p[f"{r['game']}/{k}"] = v["p_two_sided"] if k == "H3" else v["p_one_sided"]
    return S.holm(p)


def fmt(res):
    f = lambda t: f"{t[0]:.3f} +- {t[1]:.3f}" if t[1] is not None else f"{t[0]:.3f}"
    L = [f"## {res['game']} (steps {res['steps']:.0f}, {res['n_eval']} eval points; guide mu_g = {res['mu_g']:.3f} +- {res['se_g']:.3f}; stages {res['guide']})",
         "", "| arm | n | R(0) | EARLY | FINAL | greedy FINAL | extra |", "|---|---|---|---|---|---|---|"]
    for arm, r in res["arms"].items():
        ex = ""
        if "stage_reached" in r:
            ex = f"stage reached {np.mean(r['stage_reached']):.1f}; first eval at h=0: {r['first_eval_at_h0']}; guide share {np.mean(r['guide_share']):.3f}"
        L.append(f"| {arm} | {r['n']} | {f(r['r0'])} | {f(r['early'])} | {f(r['final'])} | {f(r['greedy_final'])} | {ex} |")
    L.append("")
    for k, v in {**res["primary"], **res.get("strict", {}), **res["secondary"]}.items():
        L.append(f"- {k}: " + json.dumps({a: b for a, b in v.items() if a not in ("diff_per_seed", "rel_t95", "rel_boot95")}, default=float))
    L.append("- Holm (within game, primary): " + json.dumps(res["holm_game"]))
    L.append("- Holm (within game, strict pi_e-alone): " + json.dumps(res.get("holm_strict_game", {})))
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, nargs="+", help="run dirs: runs/<x>/<Game> (one per game)")
    ap.add_argument("--game", required=True, nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    results = [analyse(o, g) for o, g in zip(a.out, a.game)]
    for r in results:
        print(fmt(r)); print()
    hg = holm_across_games(results)
    hs = holm_across_games(results, "strict")
    print("Holm across the primary family:", json.dumps(hg))
    print("Holm across the strict (pi_e alone) family:", json.dumps(hs))
    if a.json:
        json.dump(dict(games=results, holm_primary=hg, holm_strict=hs), open(a.json, "w"), indent=1, default=float)
