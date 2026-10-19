"""Aggregate full-scale: per-seed values (seed = unit), ratios to Random graded against the README MATCH bands,
the primary family H1-H6 (one-sided paired t + paired bootstrap CI, Holm), the secondary checks and the derivation
predictions, exactly as declared in README. Writes runs/full/report.json and runs/seed_<n>/metrics.json.

    python scripts/full_report.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy import stats as st

from rl_jax_bench.stats import mean_se, t_interval, boot_ci, paired_compare, holm  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G3 = os.path.join(REPO, "runs", "full")
LEARN = ["flat", "bigru", "mhsa", "mhsa_nobc", "mhsa_noae", "mhsa_lit"]
BASE = ["static", "random", "coverage", "coverage_boustro", "hand", "ceiling"]
KEYS = ["sr", "m_cov", "m_cov_lib", "gospa_d", "gospa_loc", "gospa_miss", "gospa_false", "mean_tracks", "n_init",
        "n_del", "life_median", "cov_per_track", "central3", "frac_steps_tracked", "n_episodes_no_track"]
# paper values (paper §3 run A; companion Table VI run B; Fig. 1 medians), sources.yaml
PAPER = dict(m_cov=dict(static=2630.72, random=948.15, coverage=574.27, flat=727, bigru=765, mhsa=391),
             m_cov_B=dict(flat=790.63, bigru=730.56, mhsa=461.98),
             gospa_d=dict(flat=1.25e7, bigru=1.20e7, mhsa=1.13e7, random=1.25e7, coverage=1.35e7, static=1.22e7),
             gospa_miss=dict(flat=1.33e11, bigru=1.26e11, mhsa=1.04e11, random=1.31e11, coverage=1.44e11,
                             static=1.49e11))
# README ratio targets and MATCH bands (two runs, each widened by the delta-method SE of the 100-episode SDs)
BANDS = {"m_cov:mhsa": (0.412, 0.487, 0.351, 0.564), "m_cov:flat": (0.767, 0.834, 0.652, 0.955),
         "m_cov:bigru": (0.807, 0.771, 0.662, 0.920), "m_cov:coverage": (0.605, 0.605, 0.532, 0.680),
         "m_cov:static": (2.774, 2.774, 2.41, 3.14), "sr:coverage": (0.380, 0.380, 0.366, 0.395),
         "sr:flat": (0.96, 1.04, 0.96, 1.04), "sr:bigru": (0.96, 1.04, 0.96, 1.04), "sr:mhsa": (0.96, 1.04, 0.96, 1.04),
         "gospa_d:mhsa": (0.890, 0.919, 0.871, 0.938), "gospa_miss:mhsa": (0.79, 0.79, 0.79, 0.79)}
SB3_REF = {"m_cov": "2096 ± 25", "gospa_d": "1.181e7 ± 3.9e4", "gospa_miss": "1.405e11 ± 8.4e8"}  # small-scale scale, n=3


def grade_ratio(v, band):
    a, b, lo, hi = band
    if lo <= v <= hi:
        return "MATCH"
    near = min((a, b), key=lambda p: abs(v - p))
    rel = abs(v - near) / abs(near)
    return "MATCH" if rel <= 0.05 else ("CLOSE" if rel <= 0.20 else "DIVERGENT")


def load():
    base = json.load(open(os.path.join(G3, "baselines.json")))
    vals, versions = {}, set()
    for arm, d in base["arms"].items():
        seeds = sorted(d, key=int)
        vals[arm] = {k: [d[s if len(seeds) > 1 else seeds[0]].get(k, np.nan) for s in
                         (map(str, range(5)) if len(seeds) > 1 else [seeds[0]] * 5)] for k in KEYS}
        vals[arm]["_per_episode"] = {s: d[s]["per_episode"] for s in d}
    # identity = the scenario KEYS (exact) + env source/config. The float64 truth sha differs between CPU and GPU (and
    # eager vs jit) at the ulp level: max |diff| 7.3e-12 m, max rel 3e-14 on these 100 scenarios (README full-scale).
    truth_shas = {base["eval_scenarios"]["truth_sha256"]}
    versions.add((base["eval_scenarios"]["keys_sha256"], base["env_version"]["data_py_sha256"],
                  base["env_version"]["env_cfg_sha256"]))
    metr = {}
    for arm in LEARN:
        fs = sorted(glob.glob(os.path.join(G3, arm, "seed_*", "metrics.json")))
        if not fs:
            continue
        ms = [json.load(open(f)) for f in fs]
        ms.sort(key=lambda m: m["seed"])
        metr[arm] = ms
        vals[arm] = {k: [m["eval_stochastic"].get(k, np.nan) for m in ms] for k in KEYS}
        vals[arm + "@det"] = {k: [m["eval_deterministic"].get(k, np.nan) for m in ms] for k in KEYS}
        for m in ms:
            truth_shas.add(m["eval_scenarios"]["truth_sha256"])
            versions.add((m["eval_scenarios"]["keys_sha256"], m["env_version"]["data_py_sha256"],
                          m["env_version"]["env_cfg_sha256"]))
    assert len(versions) == 1, f"env version / eval scenarios differ across arms: {versions}"
    bc = sorted(glob.glob(os.path.join(G3, "mhsa_bcmse", "seed_*", "metrics.json")))
    bcm = [json.load(open(f)) for f in bc]
    print("truth sha256 seen (device/ulp variants):", sorted(truth_shas))
    return base, vals, metr, bcm, versions.pop()


def f(x, d=4):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{d}g}"


def main():
    base, vals, metr, bcm, ver = load()
    rep = dict(version=ver, rows={}, ratios={}, primary={}, secondary={}, predictions={}, gap={}, det={})
    for arm, v in vals.items():
        if arm.startswith("_") or not isinstance(v, dict) or "sr" not in v:
            continue
        rep["rows"][arm] = {}
        for k in KEYS:
            x = np.asarray(v[k], float)
            fin = x[np.isfinite(x)]
            mu, se = mean_se(fin) if fin.size else (float("nan"), None)
            rep["rows"][arm][k] = dict(mean=mu, se=se, n=int(fin.size), per_seed=list(map(float, x)))
    rnd = vals["random"]
    # ---------------- ratios to Random (per seed, paired by seed index)
    for key in ("m_cov", "sr", "gospa_d", "gospa_miss"):
        for arm in [a for a in BASE + LEARN if a in vals]:
            r = np.asarray(vals[arm][key], float) / np.asarray(rnd[key], float)
            r = r[np.isfinite(r)]
            m, se = mean_se(r) if r.size else (float("nan"), None)
            band = BANDS.get(f"{key}:{arm}")
            rep["ratios"][f"{key}:{arm}"] = dict(mean=m, se=se, per_seed=r.tolist(), band=band,
                                                 grade=grade_ratio(m, band) if band else None)
    # ---------------- primary family H1-H6
    H = {"H1": ("mhsa", "flat", "m_cov"), "H2": ("mhsa", "bigru", "m_cov"), "H3": ("mhsa", "coverage", "m_cov"),
         "H4": ("mhsa", "random", "gospa_d"), "H5": ("mhsa", "flat", "gospa_miss"), "H6": ("mhsa", "bigru", "gospa_miss")}
    praw = {}
    for h, (a, b, k) in H.items():
        if a not in vals or b not in vals:
            continue
        x, y = np.asarray(vals[a][k], float), np.asarray(vals[b][k], float)
        pc = paired_compare(x, y)
        d = x - y
        t1 = st.ttest_1samp(d, 0.0, alternative="less")
        p2 = st.ttest_1samp(d, 0.0).pvalue
        praw[h] = float(t1.pvalue)
        rep["primary"][h] = dict(arm=a, ref=b, metric=k, mean_arm=float(x.mean()), mean_ref=float(y.mean()),
                                 diff=pc["diff_mean"], diff_se=pc["diff_se"], boot95=pc["boot95"], t95=pc["t95"],
                                 p_one_sided=float(t1.pvalue), p_two_sided=float(p2), wins_lower=int((d < 0).sum()),
                                 rel=pc["rel_improvement"], rel_boot95=pc["rel_boot95"])
    hol = holm(praw) if praw else {}
    for h, r in rep["primary"].items():
        r["p_holm"] = hol[h]["p_holm"]
        sup = r["p_holm"] < 0.05 and r["boot95"][1] < 0
        rev = r["diff"] > 0 and r["p_two_sided"] < 0.05
        r["verdict"] = "supported" if sup else ("reversed" if rev else "not supported")
        r["grade"] = "MATCH" if sup else ("DIVERGENT" if rev else ("CLOSE" if r["diff"] < 0 else "DIVERGENT (point estimate reversed, n.s.)"))
    # ---------------- secondary
    sec = {}
    sec["S1"] = dict(coverage=float(np.mean(vals["coverage"]["sr"])), random=float(np.mean(rnd["sr"])),
                     holds=bool(np.mean(vals["coverage"]["sr"]) > np.mean(rnd["sr"])))
    srs = {a: float(np.mean(vals[a]["sr"])) for a in BASE + LEARN if a in vals}
    sec["S2"] = dict(sr=srs, holds=bool(min(srs, key=srs.get) == "static"))
    sec["S3"] = {a: dict(ratio=rep["ratios"][f"sr:{a}"]["mean"], holds=bool(0.8 <= rep["ratios"][f"sr:{a}"]["mean"] <= 1.2))
                 for a in ("flat", "bigru", "mhsa") if a in vals}
    if "flat" in vals and "bigru" in vals:
        x, y = np.asarray(vals["bigru"]["m_cov"], float), np.asarray(vals["flat"]["m_cov"], float)
        d = (x - y) / np.mean(rnd["m_cov"])
        p2 = float(st.ttest_rel(x, y).pvalue)
        ci90 = t_interval(d, 0.90)
        sec["F~B"] = dict(rel_diff=float(d.mean()), p_two_sided=p2, ci90=ci90,
                          consistent=bool(p2 > 0.05 and abs(d.mean()) < 0.15),
                          established=bool(-0.15 < ci90[0] and ci90[1] < 0.15))
    s = rep["rows"]
    sec["G-static"] = dict(static_d=s["static"]["gospa_d"]["mean"], random_d=s["random"]["gospa_d"]["mean"],
                           static_loc=s["static"]["gospa_loc"]["mean"], static_false=s["static"]["gospa_false"]["mean"],
                           static_d_below_random=bool(s["static"]["gospa_d"]["mean"] < s["random"]["gospa_d"]["mean"]),
                           static_loc_lowest=bool(s["static"]["gospa_loc"]["mean"] <= min(s[a]["gospa_loc"]["mean"]
                                                  for a in BASE + LEARN if a in s)),
                           static_false_lowest=bool(s["static"]["gospa_false"]["mean"] <= min(s[a]["gospa_false"]["mean"]
                                                    for a in BASE + LEARN if a in s)))
    rep["secondary"] = sec
    # ---------------- derivation predictions
    pr = {}
    if "mhsa" in metr:
        xs, ys = [], []
        for m in metr["mhsa"]:
            for c in m["curve"]:
                xs.append(c["ret_flip"])
                ys.append(-c["m_cov"])
        rho = st.spearmanr(xs, ys)
        pr["P-MIS"] = dict(rho=float(rho.statistic), p=float(rho.pvalue), n_points=len(xs),
                           checkpoints_per_seed=len(metr["mhsa"][0]["curve"]),
                           verdict="GO (aligned)" if rho.statistic > 0.5 else ("KILL (misaligned)" if rho.statistic < 0 else "neither"))
    for name, arm in (("P-BC", "mhsa_nobc"), ("P-AE", "mhsa_noae")):
        if "mhsa" in vals and arm in vals:
            x, y = np.asarray(vals["mhsa"]["m_cov"], float), np.asarray(vals[arm]["m_cov"], float)
            p2 = float(st.ttest_rel(x, y).pvalue)
            ci = t_interval(y)
            hw = (ci[1] - ci[0]) / 2
            dd = float(x.mean() - y.mean())
            pr[name] = dict(diff=dd, p_two_sided=p2, halfwidth_ref=hw,
                            verdict="GO (equivalent)" if (p2 > 0.05 and abs(dd) < hw) else
                            f"KILL ({arm} {'better' if dd > 0 else 'worse'})")
    if "mhsa" in metr:
        aef = [m["pretrain"]["ae_final"] for m in metr["mhsa"]]
        pr["P-AE"]["ae_final"] = aef
        pr["P-AE"]["ae_within_10pct"] = bool(all(abs(a / 0.154062 - 1) <= 0.10 for a in aef))
    if bcm:
        c3 = [m["eval_stochastic"]["central3"] for m in bcm]
        pr["P-BCmse"] = dict(central3=c3, central1=[m["eval_stochastic"]["central1"] for m in bcm],
                             sr=[m["eval_stochastic"]["sr"] for m in bcm], verdict="GO" if min(c3) > 0.9 else "KILL")
    if "mhsa_lit" in vals and "mhsa" in vals:
        li, fl = vals["mhsa_lit"], vals["mhsa"]
        more_init = np.mean(li["n_init"]) > np.mean(fl["n_init"])
        more_miss = np.mean(li["gospa_miss"]) > np.mean(fl["gospa_miss"])
        pr["P-SIGN"] = dict(init=dict(lit=mean_se(li["n_init"]), flip=mean_se(fl["n_init"]),
                                      diff_t95=paired_compare(li["n_init"], fl["n_init"])["t95"]),
                            miss=dict(lit=mean_se(li["gospa_miss"]), flip=mean_se(fl["gospa_miss"]),
                                      diff_t95=paired_compare(li["gospa_miss"], fl["gospa_miss"])["t95"]),
                            m_cov=dict(lit=mean_se(li["m_cov"]), flip=mean_se(fl["m_cov"])),
                            verdict="GO" if (more_init and more_miss) else "KILL")
    pe = vals["_per_episode"] if "_per_episode" in vals else None
    stp = base["arms"]["static"]
    ratios = []
    for sd, d in stp.items():
        mi, di = np.asarray(d["per_episode"]["gospa_miss"]), np.asarray(d["per_episode"]["gospa_d"])
        ratios.append(np.mean(mi / di))
    pr["P-GOSPA"] = dict(mean_ratio=float(np.mean(ratios)), target=12247.45,
                         rel=float(np.mean(ratios) / 12247.45 - 1),
                         verdict="GO" if abs(np.mean(ratios) / 12247.45 - 1) <= 0.05 else "KILL")
    rep["predictions"] = pr
    # ---------------- budget diagnostics: training entropy, curve change start -> end, per seed (descriptive)
    bud = {}
    for a, ms in metr.items():
        ent = [m["train_hist"]["entropy"][-1] for m in ms]
        ent0 = [m["train_hist"]["entropy"][0] for m in ms]
        dc = [m["curve"][-1]["m_cov"] - m["curve"][0]["m_cov"] for m in ms]
        dr = [m["curve"][-1]["ret_flip"] - m["curve"][0]["ret_flip"] for m in ms]
        dd = [m["curve"][-1]["gospa_d"] - m["curve"][0]["gospa_d"] for m in ms]
        ev = [m["train_hist"]["ev"][-1] for m in ms]
        bud[a] = dict(entropy_first=mean_se(ent0), entropy_last=mean_se(ent), uniform=float(2 * np.log(19)),
                      d_curve_m_cov=mean_se(dc), d_curve_ret_flip=mean_se(dr), d_curve_gospa_d=mean_se(dd),
                      d_m_cov_t95=t_interval(dc), d_gospa_d_t95=t_interval(dd), ev_last=mean_se(ev),
                      steps=ms[0]["env_steps"])
    rep["budget"] = bud
    # ---------------- gap fraction (E1-GAP)
    hand, ceil = vals["hand"], vals["ceiling"]
    for key in ("m_cov", "gospa_d", "gospa_miss"):
        mh, mc = float(np.mean(hand[key])), float(np.mean(ceil[key]))
        # the hand rule and the ceiling are deterministic: their 'CI' is over the 100 episodes (seed-level SE = 0)
        epi_h = np.asarray(base["arms"]["hand"]["0"]["per_episode"][key], float)
        epi_c = np.asarray(base["arms"]["ceiling"]["0"]["per_episode"][key], float)
        epi_h, epi_c = epi_h[np.isfinite(epi_h)], epi_c[np.isfinite(epi_c)]
        gap = mh - mc
        hw = 1.96 * np.sqrt(epi_h.var(ddof=1) / len(epi_h) + epi_c.var(ddof=1) / len(epi_c))
        row = dict(hand=mh, ceiling=mc, gap=gap, gap_halfwidth=float(hw), defined=bool(gap > hw), learners={})
        for a in LEARN:
            if a not in vals:
                continue
            x = np.asarray(vals[a][key], float)
            fr = (mh - x) / gap
            row["learners"][a] = dict(f=float(fr.mean()), boot95=boot_ci(fr) if gap > hw else None)
        rep["gap"][key] = row
    json.dump(rep, open(os.path.join(G3, "report.json"), "w"), indent=1, default=float)
    # ---------------- runs/seed_<n>/metrics.json
    for s in range(5):
        out = dict(seed=s, stage="full", version=dict(eval_scenario_keys_sha256=ver[0], data_py_sha256=ver[1],
                                                       env_cfg_sha256=ver[2]), arms={})
        for a in BASE + LEARN:
            if a not in vals:
                continue
            out["arms"][a] = {k: float(vals[a][k][s]) for k in KEYS}
            if a in metr:
                m = metr[a][s]
                out["arms"][a]["per_episode"] = {k: m["per_episode"][k] for k in ("sr", "m_cov", "gospa_d", "gospa_miss")}
                out["arms"][a]["env_steps"] = m["env_steps"]
                out["arms"][a]["source"] = f"runs/full/{a}/seed_{s}/metrics.json"
            else:
                pk = str(s) if str(s) in base["arms"][a] else "0"
                out["arms"][a]["per_episode"] = {k: base["arms"][a][pk]["per_episode"][k]
                                                 for k in ("sr", "m_cov", "gospa_d", "gospa_miss")}
                out["arms"][a]["source"] = f"runs/full/baselines.json arms.{a}.{pk}"
        os.makedirs(os.path.join(REPO, "runs", f"seed_{s}"), exist_ok=True)
        json.dump(out, open(os.path.join(REPO, "runs", f"seed_{s}", "metrics.json"), "w"), default=float)
    # ---------------- print
    print("| arm | n | SR | M_cov | M_cov_lib | Σ GOSPA d | Σ loc | Σ missed | Σ false | tracks | init | life |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in BASE + LEARN + [x + "@det" for x in LEARN]:
        if a not in rep["rows"] or a == "_per_episode":
            continue
        r = rep["rows"][a]
        cell = lambda k: f(r[k]["mean"]) + (f" ± {f(r[k]['se'], 2)}" if r[k]["se"] else "")
        print(f"| {a} | {r['sr']['n']} | " + " | ".join(cell(k) for k in ("sr", "m_cov", "m_cov_lib", "gospa_d",
              "gospa_loc", "gospa_miss", "gospa_false", "mean_tracks", "n_init", "life_median")) + " |")
    print("\nratios:")
    for k, r in rep["ratios"].items():
        if r["band"]:
            print(k, f(r["mean"]), "±", f(r["se"] or 0, 2), r["band"], r["grade"])
    print("\nprimary:")
    for h, r in rep["primary"].items():
        print(h, r["metric"], r["arm"], "vs", r["ref"], "diff", f(r["diff"]), "±", f(r["diff_se"], 2), "boot", r["boot95"],
              "p1", f(r["p_one_sided"], 3), "holm", f(r["p_holm"], 3), r["verdict"], r["grade"])
    print("\nsecondary:", json.dumps(sec, default=float))
    print("\npredictions:", json.dumps(pr, default=float))
    print("\nbudget:", json.dumps(bud, default=float))
    print("\ngap:", json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in rep["gap"].items()}, default=float))


if __name__ == "__main__":
    main()
