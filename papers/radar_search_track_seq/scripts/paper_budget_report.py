"""Aggregate the paper-budget learner run (README), CPU only.

Learners: runs/paper_budget/<arm>/seed_<n>/metrics.json (each lane's SELECTED model, 100 full-scale evaluation scenarios,
stochastic and deterministic). Fixed policies: the stored full-scale rows (runs/full/baselines.json). Both evaluation modes
are graded with the frozen README rules (seed = unit): ratios to Random against the current bands, H1-H6
(one-sided paired t over seed means, paired bootstrap 95 % CI, Holm over the six), the all-pairings sensitivity, the
hand-rule position, E1-GAP, S1-S3, F~B, P-MIS, T2T / coast readings and the training diagnostics.

    python scripts/paper_budget_report.py      # -> runs/paper_budget/report.json, report.txt
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os
import sys

import numpy as np
from scipy import stats as st

from rl_jax_bench.stats import mean_se, t_interval, boot_ci, paired_compare, holm  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PB = os.environ.get("PB_DIR", os.path.join(REPO, "runs", "paper_budget"))
G3 = os.path.join(REPO, "runs", "full")
LEARN = ["flat", "bigru", "mhsa", "mhsa_nobc", "mhsa_noae", "mhsa_lit"]
BASE = ["static", "random", "coverage", "coverage_boustro", "hand", "ceiling"]
KEYS = ["sr", "m_cov", "m_cov_lib", "gospa_d", "gospa_loc", "gospa_miss", "gospa_false", "mean_tracks", "n_init",
        "n_del", "life_median", "cov_per_track", "central3", "frac_steps_tracked", "n_episodes_no_track"]
N_TGT, N_EVAL, F4_MIN = 3, 100, 12
COAST = json.load(open(os.path.join(REPO, "runs", "recheck", "measurement", "probe_coast.json")))[
    "primary_50ms_trace_r20km"]["coast_mean_live"]
SR_LO, SR_HI = 0.127 / 0.137, 0.137 / 0.127
# (paper value(s), band) -- README with the/8/9 amendments
BANDS = {"m_cov:mhsa": ([0.412, 0.487], (0.351, 0.564)), "m_cov:flat": ([0.767, 0.834], (0.652, 0.955)),
         "m_cov:bigru": ([0.807, 0.771], (0.662, 0.920)), "m_cov:coverage": ([0.605], (0.532, 0.680)),
         "m_cov:static": ([2.774], (2.505, 3.044)),
         "sr:flat": ([SR_LO, SR_HI], (SR_LO, SR_HI)), "sr:bigru": ([SR_LO, SR_HI], (SR_LO, SR_HI)),
         "sr:mhsa": ([SR_LO, SR_HI], (SR_LO, SR_HI)), "sr:coverage": ([0.380], (0.366, 0.395)),
         "gospa_d_med:mhsa": ([1.187 / 1.247], ((1.187 - 0.0017) / (1.247 + 0.0017), (1.187 + 0.0017) / (1.247 - 0.0017))),
         "gospa_miss_med:mhsa": ([1.100 / 1.308], ((1.100 - 0.0024) / (1.308 + 0.0024), (1.100 + 0.0024) / (1.308 - 0.0024)))}
PAPER_T2T = dict(random=1.585, coverage=0.942, static=0.288, flat=1.407, bigru=1.430, mhsa=1.152)
PAPER_COAST = dict(random=948.15 / 2630.72, coverage=574.27 / 2630.72, flat=727 / 2630.72, bigru=765 / 2630.72,
                   mhsa=391 / 2630.72)


def grade(x, refs, band):
    if not np.isfinite(x):
        return "UNDEFINED"
    lo, hi = band
    rel = min(abs(x / r - 1) for r in refs)
    if lo <= x <= hi or rel <= 0.05:
        return "MATCH"
    return "CLOSE" if rel <= 0.20 else "DIVERGENT"


def ms(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return dict(mean=float("nan"), se=None, n=0)
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else None, n=int(v.size),
                per_seed=[float(x) for x in v])


def load():
    base = json.load(open(os.path.join(G3, "baselines.json")))
    keys_sha = base["eval_scenarios"]["keys_sha256"]
    B = {}
    for arm, d in base["arms"].items():
        seeds = [str(s) if str(s) in d else "0" for s in range(5)]
        B[arm] = dict(vals={k: [d[s].get(k, np.nan) for s in seeds] for k in KEYS},
                      pe=[{k: np.asarray(v, float) for k, v in d[s]["per_episode"].items()} for s in seeds])
    L, M = {}, {}
    versions = set()
    for arm in LEARN:
        fs = sorted(glob.glob(os.path.join(PB, arm, "seed_*", "metrics.json")))
        if not fs:
            continue
        ms_ = sorted((json.load(open(f)) for f in fs), key=lambda m: m["seed"])
        assert [m["seed"] for m in ms_] == list(range(5)), f"{arm}: seeds {[m['seed'] for m in ms_]}"
        M[arm] = ms_
        for m in ms_:
            assert m["eval_scenarios"]["keys_sha256"] == keys_sha, f"{arm}/{m['seed']}: evaluation scenario keys differ"
            versions.add((m["env_version"]["data_py_sha256"], m["env_version"]["env_cfg_sha256"]))
        for mode, ek, pk in (("stoch", "eval_stochastic", "per_episode"), ("det", "eval_deterministic", "per_episode_det")):
            L[(arm, mode)] = dict(vals={k: [m[ek].get(k, np.nan) for m in ms_] for k in KEYS},
                                  pe=[{k: np.asarray(v, float) for k, v in m[pk].items()} for m in ms_])
    assert len(versions) == 1, versions
    return base, B, L, M, keys_sha, versions.pop()


def h_family(get, out):
    H = {"H1": ("mhsa", "flat", "m_cov"), "H2": ("mhsa", "bigru", "m_cov"), "H3": ("mhsa", "coverage", "m_cov"),
         "H4": ("mhsa", "random", "gospa_d"), "H5": ("mhsa", "flat", "gospa_miss"),
         "H6": ("mhsa", "bigru", "gospa_miss")}
    praw = {}
    for h, (a, b, k) in H.items():
        x, y = get(a, k), get(b, k)
        if x is None or y is None or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            out[h] = dict(arm=a, ref=b, metric=k, verdict="UNDEFINED (non-finite seed value)", grade="UNDEFINED")
            continue
        pc = paired_compare(x, y)
        d = x - y
        p1 = float(st.ttest_1samp(d, 0.0, alternative="less").pvalue)
        p2 = float(st.ttest_1samp(d, 0.0).pvalue)
        praw[h] = p1
        # all-pairings sensitivity: the lanes are independent, so pairing by seed index is nominal
        perms = list(itertools.permutations(range(5))) if b not in ("coverage",) else [tuple(range(5))]
        sig1 = rev = 0
        for pm in perms:
            dd = x - y[list(pm)]
            if st.ttest_1samp(dd, 0.0, alternative="less").pvalue < 0.05:
                sig1 += 1
            if dd.mean() > 0 and st.ttest_1samp(dd, 0.0).pvalue < 0.05:
                rev += 1
        welch1 = float(st.ttest_ind(x, y, equal_var=False, alternative="less").pvalue) if b != "coverage" else None
        out[h] = dict(arm=a, ref=b, metric=k, mean_arm=float(x.mean()), mean_ref=float(y.mean()), diff=pc["diff_mean"],
                      diff_se=pc["diff_se"], boot95=pc["boot95"], t95=pc["t95"], p_one_sided=p1, p_two_sided=p2,
                      wins_lower=int((d < 0).sum()), rel=pc["rel_improvement"],
                      pairings=dict(n=len(perms), frac_one_sided_p_lt_05=sig1 / len(perms),
                                    frac_reversed=rev / len(perms), welch_one_sided_p=welch1))
    hol = holm(praw) if praw else {}
    for h, r in out.items():
        if h not in hol:
            continue
        r["p_holm"] = hol[h]["p_holm"]
        sup = r["p_holm"] < 0.05 and r["boot95"][1] < 0
        revd = r["diff"] > 0 and r["p_two_sided"] < 0.05
        r["verdict"] = "supported" if sup else ("reversed" if revd else "not supported")
        r["grade"] = "MATCH" if sup else ("DIVERGENT" if revd else ("CLOSE" if r["diff"] < 0 else "DIVERGENT*"))


def main():
    base, B, L, M, keys_sha, ver = load()
    rep = dict(eval_scenario_keys_sha256=keys_sha, learner_env_version=dict(data_py_sha256=ver[0], env_cfg_sha256=ver[1]),
               full_baseline_env_version=base["env_version"], coast_value=COAST, modes={}, training={}, pmis={})
    rnd = B["random"]
    for mode in ("stoch", "det"):
        R = dict(rows={}, ratios={}, primary={}, hand={}, gap={}, secondary={}, readings={})

        def V(arm):
            return L[(arm, mode)] if (arm, mode) in L else B.get(arm)

        def get(arm, k):
            v = V(arm)
            return None if v is None else np.asarray(v["vals"][k], float)

        arms = [a for a in BASE if a in B] + [a for a in LEARN if (a, mode) in L]
        for a in arms:
            R["rows"][a] = {k: ms(get(a, k)) for k in KEYS}
            R["rows"][a]["track_bearing_eps"] = [int(N_EVAL - x) for x in get(a, "n_episodes_no_track")]
        # ratios to Random, per seed (paired by seed index)
        for a in arms:
            tb = R["rows"][a]["track_bearing_eps"]
            for k in ("m_cov", "sr", "gospa_d", "gospa_miss"):
                r = get(a, k) / get("random", k)
                band = BANDS.get(f"{k}:{a}")
                undefined = k == "m_cov" and min(tb) <= F4_MIN
                x = ms(r)["mean"]
                R["ratios"][f"{k}:{a}"] = dict(**ms(r), band=band[1] if band else None, paper=band[0] if band else None,
                                               f4_undefined=undefined,
                                               grade=None if not band else ("UNDEFINED (too few track-bearing episodes)" if undefined
                                                                           else grade(x, *band)),
                                               grade_random_itself=None if not band else grade(1.0, *band),
                                               delta_pct=None if not band else 100 * min((x / p - 1 for p in band[0]),
                                                                                        key=abs))
            for k in ("gospa_d", "gospa_miss"):
                med = [np.median(V(a)["pe"][s][k]) / np.median(rnd["pe"][s][k]) for s in range(5)]
                band = BANDS.get(f"{k}_med:{a}")
                x = float(np.mean(med))
                R["ratios"][f"{k}_med:{a}"] = dict(**ms(med), band=band[1] if band else None,
                                                   paper=band[0] if band else None,
                                                   grade=grade(x, *band) if band else None,
                                                   grade_random_itself=grade(1.0, *band) if band else None,
                                                   delta_pct=100 * (x / band[0][0] - 1) if band else None)
        h_family(get, R["primary"])
        # hand-rule position: learner_seed - hand (hand deterministic)
        for a in [x for x in LEARN if (x, mode) in L]:
            R["hand"][a] = {}
            for k in ("m_cov", "gospa_d", "gospa_miss"):
                x, y = get(a, k), get("hand", k)
                fin = np.isfinite(x)
                if fin.sum() < 2:
                    R["hand"][a][k] = dict(defined=False)
                    continue
                d = x[fin] - y[fin]
                R["hand"][a][k] = dict(learner=ms(x), hand=float(y[0]), diff=ms(d), t95=t_interval(d),
                                       p_two_sided=float(st.ttest_1samp(d, 0.0).pvalue),
                                       learner_ratio=ms(x / get("random", k)),
                                       hand_ratio=ms(get("hand", k) / get("random", k)),
                                       learner_better=bool(d.mean() < 0))
        # E1-GAP (declared)
        for k in ("m_cov", "gospa_d", "gospa_miss"):
            mh, mc = float(np.nanmean(get("hand", k))), float(np.nanmean(get("ceiling", k)))
            eh, ec = B["hand"]["pe"][0][k], B["ceiling"]["pe"][0][k]
            eh, ec = eh[np.isfinite(eh)], ec[np.isfinite(ec)]
            gap = mh - mc
            hw = 1.96 * np.sqrt(eh.var(ddof=1) / len(eh) + ec.var(ddof=1) / len(ec))
            row = dict(hand=mh, ceiling=mc, gap=gap, gap_halfwidth=float(hw), defined=bool(gap > hw), learners={})
            for a in [x for x in LEARN if (x, mode) in L]:
                fr = (mh - get(a, k)) / gap
                fr = fr[np.isfinite(fr)]
                row["learners"][a] = dict(f=float(fr.mean()) if fr.size else None,
                                          boot95=boot_ci(fr) if (gap > hw and fr.size > 1) else None)
            R["gap"][k] = row
        # secondary
        sec = {}
        srs = {a: float(np.mean(get(a, "sr"))) for a in arms}
        sec["S1"] = dict(coverage=srs["coverage"], random=srs["random"], holds=srs["coverage"] > srs["random"])
        sec["S2"] = dict(sr=srs, holds=min(srs, key=srs.get) == "static")
        sec["S3"] = {a: dict(ratio=R["ratios"][f"sr:{a}"]["mean"],
                             holds=bool(0.8 <= R["ratios"][f"sr:{a}"]["mean"] <= 1.2))
                     for a in ("flat", "bigru", "mhsa") if (a, mode) in L}
        x, y = get("bigru", "m_cov"), get("flat", "m_cov")
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            d = (x - y) / np.mean(get("random", "m_cov"))
            p2 = float(st.ttest_rel(x, y).pvalue)
            ci90 = t_interval(d, 0.90)
            sec["F~B"] = dict(rel_diff=float(d.mean()), p_two_sided=p2, ci90=ci90,
                              consistent=bool(p2 > 0.05 and abs(d.mean()) < 0.15),
                              established=bool(-0.15 < ci90[0] and ci90[1] < 0.15))
        R["secondary"] = sec
        # readings: T2T, coast-normalised M_cov, median M_cov ratio
        for a in arms:
            pe = V(a)["pe"]
            t2t = [p["n_init"].mean() / N_TGT for p in pe]
            cn, medr = [], []
            for s, p in enumerate(pe):
                m = p["m_cov"][np.isfinite(p["m_cov"])]
                mr = rnd["pe"][s]["m_cov"][np.isfinite(rnd["pe"][s]["m_cov"])]
                if m.size:
                    cn.append(m.mean() / COAST)
                    medr.append(np.median(m) / np.median(mr))
            R["readings"][a] = dict(t2t=ms(t2t), paper_t2t=PAPER_T2T.get(a), mcov_over_coast=ms(cn),
                                    paper_mcov_over_static=PAPER_COAST.get(a), mcov_median_ratio=ms(medr))
        rep["modes"][mode] = R
    # ---- training diagnostics (suspect our implementation first)
    for a, ms_ in M.items():
        per_it = 16384
        rows = []
        for m in ms_:
            th = m["train_hist"]
            bi = max(0, m["best_step"] // per_it - 1)
            c = m["curve"]
            cb = [x for x in c if x["env_steps"] == m["best_step"]]
            rows.append(dict(seed=m["seed"], iterations=m["iterations"], env_steps=m["env_steps"],
                             best_step=m["best_step"], stalled_at_end=m["stalled_at_end"],
                             ent_first=th["entropy"][0], ent_last=th["entropy"][-1],
                             ent_at_best=th["entropy"][bi] if m["best_step"] > 0 else None,
                             ev_last=th["ev"][-1], ev_max=max(th["ev"]), kl_last=th["approx_kl"][-1],
                             clip_last=th["clip_frac"][-1], gnorm_last=th["gnorm"][-1], vl_last=th["vl"][-1],
                             cnn_alive_ppo_start_teacher=m["pretrain"].get("cnn_alive_ppo_start", float("nan")),
                             cnn_alive=m["cnn_alive"],
                             curve_start=dict(m_cov=c[0]["m_cov"], ret_flip=c[0]["ret_flip"], gospa_d=c[0]["gospa_d"]),
                             curve_best=dict(m_cov=cb[0]["m_cov"], ret_flip=cb[0]["ret_flip"]) if cb else None,
                             curve_end=dict(m_cov=c[-1]["m_cov"], ret_flip=c[-1]["ret_flip"], gospa_d=c[-1]["gospa_d"])))
        steps = max(r["env_steps"] for r in rows)
        k_cap = sum(r["best_step"] >= r["env_steps"] - 1_000_000 for r in rows)
        rep["training"][a] = dict(lanes=rows, cap_limited_k=int(k_cap), cap_limited=bool(k_cap >= 3), steps=steps,
                                  uniform_entropy=float(2 * np.log(19)),
                                  best_step=ms([r["best_step"] for r in rows]),
                                  ent_last=ms([r["ent_last"] for r in rows]), ev_last=ms([r["ev_last"] for r in rows]),
                                  d_curve_m_cov=ms([r["curve_end"]["m_cov"] - r["curve_start"]["m_cov"] for r in rows]),
                                  d_curve_ret=ms([r["curve_end"]["ret_flip"] - r["curve_start"]["ret_flip"] for r in rows]))
    # ---- P-MIS (declared: Spearman(validation return, -M_cov) over MHSA seeds x checkpoints)
    if "mhsa" in M:
        xs, ys = [], []
        for m in M["mhsa"]:
            for c in m["curve"]:
                if np.isfinite(c["m_cov"]):
                    xs.append(c["ret_flip"])
                    ys.append(-c["m_cov"])
        rho = st.spearmanr(xs, ys)
        rep["pmis"] = dict(rho=float(rho.statistic), p=float(rho.pvalue), n_points=len(xs),
                           checkpoints_per_seed=len(M["mhsa"][0]["curve"]),
                           verdict="GO (aligned)" if rho.statistic > 0.5 else ("KILL (misaligned)" if rho.statistic < 0
                                                                               else "neither"))
    json.dump(rep, open(os.path.join(PB, "report.json"), "w"), indent=1, default=float)
    # ---- text summary
    out = []
    f = lambda x, d=4: "nan" if x is None or not np.isfinite(x) else f"{x:.{d}g}"  # noqa: E731
    for mode, R in rep["modes"].items():
        out.append(f"\n=== mode {mode} ===")
        out.append("| arm | M_cov | Σd | Σmissed | SR | tracks | init | track-bearing eps |")
        for a, r in R["rows"].items():
            c = lambda k: f"{f(r[k]['mean'])} ± {f(r[k]['se'] or 0, 2)}"  # noqa: E731
            out.append(f"| {a} | {c('m_cov')} | {c('gospa_d')} | {c('gospa_miss')} | {c('sr')} | {c('mean_tracks')} | "
                       f"{c('n_init')} | {r['track_bearing_eps']} |")
        out.append("ratios:")
        for k, r in R["ratios"].items():
            if r.get("band") or any(k.endswith(":" + a) for a in ("flat", "bigru", "mhsa", "hand")):
                out.append(f"  {k}: {f(r['mean'])} ± {f(r['se'] or 0, 2)} band {r.get('band')} grade {r.get('grade')} "
                           f"(Random itself {r.get('grade_random_itself')}) Δ% {f(r.get('delta_pct'), 3)}")
        out.append("primary:")
        for h, r in R["primary"].items():
            if "diff" in r:
                out.append(f"  {h} {r['metric']} {r['arm']}<{r['ref']}: {f(r['mean_arm'])} vs {f(r['mean_ref'])}, "
                           f"Δ {f(r['diff'])} ± {f(r['diff_se'], 2)}, boot {[f(x, 3) for x in r['boot95']]}, "
                           f"p1 {f(r['p_one_sided'], 3)}, holm {f(r['p_holm'], 3)}, p2 {f(r['p_two_sided'], 3)}, "
                           f"wins {r['wins_lower']}/5 -> {r['verdict']} / {r['grade']}; pairings {r['pairings']}")
            else:
                out.append(f"  {h}: {r['verdict']}")
        out.append("hand: " + json.dumps({a: {k: dict(diff=v.get("diff", {}).get("mean"), p=v.get("p_two_sided"),
                                                       lr=v.get("learner_ratio", {}).get("mean"),
                                                       hr=v.get("hand_ratio", {}).get("mean"))
                                                for k, v in d.items()} for a, d in R["hand"].items()}, default=float))
        out.append("secondary: " + json.dumps(R["secondary"], default=float))
        out.append("gap: " + json.dumps({k: dict(defined=v["defined"], gap=v["gap"],
                                                  f={a: x["f"] for a, x in v["learners"].items()})
                                         for k, v in R["gap"].items()}, default=float))
        out.append("readings: " + json.dumps({a: dict(t2t=v["t2t"]["mean"], coast=v["mcov_over_coast"]["mean"],
                                                       med=v["mcov_median_ratio"]["mean"])
                                              for a, v in R["readings"].items()}, default=float))
    out.append("\n=== training ===")
    for a, t in rep["training"].items():
        out.append(f"{a}: steps {t['steps']}, best_step {t['best_step']['per_seed']}, cap-limited {t['cap_limited_k']}/5,"
                   f" ent_last {f(t['ent_last']['mean'])} (uniform {f(t['uniform_entropy'])}), ev_last "
                   f"{f(t['ev_last']['mean'])}, d_curve_m_cov {f(t['d_curve_m_cov']['mean'])} ± "
                   f"{f(t['d_curve_m_cov']['se'] or 0, 2)}, d_curve_ret {f(t['d_curve_ret']['mean'])}")
        for r in t["lanes"]:
            out.append(f"   s{r['seed']}: best {r['best_step']} stalled {r['stalled_at_end']} ent {f(r['ent_first'])}->"
                       f"{f(r['ent_last'])} (at best {f(r['ent_at_best'])}) ev {f(r['ev_last'])} kl {f(r['kl_last'], 3)}"
                       f" clip {f(r['clip_last'], 3)} cnn teacher {f(r['cnn_alive_ppo_start_teacher'], 3)} "
                       f"valid start/final/best {[f(r['cnn_alive'][k]['alive'], 3) for k in ('ppo_start', 'final', 'best')]}")
    out.append("P-MIS: " + json.dumps(rep["pmis"], default=float))
    txt = "\n".join(out)
    open(os.path.join(PB, "report.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
