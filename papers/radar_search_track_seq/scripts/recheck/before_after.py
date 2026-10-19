"""Recheck before/after tests (see README): paired comparisons for every affected metric.
    GOSPA ratio, mean/mean (before) vs median/median (after), paired over 5 seeds
   M_cov Coverage/Random per-episode vs pooled reduction (the 5 Random seeds)
    T2T Random and Coverage: primary vs dt5 (sigma_v 30) vs dt5_svrule (sigma_v 18), paired over scenarios
 /12  Coverage - Random per-scenario M_cov (live) within each scene, paired over the 100 scenarios (Random seed 0):
         does Coverage beat Random anywhere?  Static - Random over Static's track-bearing episodes (uniform_speed)
Writes runs/recheck/fixstage/before_after.json."""
import json, sys
import numpy as np
from rl_jax_bench.stats import paired_compare, mean_se  # noqa: E402

R = "runs"
out = {}
B = json.load(open(f"{R}/full/baselines.json"))["arms"]
rnd = {s: {k: np.asarray(v, float) for k, v in d["per_episode"].items()} for s, d in B["random"].items()}
mh = {}
for s in range(5):
    m = json.load(open(f"{R}/full/mhsa/seed_{s}/metrics.json"))["per_episode"]
    mh[str(s)] = {k: np.asarray(v, float) for k, v in m.items()}
for key in ("gospa_d", "gospa_miss"):
    mean_r = [mh[s][key].mean() / rnd[s][key].mean() for s in sorted(mh)]
    med_r = [np.median(mh[s][key]) / np.median(rnd[s][key]) for s in sorted(mh)]
    out[f"gospa_median_{key}"] = dict(before_mean_over_mean=mean_se(mean_r), after_median_over_median=mean_se(med_r),
                              paired=paired_compare(med_r, mean_r))
cov = {k: np.asarray(v, float) for k, v in B["coverage"]["0"]["per_episode"].items()}
per_ep, pooled = [], []
for s in sorted(rnd):
    r = rnd[s]
    per_ep.append(np.nanmean(cov["m_cov"]) / np.nanmean(r["m_cov"]))
    w, wr = cov["frac_steps_tracked"], r["frac_steps_tracked"]
    ok, okr = np.isfinite(cov["m_cov"]), np.isfinite(r["m_cov"])
    pooled.append((np.sum(cov["m_cov"][ok] * w[ok]) / np.sum(w[ok])) / (np.sum(r["m_cov"][okr] * wr[okr]) / np.sum(wr[okr])))
out["pooled_cov_over_rnd"] = dict(before_per_episode=mean_se(per_ep), after_pooled=mean_se(pooled), paired=paired_compare(pooled, per_ep))


def scene(name, d="review"):
    p = f"{R}/review/scene_{name}.json" if d == "review" else f"{R}/recheck/fixstage/scenes/scene_{name}.json"
    A = json.load(open(p))["arms"]
    return {a: {s: {k: np.asarray(v, float) for k, v in x["per_episode"].items()} for s, x in v.items()} for a, v in A.items()}


S = {"primary": scene("primary"), "dt5": scene("dt5"), "dt5_svrule": scene("dt5_svrule", "fix"),
     "uniform_speed": scene("uniform_speed", "fix"), "filt_sv10": scene("filt_sv10", "fix"), "frob": scene("frob"),
     "dt5_frob": scene("dt5_frob")}
for arm in ("random", "coverage"):
    t = {n: S[n][arm]["0"]["n_init"] / 3 for n in S}
    out[f"T2T_{arm}"] = {n: mean_se(v) for n, v in t.items()}
    out[f"T2T_{arm}_dt5_svrule_vs_dt5"] = paired_compare(t["dt5_svrule"], t["dt5"])
    out[f"T2T_{arm}_dt5_svrule_vs_primary"] = paired_compare(t["dt5_svrule"], t["primary"])
for n in S:
    c, r = S[n]["coverage"]["0"]["m_cov"], S[n]["random"]["0"]["m_cov"]
    ok = np.isfinite(c) & np.isfinite(r)
    out[f"cov_minus_rnd_{n}"] = dict(n_pairs=int(ok.sum()), paired=paired_compare(c[ok], r[ok]))
    st = np.concatenate([S[n]["static"][s]["m_cov"] for s in sorted(S[n]["static"])])
    rr = np.concatenate([S[n]["random"][s]["m_cov"] for s in sorted(S[n]["static"])])
    ok = np.isfinite(st) & np.isfinite(rr)
    out[f"static_minus_rnd_{n}"] = dict(n_pairs=int(ok.sum()),
                                        paired=paired_compare(st[ok], rr[ok]) if ok.sum() >= 2 else None)
json.dump(out, open(f"{R}/recheck/fixstage/before_after.json", "w"), indent=1)
for k, v in out.items():
    if "paired" in v and v["paired"]:
        p = v["paired"]
        print(k, v.get("n_pairs", ""), "diff %.4g ± %.3g  t95 [%.4g, %.4g]  p %.3g  n %d" % (p["diff_mean"], p["diff_se"], *p["t95"], p["t_p"], p["n"]))
    elif "diff_mean" in v:
        print(k, "diff %.4g ± %.3g  t95 [%.4g, %.4g]  p %.3g n %d" % (v["diff_mean"], v["diff_se"], *v["t95"], v["t_p"], v["n"]))
    else:
        print(k, {a: (round(b[0], 3), round(b[1], 4) if b[1] else None) for a, b in v.items()} if isinstance(v, dict) else v)
