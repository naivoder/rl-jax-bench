"""Recheck fix stage (see README): readings of STORED full-scale
outputs -- no environment re-run. Writes runs/recheck/fixstage/readings.json.

   T2T = n_init / N_tgt (companion Table VI 'Mean T2T', Stone Soup BasicMetrics tracks/truths)
   M_cov / never-revisited coast value of the same scene (companion Fig. 7d point mass at Static 2630.72)
   SR(PPO)/SR(Random) re-declared band [0.127/0.137, 0.137/0.127]
   Sigma GOSPA d and missed: median/median per seed vs the paper's Fig. 1 median/median references
  pooled (track-step weighted) vs per-episode M_cov reduction
   absolute SR under ssv_norm = "single" (SR_fifo * p_max / phi_0(0), exact)
   joint check of the SR means (T = 0 or T = N_SV-1 normaliser) and the per-step CoVs, from the existing sweeps
     median M_cov ratios vs companion Fig. 7d medians
"""
import json
import math
import os
import sys

import numpy as np

R = "."
G3 = R + "/runs/full"
SA = R + "/runs/recheck"
from rl_jax_bench.stats import paired_compare  # noqa: E402,F401

N_TGT = 3
COAST_PRIMARY = json.load(open(SA + "/measurement/probe_coast.json"))["primary_50ms_trace_r20km"]["coast_mean_live"]
PAPER = dict(
    t2t=dict(random=1.585, coverage=0.942, static=0.288, flat=1.407, bigru=1.430, mhsa=1.152),
    mcov_mean=dict(random=948.15, coverage=574.27, static=2630.72, flat=727.0, bigru=765.0, mhsa=391.0),
    mcov_median_fig7d=dict(random=551, coverage=453, static=2629, flat=343, bigru=411, mhsa=270),
    gospa_median=dict(d=dict(mhsa=1.187e7, random=1.247e7), miss=dict(mhsa=1.100e11, random=1.308e11)),
)


def grade(x, lo, hi, refs):
    if lo <= x <= hi or min(abs(x / r - 1) for r in refs) <= 0.05:
        return "MATCH"
    if min(abs(x / r - 1) for r in refs) <= 0.20:
        return "CLOSE"
    return "DIVERGENT"


def ms(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None, n=len(v),
                per_seed=[float(x) for x in v])


def load():
    b = json.load(open(G3 + "/baselines.json"))["arms"]
    arms = {k: {s: {kk: np.asarray(vv, float) for kk, vv in d["per_episode"].items()} for s, d in v.items()}
            for k, v in b.items()}
    for arm in ("flat", "bigru", "mhsa", "mhsa_noae", "mhsa_nobc", "mhsa_lit", "mhsa_bcmse"):
        d = G3 + "/" + arm
        if not os.path.isdir(d):
            continue
        arms[arm] = {}
        arms[arm + "_det"] = {}
        for sd in sorted(os.listdir(d)):
            m = json.load(open(f"{d}/{sd}/metrics.json"))
            if "per_episode" not in m:
                continue
            s = sd.split("_")[1]
            arms[arm][s] = {k: np.asarray(v, float) for k, v in m["per_episode"].items()}
            if m.get("per_episode_det"):
                arms[arm + "_det"][s] = {k: np.asarray(v, float) for k, v in m["per_episode_det"].items()}
    return arms


def main():
    A = load()
    out = {"coast_value_primary_live": COAST_PRIMARY}
    rnd = A["random"]
    # --- T2T, coast-normalised, medians, pooled
    rows = {}
    for arm, seeds in A.items():
        if not seeds:
            continue
        t2t, cn, med_r, mean_r, pool_r, pooled = [], [], [], [], [], []
        for s, pe in seeds.items():
            rp = rnd[s] if s in rnd else rnd["0"]
            t2t.append(pe["n_init"].mean() / N_TGT)
            m = pe["m_cov"][np.isfinite(pe["m_cov"])]
            mr = rp["m_cov"][np.isfinite(rp["m_cov"])]
            if m.size:
                cn.append(m.mean() / COAST_PRIMARY)
                med_r.append(np.median(m) / np.median(mr))
                mean_r.append(m.mean() / mr.mean())
            w, wr = pe["frac_steps_tracked"], rp["frac_steps_tracked"]
            ok, okr = np.isfinite(pe["m_cov"]) & (w > 0), np.isfinite(rp["m_cov"]) & (wr > 0)
            if ok.any():
                p = np.sum(pe["m_cov"][ok] * w[ok]) / np.sum(w[ok])
                pr = np.sum(rp["m_cov"][okr] * wr[okr]) / np.sum(wr[okr])
                pool_r.append(p / pr)
        rows[arm] = dict(t2t=ms(t2t), n_track_bearing_eps=[int(np.isfinite(pe["m_cov"]).sum()) for pe in seeds.values()],
                         mcov_over_coast=ms(cn) if cn else None, mcov_median_ratio=ms(med_r) if med_r else None,
                         mcov_mean_ratio=ms(mean_r) if mean_r else None, mcov_pooled_ratio=ms(pool_r) if pool_r else None)
    out["per_arm"] = rows
    out["paper"] = dict(t2t=PAPER["t2t"],
                        mcov_over_static_point_mass={k: v / 2630.72 for k, v in PAPER["mcov_mean"].items()},
                        mcov_median_ratio_fig7d={k: v / PAPER["mcov_median_fig7d"]["random"]
                                                 for k, v in PAPER["mcov_median_fig7d"].items()},
                        mcov_mean_ratio={k: v / PAPER["mcov_mean"]["random"] for k, v in PAPER["mcov_mean"].items()})
    # --- GOSPA median/median (MHSA vs Random, paired by seed index) and the Random-itself grade
    g = {}
    for key, ref, lo, hi in (("gospa_d", 1.187 / 1.247, (1.187 - 0.0017) / (1.247 + 0.0017), (1.187 + 0.0017) / (1.247 - 0.0017)),
                             ("gospa_miss", 1.100 / 1.308, (1.100 - 0.0024) / (1.308 + 0.0024), (1.100 + 0.0024) / (1.308 - 0.0024))):
        med = [np.median(A["mhsa"][s][key]) / np.median(rnd[s][key]) for s in sorted(A["mhsa"])]
        mean = [A["mhsa"][s][key].mean() / rnd[s][key].mean() for s in sorted(A["mhsa"])]
        x = float(np.mean(med))
        g[key] = dict(paper_ref_median_over_median=ref, band=[lo, hi], ours_median_ratio=ms(med),
                      ours_mean_ratio=ms(mean), delta_pct=100 * (x / ref - 1),
                      grade=grade(x, lo, hi, [ref]), grade_random_itself=grade(1.0, lo, hi, [ref]),
                      old_ref=0.90 if key == "gospa_d" else 0.79,
                      old_grade=grade(float(np.mean(mean)), 0.871 if key == "gospa_d" else 0.79,
                                      0.938 if key == "gospa_d" else 0.79, [0.90 if key == "gospa_d" else 0.79]),
                      old_grade_random_itself=grade(1.0, 0.871 if key == "gospa_d" else 0.79,
                                                    0.938 if key == "gospa_d" else 0.79, [0.90 if key == "gospa_d" else 0.79]))
        for arm in ("flat", "bigru"):
            g[key][arm + "_median_ratio"] = ms([np.median(A[arm][s][key]) / np.median(rnd[s][key]) for s in sorted(A[arm])])
    out["gospa_median"] = g
    # --- SR band
    lo, hi = 0.127 / 0.137, 0.137 / 0.127
    sr = {}
    for arm in ("flat", "bigru", "mhsa"):
        r = [A[arm][s]["sr"].mean() / rnd[s]["sr"].mean() for s in sorted(A[arm])]
        x = float(np.mean(r))
        sr[arm] = dict(ratio=ms(r), new_band=[lo, hi], new_grade=grade(x, lo, hi, [lo, hi]),
                       old_band=[0.96, 1.04], old_grade=grade(x, 0.96, 1.04, [0.96, 1.04]))
    sr["random_itself"] = dict(new_grade=grade(1.0, lo, hi, [lo, hi]), old_grade=grade(1.0, 0.96, 1.04, [0.96, 1.04]))
    out["sr_band"] = sr
    # --- Static band (SD 0.0236 instead of 2358.40)
    r = 2630.72 / 948.15
    se_new = r * math.sqrt((0.0236 / (10 * 2630.72)) ** 2 + (922.0 / (10 * 948.15)) ** 2)
    se_old = r * math.sqrt((2358.40 / (10 * 2630.72)) ** 2 + (922.0 / (10 * 948.15)) ** 2)
    out["static_band"] = dict(ratio=r, new=[r - se_new, r + se_new], old=[r - se_old, r + se_old])
    # --- absolute SR under the single-scan normaliser (frozen gamma 0.01, N_SV 100, radial)
    s0 = 4.5 / math.sqrt(2 * math.log(20.0))
    phi0 = 1 / (2 * math.pi * s0 ** 2)
    pmax = sum(1 / (2 * math.pi * (s0 * 1.01 ** k) ** 2) for k in range(100))
    fac = pmax / phi0
    absr = {}
    for arm in ("static", "random", "coverage", "hand", "ceiling", "flat", "bigru", "mhsa"):
        v = [A[arm][s]["sr"].mean() for s in sorted(A[arm])]
        absr[arm] = dict(fifo=ms(v), single=ms(np.array(v) * fac))
    out["sr_absolute"] = dict(factor_pmax_over_phi0=fac, arms=absr,
                              paper=dict(static=-25.2, random=[-0.137, -0.127], coverage=-0.0502))
    # --- joint means + per-step CoV check (existing sweeps)
    tgt_mean = dict(static=-25.2, random=-0.132, coverage=-0.0502)
    tgt_cov = dict(static=6.48 / 25.2, coverage=0.0344 / 0.0502)
    sc = json.load(open(SA + "/measurement/probe_ssv_scale.json"))
    joint = []
    for c in sc:
        le_mean = dict(static=math.log(c["static_mean_phi0"] / 25.2), random=math.log(c["rnd_mean_phi0"] / 0.132),
                       coverage=math.log(c["cov_mean_phi0"] / 0.0502))
        le_cov = dict(static=math.log(c["cv_static"] / tgt_cov["static"]), coverage=math.log(c["cv_cov"] / tgt_cov["coverage"]))
        fit_mean = all(abs(v) <= math.log(1.25) for v in le_mean.values())
        fit_cov = all(abs(v) <= math.log(1.25) for v in le_cov.values())
        joint.append(dict(mode=c["mode"], g=c["g"], nsv=c["nsv"], after_push=c["after_push"], logerr_mean_T0=le_mean,
                          logerr_cov=le_cov, fit_means_T0=fit_mean, fit_covs=fit_cov, fit_both=fit_mean and fit_cov,
                          static_mean_pmax=c["static_mean_pmax"]))
    fifo_fits_static = [j for j in joint if abs(math.log(j["static_mean_pmax"] / 25.2)) <= math.log(1.25)]
    out["ssv_joint"] = dict(
        n_cells=len(joint), n_fit_means_T0=sum(j["fit_means_T0"] for j in joint),
        n_fit_covs=sum(j["fit_covs"] for j in joint), n_fit_both=sum(j["fit_both"] for j in joint),
        n_fifo_static_within_25pct=len(fifo_fits_static),
        cells_fit_means_T0=[j for j in joint if j["fit_means_T0"]], cells_fit_covs=[j for j in joint if j["fit_covs"]],
        targets=dict(means=tgt_mean, covs=tgt_cov, tol="each within 25 % (|log err| <= log 1.25)"),
        grid_axes=sorted({(j["mode"], j["after_push"]) for j in joint}),
        gammas=sorted({j["g"] for j in joint}), nsvs=sorted({j["nsv"] for j in joint}))
    fit = json.load(open(SA + "/equations/probe_ssv_fit.json"))
    out["ssv_fit_T0_grid_best"] = sorted(fit, key=lambda r: r["rms_logerr"])[:3]
    json.dump(out, open(SA + "/fixstage/readings.json", "w"), indent=1, default=str)
    # brief print
    for arm, r in rows.items():
        print(f"{arm:14s} T2T {r['t2t']['mean']:.3f}  /coast {r['mcov_over_coast']['mean'] if r['mcov_over_coast'] else float('nan'):.3f}"
              f"  medR {r['mcov_median_ratio']['mean'] if r['mcov_median_ratio'] else float('nan'):.3f}"
              f"  meanR {r['mcov_mean_ratio']['mean'] if r['mcov_mean_ratio'] else float('nan'):.3f}"
              f"  poolR {r['mcov_pooled_ratio']['mean'] if r['mcov_pooled_ratio'] else float('nan'):.3f}  eps {r['n_track_bearing_eps']}")
    print("gospa", json.dumps({k: (round(v['ours_median_ratio']['mean'], 4), round(v['ours_median_ratio']['se'], 4), round(v['delta_pct'], 1), v['grade'], v['grade_random_itself'], v['old_grade'], v['old_grade_random_itself']) for k, v in g.items()}))
    print("sr band", json.dumps({k: (v.get('ratio', {}).get('mean'), v['new_grade'], v['old_grade']) for k, v in sr.items()}))
    print("static band", out["static_band"])
    print("sr abs", fac, {k: (round(v['fifo']['mean'], 6), round(v['single']['mean'], 4)) for k, v in absr.items()})
    print("ssv joint", {k: v for k, v in out["ssv_joint"].items() if k.startswith("n_")}, out["ssv_joint"]["gammas"], out["ssv_joint"]["nsvs"])


if __name__ == "__main__":
    main()
