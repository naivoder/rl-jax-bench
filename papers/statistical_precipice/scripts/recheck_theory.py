"""recheck re-analysis the README,  ..). CPU only, no training.

    finite-population (FPC) form of the D10 coverage prediction and the FPC target, applied to T2, the §3.5
         prediction column, the M3 ranking and the §3.4 'CI covers pool diff' reading
    T1 / T2 on the paper's Fig 6 VECTOR read (runs/full/paper_readoff.json['fig6']); --fig6 eyeballed = pre-fix
    T3 on ONE fresh key set, IQM mode c at every N, kappa with MC-error propagation, b = a/N + c/N^2 chi^2 rule;
         the pre-fix fit (mixed key sets, OLS SE) is recomputed beside it from the stored JSON
    paper-side T3 on Fig A.17 (runs/full/paper_a17.json)
    toy checks D3 re-run with the average-score profile judged against its own target (same key as toys.py)
    Fig 10 right: paper (runs/full/paper_fig10.json) vs ours (runs/full/ale.json) rows

    JAX_PLATFORMS=cpu python scripts/recheck_theory.py
        [--fig6 vector|eyeballed] [--out runs/full/recheck_theory.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import optimize, stats as st  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import eval as E  # noqa: E402

G3 = os.path.join(REPO, "runs", "full")
NG = [3, 5, 10, 20, 50, 100]
# pre-fix literal, copied verbatim from scripts/full_e6.py PAPER_FIG6_EYEBALLED the README
EYE = {"median": dict(percentile=[86, 92, 96, 98, 99, 100], width=[.087, .078, .063, .050, .035, .023]),
       "iqm": dict(percentile=[86, 90, 93, 94, 96, 99], basic=[85, 88, 90, 95, 96, 99], bc=[86, 90, 93, 94, 96, 99],
                   bca=[85, 89, 93, 94, 94, 99], width=[.068, .058, .042, .031, .022, .015])}
SA_STAT = 20   # fresh branch: branch_key(0, EXP_IDS['e6'], 0, 20) -- never used by any earlier run


def J(p):
    return json.load(open(p))


def powfit(N, y):
    r = st.linregress(np.log(N), np.log(y))
    return dict(beta=-r.slope, se=r.stderr, r2=r.rvalue ** 2)


def paper_fig6(src):
    if src == "eyeballed":
        return {s: dict(cov=[v / 100 for v in EYE[s]["percentile"]], width=EYE[s]["width"]) for s in EYE}
    f = J(os.path.join(G3, "paper_readoff.json"))["fig6"]
    return {s: dict(cov=f[s]["coverage"]["percentile"], width=f[s]["width"]["percentile"],
                    cov_all={m: f[s]["coverage"][m] for m in ("percentile", "basic", "bc", "bca")}) for s in f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig6", default="vector", choices=["vector", "eyeballed"])
    ap.add_argument("--out", default=os.path.join(G3, "recheck_theory.json"))
    ap.add_argument("--t3-reps", type=int, default=100_000)
    a = ap.parse_args()
    out = dict(fig6_source=a.fig6)
    P = paper_fig6(a.fig6)
    decl = J(os.path.join(G3, "e6_declared.json"))
    e6 = J(os.path.join(G3, "e6.json"))

    # ------------------------------------------------------------------ T1
    law = np.sqrt(np.array(NG) - 1.0) / np.array(NG)
    t1 = dict(law_beta=powfit(NG, law)["beta"])
    for s in ("iqm", "median"):
        f = powfit(NG, P[s]["width"])
        f.update(se_from_law=(f["beta"] - t1["law_beta"]) / f["se"], se_from_half=(0.5 - f["beta"]) / f["se"],
                 width_over_law_norm=(np.array(P[s]["width"]) / law / (P[s]["width"][0] / law[0])).tolist())
        t1[f"paper_{s}"] = f
    for arm in ("separate", "shared", "advnorm_batch"):
        for s in ("iqm", "mean"):
            K = [3, 5, 10, 20, 50]
            t1[f"e6_{arm}_{s}"] = powfit(K, [decl["coverage"][f"{arm}/{s}/K{k}"]["percentile"]["width"] for k in K])
    out["T1"] = t1

    # ------------------------------------------------------------------ T2 + §3.5 column
    t2 = dict(paper=[], e6_shared_iqm=[])
    for i, n in enumerate(NG):
        row = dict(N=n, paper_iqm_pct=P["iqm"]["cov"][i], d10=E.predicted_mean_coverage(n, 26),
                   d10_fpc=E.predicted_mean_coverage(n, 26, R=200), fpc_target=E.fpc_target_coverage(n, 200))
        row.update(gap_d10_pp=100 * (row["paper_iqm_pct"] - row["d10"]), gap_fpc_pp=100 * (row["paper_iqm_pct"] - row["d10_fpc"]))
        t2["paper"].append(row)
    for k in (3, 5, 10, 20, 50):
        c = decl["coverage"][f"shared/iqm/K{k}"]["percentile"]["coverage"]
        row = dict(K=k, cov=c, d10=E.predicted_mean_coverage(k, 2), d10_fpc=E.predicted_mean_coverage(k, 2, R=100),
                   fpc_target=E.fpc_target_coverage(k, 100))
        row.update(gap_d10_pp=100 * (c - row["d10"]), gap_fpc_pp=100 * (c - row["d10_fpc"]))
        t2["e6_shared_iqm"].append(row)
    out["T2"] = t2

    # ------------------------------------------------------------------ M3 ranking against 0.95 and against the FPC target
    meths = ("percentile", "basic", "bc", "bca")
    m3 = []
    for arm in ("separate", "shared", "advnorm_batch"):
        for s in ("iqm", "mean"):
            for k in (10, 20):
                cv = {m: decl["coverage"][f"{arm}/{s}/K{k}"][m]["coverage"] for m in meths}
                tgt = E.fpc_target_coverage(k, 100)
                by95 = min(meths, key=lambda m: abs(cv[m] - 0.95))
                byT = min(meths, key=lambda m: abs(cv[m] - tgt))
                m3.append(dict(cell=f"{arm}/{s}/K{k}", cov=cv, fpc_target=tgt, closest_to_95=by95, closest_to_fpc=byT,
                               pct_gap_to_closest_fpc_pp=100 * (abs(cv["percentile"] - tgt) - abs(cv[byT] - tgt))))
    out["M3"] = dict(cells=m3, pct_closest_95=sum(r["closest_to_95"] == "percentile" for r in m3),
                     pct_closest_fpc=sum(r["closest_to_fpc"] == "percentile" for r in m3))
    if a.fig6 == "vector":   # the paper's own Fig 6 under the same ranking (informational)
        pm = []
        for s in ("iqm", "median"):
            for i, n in enumerate(NG[:4]):
                tgt = E.fpc_target_coverage(n, 200)
                cv = {m: P[s]["cov_all"][m][i] for m in meths}
                pm.append(dict(stat=s, N=n, cov=cv, fpc_target=tgt, closest_to_fpc=min(meths, key=lambda m: abs(cv[m] - tgt)),
                               closest_to_95=min(meths, key=lambda m: abs(cv[m] - 0.95))))
        out["M3"]["paper_fig6_ranking"] = pm

    # ------------------------------------------------------------------ §3.4 coverage reading against the FPC target
    out["fewrun_cover"] = [dict(pair=pr, n=n, covers=decl["fewrun"][f"{pr}/n{n}"]["ci_covers_pool_diff"],
                                fpc_target=E.fpc_target_coverage(n, 100), d10_fpc=E.predicted_mean_coverage(n, 2, R=100))
                           for pr in ("separate-shared", "separate-advnorm_batch") for n in (3, 5, 10, 20)]

    # ------------------------------------------------------------------ T3 E6 side
    key = A.branch_key(0, A.EXP_IDS["e6"], 0, SA_STAT)
    Ns = [3, 5, 10, 25, 50]
    t3 = {}
    rng = np.random.default_rng(20261016)
    for ai, arm in enumerate(("separate", "shared", "advnorm_batch")):
        pool = jnp.asarray(np.array(e6["per_seed_normalised"][arm]), jnp.float64)
        truth = float(A.agg_iqm(pool, "c"))
        b, se = [], []
        for n in Ns:
            v = np.asarray(A.sampling_distribution(jax.random.fold_in(jax.random.fold_in(key, ai), n), pool,
                                                   lambda x: A.agg_iqm(x, "c"), n, a.t3_reps, replace=True))
            b.append(float(v.mean() - truth))
            se.append(float(v.std(ddof=1) / math.sqrt(v.size)))
        b, se, Nn = np.array(b), np.array(se), np.array(Ns, float)
        ols = st.linregress(np.log(Nn), np.log(np.abs(b)))
        draws = rng.normal(b, se, (4000, len(Ns)))
        ks = np.array([-st.linregress(np.log(Nn), np.log(np.abs(d))).slope for d in draws])
        X = np.stack([1 / Nn, 1 / Nn ** 2], 1)
        W = 1 / se ** 2
        coef = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * b))
        chi2 = float(np.sum(((b - X @ coef) / se) ** 2))
        t3[arm] = dict(truth=truth, N=Ns, bias=b.tolist(), bias_se=se.tolist(), bN=(b * Nn).tolist(),
                       kappa=-ols.slope, kappa_se_ols=ols.stderr, kappa_sd_mc=float(ks.std()),
                       kappa_se_total=float(math.hypot(ols.stderr, ks.std())),
                       a_over_N_plus_c_over_N2=dict(a=float(coef[0]), c=float(coef[1]), chi2=chi2, dof=3,
                                                    p=float(st.chi2.sf(chi2, 3))),
                       consistent_O1N_leading=bool(st.chi2.sf(chi2, 3) >= 0.01))
    # pre-fix (legacy) fit, recomputed from the stored JSON exactly as the README did (mode c N = 3, 5, 25 from
    # e6_sens.json + mode a N = 10, 50 from e6.json, plain OLS SE)
    sens = J(os.path.join(G3, "e6_sens.json"))["iqm_mode_c"]["sampling"]
    for arm in t3:
        bl = {3: sens[f"{arm}/iqm_c/N3"]["bias"], 5: sens[f"{arm}/iqm_c/N5"]["bias"], 25: sens[f"{arm}/iqm_c/N25"]["bias"],
              10: e6["sampling"][f"{arm}/iqm/N10"]["bias"], 50: e6["sampling"][f"{arm}/iqm/N50"]["bias"]}
        r = st.linregress(np.log(Ns), np.log(np.abs([bl[n] for n in Ns])))
        t3[arm]["legacy_fit"] = dict(kappa=-r.slope, se=r.stderr)
    out["T3_e6"] = t3

    # ------------------------------------------------------------------ T3 paper side, Fig A.17
    a17 = J(os.path.join(G3, "paper_a17.json"))
    Np = np.array(a17["N"], float)
    pt3 = {}
    for nm, d in a17["panels"].items():
        y = np.array(d["expected_iqm"])
        sig = d["units_per_px"]
        bias = y[:-1] - y[-1]
        dec = bool(np.all(np.diff(y) < 0))
        dec_tol = bool(np.all(np.diff(y) < 2 * sig))
        f1 = np.polyfit(1 / Np, y, 1)
        res1 = (y - np.polyval(f1, 1 / Np)) / sig
        try:
            popt, pcov = optimize.curve_fit(lambda n, th, c, k: th + c * n ** (-k), Np, y, p0=[y[-1], 0.01, 1.0],
                                            sigma=np.full(6, sig), absolute_sigma=True, maxfev=20000)
            kap, kse = float(popt[2]), float(math.sqrt(pcov[2, 2]))
        except RuntimeError:
            kap, kse = float("nan"), float("nan")
        ci = [kap - 1.96 * kse, kap + 1.96 * kse]
        pt3[nm] = dict(expected_iqm=y.tolist(), bias_vs_N100=bias.tolist(), read_sigma=sig, strictly_decreasing=dec,
                       decreasing_within_2px=dec_tol, kappa_free=kap, kappa_free_se=kse, kappa_ci95=ci,
                       kappa1_fit=dict(theta=float(f1[1]), c=float(f1[0]), max_abs_resid_px=float(np.abs(res1).max())),
                       rule_strict=bool(np.all(bias > 0) and dec and (ci[0] <= 1 <= ci[1] or np.abs(res1).max() <= 2)),
                       rule_within_read_error=bool(np.all(bias > 0) and dec_tol and (ci[0] <= 1 <= ci[1] or np.abs(res1).max() <= 2)))
    out["T3_paper_a17"] = pt3

    # ------------------------------------------------------------------ toy checks D3, toys.py's key k(4)
    k4 = jax.random.fold_in(A.branch_key(0, A.EXP_IDS["toys"]), 4)
    d3 = E.toys_profile_unbiased(k4, reps=20_000, N=3)
    out["toys_d3"] = {k: v for k, v in d3.items() if k not in ("taus", "truth", "run_profile_mean")}

    # ------------------------------------------------------------------ Fig 10 right rows
    pf = J(os.path.join(G3, "paper_fig10.json"))["agents"]
    ours = J(os.path.join(G3, "ale.json"))["fig10_iqm_curve"]
    rows = []
    for ag in ("C51", "IQN", "Rainbow", "DQN (Nature)", "DQN (Adam)"):
        pp, oc = pf[ag], ours[ag]
        for i, fr in enumerate([1, 10, 25, 50, 75, 100, 125, 150, 175, 200]):
            po, lo, hi = oc["point"][i], oc["ci"][0][i], oc["ci"][1][i]
            rel = abs(po - pp["point"][i]) / abs(pp["point"][i])
            inside = pp["lo"][i] <= po <= pp["hi"][i]
            erel = max(abs(lo - pp["lo"][i]) / abs(pp["lo"][i]), abs(hi - pp["hi"][i]) / abs(pp["hi"][i]))
            rows.append(dict(agent=ag, frame_paper=fr, frame_ours=oc["frames_M"][i], paper=pp["point"][i], ours=po,
                             abs_diff=abs(po - pp["point"][i]), rel=rel, inside_paper_band=inside,
                             point_judgment="MATCH" if (rel <= 0.05 or inside) else ("CLOSE" if rel <= 0.20 else "DIVERGENT"),
                             paper_band=[pp["lo"][i], pp["hi"][i]], ours_band=[lo, hi],
                             band_abs_diff=[abs(lo - pp["lo"][i]), abs(hi - pp["hi"][i])], band_rel_worse=erel,
                             band_judgment="MATCH" if erel <= 0.05 else ("CLOSE" if erel <= 0.20 else "DIVERGENT")))
    out["fig10_right"] = rows
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("[INFO] wrote", a.out)
    print("T1", {k: (round(v["beta"], 3), round(v.get("se", 0), 3)) for k, v in t1.items() if isinstance(v, dict)})
    print("T2 paper", [(r["N"], round(r["paper_iqm_pct"], 3), round(r["d10"], 4), round(r["d10_fpc"], 4)) for r in t2["paper"]])
    print("M3 percentile closest: to .95", out["M3"]["pct_closest_95"], "to FPC", out["M3"]["pct_closest_fpc"])
    for arm, r in t3.items():
        print("T3", arm, round(r["kappa"], 3), "+-", round(r["kappa_se_total"], 3), "chi2 p", round(r["a_over_N_plus_c_over_N2"]["p"], 4),
              "legacy", round(r["legacy_fit"]["kappa"], 3))
    for nm, r in pt3.items():
        print("A17", nm, round(r["kappa_free"], 2), "+-", round(r["kappa_free_se"], 2), "k1 resid px", round(r["kappa1_fit"]["max_abs_resid_px"], 2),
              r["rule_strict"], r["rule_within_read_error"])
    print("D3", out["toys_d3"]["pass_"], [(r["N"], round(r["max_abs_bias_vs_own_target"], 4), round(r["avg_gap_to_run_tail"], 4)) for r in d3["avg_rows"]])
    print("Fig10", {j: sum(r["point_judgment"] == j for r in rows if r["agent"] in ("C51", "IQN", "Rainbow") and r["frame_paper"] <= 175)
                    for j in ("MATCH", "CLOSE", "DIVERGENT")})


if __name__ == "__main__":
    main()
