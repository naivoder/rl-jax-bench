"""E0 differential test: our JAX estimators (src/algorithm.py) vs rliable@3ccd9f4 outputs saved by
scripts/official/rliable_reference.py the README). Writes runs/e0/difftest.json.

    python scripts/e0_difftest.py [--oracle data/e0/oracle.npz]

Criteria (pre-declared in the README before any run):
  points      |ours - oracle| <= 1e-12 (float64) for every aggregate, trimmed mean, profile value, P(X>Y)
  replay      the oracle's own np.random indices replayed through our gather + statistic: |diff| <= 1e-12
  in-dist     different RNGs, B = 50,000 each side: bootstrap mean within 4 SE; every CI endpoint (percentile,
              basic, bc, bca) within 4 x its Monte-Carlo SE (quantile SE via the bootstrap density, plus the z0 term
              for bc/bca); two-sample KS p-values Holm-corrected across the family, min > 1e-3
  ranks       |ours - oracle| <= 4 sqrt(2 p (1-p) / n) per entry
  studentized exact (1e-12)
AMENDMENT 1 (2026-10-16, after the --quick oracle run, before the full one; reasons in the READMEsmall-scale run):
  * CI endpoints: "4 x MC SE via the bootstrap density" is replaced by a QUANTILE BAND: each side's endpoint must lie
    within q(p +- delta) of the pooled replicate distribution, delta = 4 s_p + 2/B, s_p = sqrt(p(1-p)/B) for that
    side's B (plus the z0 term for bc/bca); tested as |diff| <= half_band(B_o) + half_band(B_r). For smooth
    distributions this is ~5.7 SE of the difference (old rule: 4); the old rule's density estimate is 0 between atoms
    of discrete statistics (median / OG of 3-run inputs) and failed there.
    The 2/B term: at the extreme levels bc/bca reach on discrete medians
    (p ~ 1e-6 .. 1e-3, i.e. < 1 expected replicate below the level) the endpoint is a sample minimum, whose rank
    noise the normal s_p does not describe; two ranks of slack cover the order-statistic discreteness.
  * KS: replicate values are rounded to 1e-12 before the test (1-ulp summation-order differences split atoms).
  * studentized: 1e-10 (scipy 1.13.1 in the oracle venv returns t_{.975,9} = 2.2621571628540993, scipy 1.18.1
    2.262157162798205; mpmath 2.26215716279820551 -- the oracle's library is off by 6e-11).
"""
from __future__ import annotations

import argparse
import os
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "external", "data"))  # large inputs, see README
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, warn, write_json  # noqa: E402
from scipy import stats as st  # noqa: E402
from rl_jax_bench.stats import holm  # noqa: E402

from src import algorithm as A  # noqa: E402
from src.data import sha256  # noqa: E402

METHODS = ("percentile", "basic", "bc", "bca")
TRIMS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)


def half_band(dist_pooled, base, method, B, a=0.0, level=0.95):
    """One side's allowed deviation from the pooled quantile (AMENDMENT 1): max one-sided distance from q(p) to
    q(p +- delta), delta = 4 s_p + 2/B, s_p the MC SE in probability for B replicates (plus the z0 term for bc/bca)."""
    d = np.sort(dist_pooled)
    alpha = (1 - level) / 2
    zq = st.norm.ppf([alpha, 1 - alpha])
    if method in ("percentile", "basic"):
        p = np.array([alpha, 1 - alpha])
        var_p = p * (1 - p) / B
    else:
        P = float(np.mean(d < base))
        if P <= 0 or P >= 1:
            return np.array([np.nan, np.nan])

        def pfun(Pv):
            z0 = st.norm.ppf(Pv)
            return st.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))

        p = pfun(P)
        eps = 1e-4
        dpdP = (pfun(min(P + eps, 1 - 1e-9)) - pfun(max(P - eps, 1e-9))) / (2 * eps)
        var_p = (p * (1 - p) + dpdP ** 2 * P * (1 - P)) / B
    dl = 4 * np.sqrt(var_p) + 2.0 / B   # + 2 ranks: order-statistic discreteness at extreme p
    q0 = np.quantile(d, p)
    return np.maximum(np.quantile(d, np.clip(p + dl, 0, 1)) - q0, q0 - np.quantile(d, np.clip(p - dl, 0, 1)))


def endpoint_se(dist_pooled, base, method, B_o, B_r, level=0.95, a=0.0):
    """Tolerance / 4: |ours - oracle| <= half_band(B_o) + half_band(B_r) (each endpoint within +-4 SE of the pooled
    quantile; triangle inequality). For a smooth distribution this is ~5.7 SE of the difference."""
    return (half_band(dist_pooled, base, method, B_o, a, level) + half_band(dist_pooled, base, method, B_r, a, level)) / 4.0


def endpoint_se_bca(dist, base, a, B_o, B_r, level=0.95):
    return endpoint_se(dist, base, "bca", B_o, B_r, level, a)


def r12(v):
    return np.round(np.asarray(v, np.float64), 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=os.path.join(DATA_ROOT, "e0", "oracle.npz"))
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "e0", "difftest.json"))
    a = ap.parse_args()
    check_device()
    T = Timer()
    z = np.load(os.path.join(DATA_ROOT, "e0", "inputs.npz"))
    o = np.load(a.oracle)
    reps = int(o["meta_reps"])
    res = dict(oracle_file=a.oracle, oracle_sha256=sha256(a.oracle), inputs_sha256=sha256(os.path.join(DATA_ROOT, "e0", "inputs.npz")),
               reps=reps)

    # ------------------------------------------------------------------ points
    worst = {}

    def upd(name, diff):
        worst[name] = max(worst.get(name, 0.0), float(np.nanmax(np.abs(diff))) if np.size(diff) else 0.0)

    for i in range(200):
        x, y = jnp.asarray(z[f"x{i:03d}"]), jnp.asarray(z[f"y{i:03d}"])
        ours = np.array([A.agg_mean(x), A.agg_median(x), A.agg_iqm(x), A.agg_og(x), A.agg_og(x, 0.5)])
        upd("aggregates", ours - o[f"pt_{i:03d}_agg"])
        upd("trimmed_means", np.array([A.trimmed_mean(x, t) for t in TRIMS]) - o[f"pt_{i:03d}_trim"])
        t = o[f"pt_{i:03d}_taus"]
        upd("score_profile", np.asarray(A.score_profile(x, t)) - o[f"pt_{i:03d}_prof"])
        upd("avg_score_profile", np.asarray(A.avg_score_profile(x, t)) - o[f"pt_{i:03d}_avgprof"])
        upd("poi", np.array([A.poi(x, y), A.poi(y, x), A.poi(x, x)]) - o[f"pt_{i:03d}_poi"])
    for nm in ("toy_x", "toy_y"):
        x = jnp.asarray(z[nm])
        upd("toy", np.array([A.agg_mean(x), A.agg_median(x), A.agg_iqm(x), A.agg_og(x)]) - o[f"pt_{nm}_agg"])
    upd("toy", np.array([A.poi(jnp.asarray(z["toy_x"]), jnp.asarray(z["toy_y"]))]) - o["pt_toy_poi"])
    res["points_max_abs_diff"] = worst
    res["points_pass"] = all(v <= 1e-12 for v in worst.values())
    metric("points", **worst)

    # ------------------------------------------------------------------ exact replay
    rw = {}
    for i in range(200):
        x = jnp.asarray(z[f"x{i:03d}"])
        ours = A.bootstrap_dist_from_indices((x,), (jnp.asarray(o[f"rp_{i:03d}_idx"]),), A.aggregates4)
        rw["stratified"] = max(rw.get("stratified", 0.0), float(np.abs(np.asarray(ours) - o[f"rp_{i:03d}_res"]).max()))
    for i in range(0, 200, 10):
        x = jnp.asarray(z[f"x{i:03d}"])
        ours = jax.vmap(lambda ii, tt: A.aggregates4(A.gather_runs(x, ii, tt)))(
            jnp.asarray(o[f"rptask_{i:03d}_idx"]), jnp.asarray(o[f"rptask_{i:03d}_tidx"]))
        rw["task"] = max(rw.get("task", 0.0), float(np.abs(np.asarray(ours) - o[f"rptask_{i:03d}_res"]).max()))
        y = jnp.asarray(z[f"y{i:03d}"])
        ours = A.bootstrap_dist_from_indices((x, y), (jnp.asarray(o[f"rppoi_{i:03d}_ix"]), jnp.asarray(o[f"rppoi_{i:03d}_iy"])), A.poi)
        rw["poi_independent"] = max(rw.get("poi_independent", 0.0), float(np.abs(np.asarray(ours) - o[f"rppoi_{i:03d}_res"]).max()))
    x3 = jnp.asarray(z["x003"])
    subs = jax.vmap(lambda ii: A.gather_runs(x3, ii))(jnp.asarray(o["sub_idx"]))
    rw["subsample_with_replacement"] = float(np.abs(np.asarray(subs) - o["sub_res"]).max())
    res["replay_max_abs_diff"] = rw
    res["replay_pass"] = all(v <= 1e-12 for v in rw.values())
    metric("replay", **rw)

    # ------------------------------------------------------------------ in distribution
    ks_p, ep_rows, mean_rows = {}, [], []
    sel = [int(v) for v in o["meta_sel"]] if "meta_sel" in o.files else list(range(0, 200, 8))
    for i in sel:
        x = jnp.asarray(z[f"x{i:03d}"])
        key = A.branch_key(0, A.EXP_IDS["tests"], 1, i)
        base = np.asarray(A.aggregates4(x))
        dist = np.asarray(A.bootstrap_dist(key, (x,), A.aggregates4, reps))
        jk = np.asarray(A.jackknife((x,), A.aggregates4))
        od = o[f"dist_{i:03d}_res"]
        for s in range(4):
            ks_p[f"{i:03d}_{A.AGG_NAMES[s]}"] = float(st.ks_2samp(r12(dist[:, s]), r12(od[:, s])).pvalue)
            se_m = np.sqrt(dist[:, s].var() / reps + od[:, s].var() / len(od))
            mean_rows.append(dict(input=i, stat=A.AGG_NAMES[s], ours=float(dist[:, s].mean()), oracle=float(od[:, s].mean()),
                                  z=float((dist[:, s].mean() - od[:, s].mean()) / se_m) if se_m > 0 else 0.0))
        for m in METHODS:
            ours_ci = np.asarray(A.ci_from_dist(jnp.asarray(dist), jnp.asarray(base), m, jk=jnp.asarray(jk), pooled_jk_mean=True))
            oc = o[f"dist_{i:03d}_ci_{m}"]
            for s in range(4):
                pooled = np.concatenate([dist[:, s], od[:, s]])
                if m == "bca":  # a for output s under arch's pooled jackknife mean (R9), identical on both sides
                    a_s = float(np.asarray(A.bca_acceleration(jnp.asarray(jk), pooled_mean=True))[s])
                    se = endpoint_se_bca(pooled, base[s], a_s, len(od), reps)
                else:
                    se = endpoint_se(pooled, base[s], m, len(od), reps)
                for e in range(2):
                    d = ours_ci[e, s] - oc[e, s]
                    both_nan = np.isnan(ours_ci[e, s]) and np.isnan(oc[e, s])
                    ok = both_nan or (np.isfinite(d) and abs(d) <= 4 * se[e] + 1e-12)
                    ep_rows.append(dict(input=i, shape=list(x.shape), stat=A.AGG_NAMES[s], method=m, end=("lo", "hi")[e],
                                        ours=float(ours_ci[e, s]), oracle=float(oc[e, s]), diff=float(d) if np.isfinite(d) else None,
                                        se=float(se[e]), z=float(d / se[e]) if (np.isfinite(d) and se[e] > 0) else None, ok=bool(ok)))
        info("in-dist input", i=i, shape=str(tuple(x.shape)), t=T.lap())
    hk = holm(ks_p)
    res["ks_min_holm_p"] = min(v["p_holm"] for v in hk.values())
    res["ks_min_raw_p"] = min(ks_p.values())
    res["bootstrap_mean_max_abs_z"] = max(abs(r["z"]) for r in mean_rows)
    n_fail = sum(not r["ok"] for r in ep_rows)
    res["endpoint_checks"] = len(ep_rows)
    res["endpoint_failures"] = [r for r in ep_rows if not r["ok"]]
    zs = [abs(r["z"]) for r in ep_rows if r["z"] is not None]
    res["endpoint_abs_z_quantiles"] = dict(q50=float(np.quantile(zs, .5)), q95=float(np.quantile(zs, .95)), max=float(max(zs)))
    res["endpoint_nan_pattern_mismatch"] = sum((r["ours"] != r["ours"]) != (r["oracle"] != r["oracle"]) for r in ep_rows)
    res["indist_pass"] = (n_fail == 0 and res["ks_min_holm_p"] > 1e-3 and res["bootstrap_mean_max_abs_z"] < 5)
    metric("in-dist", endpoint_checks=len(ep_rows), failures=n_fail, ks_min_holm_p=res["ks_min_holm_p"],
           mean_max_z=res["bootstrap_mean_max_abs_z"], z_q95=res["endpoint_abs_z_quantiles"]["q95"])

    # single-statistic bca (per-output jackknife; oracle 10k reps)
    rows = []
    for i in sel[:8]:
        x = jnp.asarray(z[f"x{i:03d}"])
        iqm1 = lambda s: A.agg_iqm(s)  # noqa: E731
        b, ci, dist = A.interval_estimate(A.branch_key(0, A.EXP_IDS["tests"], 2, i), (x,), iqm1, reps, method="bca")
        jk = np.asarray(A.jackknife((x,), iqm1))
        a_s = float(np.asarray(A.bca_acceleration(jnp.asarray(jk)))[0])
        B_o = 10000 if reps == 50000 else 1000
        se = endpoint_se_bca(np.asarray(dist[:, 0]), float(b[0]), a_s, B_o, reps)
        oc = o[f"bca1_{i:03d}_ci"]
        for e in range(2):
            d = float(ci[e, 0] - oc[e, 0])
            rows.append(dict(input=i, end=e, diff=d, se=float(se[e]), ok=bool(abs(d) <= 4 * se[e] + 1e-12)))
    res["bca_single"] = rows
    res["bca_single_pass"] = all(r["ok"] for r in rows)

    # task bootstrap in distribution
    rows = []
    for i in sel[:6]:
        x = jnp.asarray(z[f"x{i:03d}"])
        b, ci, dist = A.interval_estimate(A.branch_key(0, A.EXP_IDS["tests"], 3, i), (x,), A.aggregates4, reps,
                                          task_bootstrap=True)
        od, oc = o[f"task_{i:03d}_res"], o[f"task_{i:03d}_ci"]
        for s in range(4):
            se = endpoint_se(np.concatenate([np.asarray(dist[:, s]), od[:, s]]), float(b[s]), "percentile", len(od), reps)
            for e in range(2):
                d = float(ci[e, s] - oc[e, s])
                rows.append(dict(input=i, stat=A.AGG_NAMES[s], end=e, diff=d, se=float(se[e]),
                                 ks_p=float(st.ks_2samp(r12(dist[:, s]), r12(od[:, s])).pvalue), ok=bool(abs(d) <= 4 * se[e] + 1e-12)))
    res["task_bootstrap"] = dict(fails=[r for r in rows if not r["ok"]], n=len(rows),
                                 min_ks_p=min(r["ks_p"] for r in rows))
    res["task_pass"] = all(r["ok"] for r in rows)

    # P(X>Y) and profile CIs, 2,000 reps each side
    rows = []
    taus = np.linspace(-0.5, 3.0, 36)
    for i in sel[:10]:
        x, y = jnp.asarray(z[f"x{i:03d}"]), jnp.asarray(z[f"y{i:03d}"])
        b, ci, dist = A.interval_estimate(A.branch_key(0, A.EXP_IDS["tests"], 4, i), (x, y), A.poi, 2000)
        se = endpoint_se(np.asarray(dist[:, 0]), float(b[0]), "percentile", 2000, 2000)
        rows.append(dict(kind="poi", input=i, point_diff=float(b[0] - o[f"poi_{i:03d}_pe"][0]),
                         diffs=[float(ci[e, 0] - o[f"poi_{i:03d}_ci"][e, 0]) for e in range(2)], se=se.tolist(),
                         ok=bool(abs(b[0] - o[f"poi_{i:03d}_pe"][0]) <= 1e-12 and all(
                             abs(ci[e, 0] - o[f"poi_{i:03d}_ci"][e, 0]) <= 4 * se[e] + 1e-12 for e in range(2)))))
        prof = lambda s: A.score_profile(s, taus)  # noqa: E731
        b, ci, dist = A.interval_estimate(A.branch_key(0, A.EXP_IDS["tests"], 5, i), (x,), prof, 2000)
        oc = o[f"prof_{i:03d}_ci"]
        bad = 0
        for t_i in range(len(taus)):
            se = endpoint_se(np.asarray(dist[:, t_i]), float(b[t_i]), "percentile", 2000, 2000)
            for e in range(2):
                # profile values live on a 1/(N M) lattice: allow one lattice step on top of 4 SE
                if abs(float(ci[e, t_i] - oc[e, t_i])) > 4 * se[e] + 1.0 / x.size + 1e-12:
                    bad += 1
        rows.append(dict(kind="profile", input=i, point_max_diff=float(np.abs(np.asarray(b) - o[f"prof_{i:03d}_pt"]).max()),
                         endpoint_fails=bad, ok=bool(bad == 0 and np.abs(np.asarray(b) - o[f"prof_{i:03d}_pt"]).max() <= 1e-12)))
    res["poi_profile"] = rows
    res["poi_profile_pass"] = all(r["ok"] for r in rows)

    # rank matrix (Colab get_rank_matrix), n = 200,000
    n_rank = 200000 if reps == 50000 else 20000
    xs = (jnp.asarray(z["x002"]), jnp.asarray(z["y002"]), jnp.asarray(z["x006"][:, :z["x002"].shape[1]]))
    ours = np.asarray(A.rank_matrix(A.branch_key(0, A.EXP_IDS["tests"], 6, 0), xs, n_rank))
    orc = o["rank_mat"]
    pbar = (ours + orc) / 2
    se = np.sqrt(2 * pbar * (1 - pbar) / n_rank)
    zr = np.abs(ours - orc) / np.maximum(se, 1e-12)
    res["rank"] = dict(max_abs_diff=float(np.abs(ours - orc).max()), max_z=float(zr.max()), shape=list(ours.shape))
    res["rank_pass"] = bool(np.all(np.abs(ours - orc) <= 4 * se + 1e-12))

    # studentized (Colab mean_CI) exact
    import json
    dmc = json.load(open(os.path.join(REPO, "data", "e0", "dmc_reported.json")))
    sd = {}
    for steps in ("100k", "500k"):
        for alg in ("CURL", "SUNRISE", "CURL-D2RL"):
            n = 5 if alg == "SUNRISE" else 10
            m, ci = A.studentized_ci_official(dmc["means"][steps][alg], dmc["stds"][steps][alg], n)
            sd[f"{steps}_{alg}"] = float(np.abs(np.array([m, *ci]) - o[f"stud_{steps}_{alg}"]).max())
    res["studentized_max_abs_diff"] = sd
    res["studentized_pass"] = all(v <= 1e-10 for v in sd.values())  # AMENDMENT 1

    res["all_pass"] = all(res[k] for k in ("points_pass", "replay_pass", "indist_pass", "bca_single_pass", "task_pass",
                                          "poi_profile_pass", "rank_pass", "studentized_pass"))
    res["seconds"] = T.lap()
    write_json(a.out, res)
    metric("e0 difftest", **{k: res[k] for k in res if k.endswith("_pass")})



if __name__ == "__main__":
    main()
