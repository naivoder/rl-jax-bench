"""full run, E6 own-compute pool (100 seeds x 3 arms x 2 tasks, CPU-trained): the pre-declared primary family (H1, H2,
Holm), the few-run study (3 / 5 / 10 / 20 / 50 runs), Fig 6-style coverage, the E1-sub claims M1-M4, the theory checks
T1-T3 and the ablations A1-A3 the README run pre-declaration). Writes runs/full/e6.json.

    JAX_PLATFORMS=cpu python scripts/full_e6.py [--quick] [--root runs/full]

Key tree: branch_key(0, EXP_IDS['e6'], 0, STAT) with STAT ids below (logged). --quick divides every count by 10-20
(for debugging only; never reported).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, warn, write_json  # noqa: E402
from scipy import stats as st  # noqa: E402
from rl_jax_bench.stats import holm  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src import eval as E  # noqa: E402

STAT = dict(agg=1, h_diff=2, h_poi=3, h_meandiff=4, fewrun=5, coverage=6, sampdist=7, lift=8, mse=9, ablate_pool=10,
            ale_pool=11, ale_profiles=12, timing=13)
TASKS = ("CartPole-v1", "Acrobot-v1")
ARMS = ("separate", "shared", "advnorm_batch")
# recheck: these were EYEBALLED the README; the exact values are the vector read-off
# runs/full/paper_readoff.json["fig6"] (scripts/paper/vector_readoff.py), used by scripts/recheck_theory.py.
# The literal is kept unchanged (as PAPER_FIG6_EYEBALLED, alias PAPER_FIG6) so the pre-fix fits can be reproduced.
PAPER_FIG6_EYEBALLED = {  # the README (read +-1.5 pp; widths +-0.003), N = 3, 5, 10, 20, 50, 100
    "N": [3, 5, 10, 20, 50, 100],
    "median": dict(percentile=[86, 92, 96, 98, 99, 100], basic=[76, 79, 82, 87, 91, 96], bc=[80, 83, 86, 88, 92, 96],
                   bca=[84, 85, 87, 88, 91, 97], width=[.087, .078, .063, .050, .035, .023]),
    "iqm": dict(percentile=[86, 90, 93, 94, 96, 99], basic=[85, 88, 90, 95, 96, 99], bc=[86, 90, 93, 94, 96, 99],
                bca=[85, 89, 93, 94, 94, 99], width=[.068, .058, .042, .031, .022, .015])}


PAPER_FIG6 = PAPER_FIG6_EYEBALLED


def pooled_median(x):
    """E6 'median' variant (labelled): median of all N*M runs = the alpha -> 0.5 limit of the trimmed mean; the median
    of task means is the mean when M = 2 the README-sub)."""
    return jnp.median(jnp.ravel(x))


def welch_t(x, y):
    """Welch t statistic, df and two-sided p (JAX; regularised incomplete beta)."""
    nx, ny = x.shape[0], y.shape[0]
    vx, vy = jnp.var(x, ddof=1) / nx, jnp.var(y, ddof=1) / ny
    t = (jnp.mean(x) - jnp.mean(y)) / jnp.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (nx - 1) + vy ** 2 / (ny - 1))
    p = jax.scipy.special.betainc(df / 2.0, 0.5, df / (df + t ** 2))
    return t, df, jnp.where(jnp.isfinite(p), p, 1.0)


def pooled_resample(key, x):
    """ABLATION A2: un-stratified IID bootstrap -- draw N*M scores from the pooled set ignoring task labels."""
    flat = jnp.ravel(x)
    return flat[jax.random.randint(key, flat.shape, 0, flat.shape[0])].reshape(x.shape)


def fewrun(key, X, Y, n, reps, inner):
    """n runs per task per arm WITHOUT replacement from the pools, IQM-difference percentile CI (inner reps,
    independent stratified bootstrap) and Welch p on the n vs n task-mean scores. Returns (reps, 4): diff, lo, hi, p."""
    def one(k):
        sx = A.subsample(jax.random.fold_in(k, 0), X, n, replace=False)
        sy = A.subsample(jax.random.fold_in(k, 1), Y, n, replace=False)
        diff = A.agg_iqm(sx) - A.agg_iqm(sy)
        bk = jax.random.split(jax.random.fold_in(k, 2), inner)
        dd = jax.vmap(lambda bb: (lambda r: A.agg_iqm(r[0]) - A.agg_iqm(r[1]))(A.resample_many(bb, (sx, sy))))(bk)
        _, _, p = welch_t(jnp.mean(sx, 1), jnp.mean(sy, 1))
        return jnp.stack([diff, jnp.percentile(dd, 2.5), jnp.percentile(dd, 97.5), p])
    return np.asarray(jax.lax.map(jax.jit(one), jax.random.split(key, reps), batch_size=max(1, min(reps, 20_000 // max(inner, 1) * 10))))


def coverage_ablation(key, pool, stat, k, sets, inner):
    """A2: percentile coverage + width of stratified vs pooled (un-stratified) bootstrap on the SAME k-run sets."""
    pool = jnp.asarray(pool)
    truth = stat(pool)

    def one(kk):
        sub = A.subsample(jax.random.fold_in(kk, 0), pool, k, replace=False)
        bk = jax.random.split(jax.random.fold_in(kk, 1), inner)
        ds = jax.vmap(lambda b: stat(A.resample(b, sub)))(bk)
        dp = jax.vmap(lambda b: stat(pooled_resample(b, sub)))(bk)
        q = lambda d: (jnp.percentile(d, 2.5), jnp.percentile(d, 97.5))  # noqa: E731
        (a0, a1), (b0, b1) = q(ds), q(dp)
        return jnp.stack([(a0 <= truth) & (truth <= a1), a1 - a0, (b0 <= truth) & (truth <= b1), b1 - b0])
    r = np.asarray(jax.lax.map(jax.jit(one), jax.random.split(key, sets), batch_size=min(sets, 50)))
    c = lambda v: dict(coverage=float(v.mean()), se=float(math.sqrt(v.mean() * (1 - v.mean()) / len(v))))  # noqa: E731
    return dict(stratified=dict(**c(r[:, 0]), width=float(r[:, 1].mean())), pooled=dict(**c(r[:, 2]), width=float(r[:, 3].mean())))


def fit_power(N, y):
    """log y = c - beta log N (least squares); returns beta, its SE and R^2."""
    lx, ly = np.log(np.asarray(N, float)), np.log(np.asarray(y, float))
    res = st.linregress(lx, ly)
    return dict(beta=float(-res.slope), se=float(res.stderr), r2=float(res.rvalue ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "runs", "full"))
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "full", "e6.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--declared", action="store_true", help="run few-run / coverage at the originally declared counts (10,000 x 2,000)")
    ap.add_argument("--only", default="", help="comma list of blocks to run (debug)")
    args = ap.parse_args()
    check_device()
    only = set(filter(None, args.only.split(",")))
    run = lambda b: not only or b in only  # noqa: E731
    q = 20 if args.quick else 1
    src = D.sources()
    norm = src["e6"]["normalise"]["value"]
    assert {k: tuple(v) for k, v in norm.items()} == D.E6_BOUNDS, "E6_BOUNDS != sources.yaml e6.normalise"
    B_AGG = src["bootstrap"]["reps_aggregate"]["value"] // q
    B_POI = src["bootstrap"]["reps_poi"]["value"] // q
    # Full-run counts: coverage 2,000 sets x 1,000 inner reps (sources.yaml declares 10,000 x 2,000)
    SETS = 2_000 // q
    INNER = 1_000 // q
    FR_INNER = 1_000 // q
    if args.declared:
        SETS, INNER, FR_INNER = 10_000, 2_000, 2_000
    SETS_DECLARED = (src["case_study"]["coverage_sets"]["value"], src["case_study"]["coverage_inner_reps"]["value"])
    SUBREPS = src["case_study"]["subsample_reps"]["value"] // q
    MSE = src["case_study"]["subsample_reps_mse"]
    key = A.branch_key(0, A.EXP_IDS["e6"])
    K = lambda s: jax.random.fold_in(key, STAT[s])  # noqa: E731
    T = Timer()
    pool, seeds = D.load_e6_pool(args.root, envs=TASKS, arms=ARMS)
    raw, _ = D.load_e6_pool(args.root, envs=TASKS, arms=ARMS, normalised=False)
    rawg, _ = D.load_e6_pool(args.root, envs=TASKS, arms=ARMS, key="eval_greedy_mean", normalised=False)
    out = dict(root=os.path.relpath(args.root, REPO), quick=args.quick, stat_ids=STAT, exp=A.EXP_IDS["e6"], master_seed=0,
               counts=dict(B_AGG=B_AGG, B_POI=B_POI, sets=SETS, inner=INNER, subsample_reps=SUBREPS, fewrun_reps=10_000 // q, fewrun_inner=FR_INNER, declared_run=args.declared, declared_sets_inner=SETS_DECLARED, amendment="the full-run amendment"),
               seeds={a: len(v) for a, v in seeds.items()})
    if set(pool) != set(ARMS):
        warn("pool incomplete", arms=list(pool))
    out["raw_mean_se"] = {a: {t: [float(raw[a][:, j].mean()), float(raw[a][:, j].std(ddof=1) / math.sqrt(len(raw[a])))]
                              for j, t in enumerate(TASKS)} for a in raw}
    out["raw_greedy_mean_se"] = {a: {t: [float(rawg[a][:, j].mean()), float(rawg[a][:, j].std(ddof=1) / math.sqrt(len(rawg[a])))]
                                     for j, t in enumerate(TASKS)} for a in rawg}
    out["failed_runs"] = {a: {t: int((pool[a][:, j] < 0.5).sum()) for j, t in enumerate(TASKS)} for a in pool}
    out["per_seed_normalised"] = {a: pool[a].tolist() for a in pool}
    P = {a: jnp.asarray(pool[a]) for a in pool}

    # ------------------------------------------------------------------ device cross-check (GPU small-scale run vs CPU full run, seeds 1-20)
    if run("device"):
        dev = {}
        g2, s2 = D.load_e6_pool(os.path.join(REPO, "runs", "small"), envs=TASKS, arms=ARMS, normalised=False)
        for a in ARMS:
            if a not in g2 or a not in raw:
                continue
            idx = [seeds[a].index(s_) for s_ in s2[a] if s_ in seeds[a]]
            for j, t in enumerate(TASKS):
                gpu, cpu = g2[a][:, j], raw[a][idx, j]
                w = st.ttest_ind(gpu, cpu, equal_var=False)
                ks = st.ks_2samp(gpu, cpu)
                dev[f"{t}/{a}"] = dict(n=len(idx), gpu_mean=float(gpu.mean()), cpu_mean=float(cpu.mean()),
                                       welch_p=float(w.pvalue) if np.isfinite(w.pvalue) else None, ks_p=float(ks.pvalue),
                                       identical_seeds=int(np.sum(gpu == cpu)),
                                       cpu_seeds21_100_mean=float(np.delete(raw[a][:, j], idx).mean()))
        out["device_check"] = dev
        metric("device", **{k: round(v["ks_p"], 3) for k, v in dev.items()})

    # ------------------------------------------------------------------ aggregates per arm
    if run("agg"):
        out["aggregates"] = E.aggregate_table(K("agg"), pool, B_AGG)
        metric("aggregates", seconds=T.lap())

    # ------------------------------------------------------------------ primary family H1, H2
    if run("primary") and len(pool) == 3:
        fam, pv = {}, {}
        for j, (h, b) in enumerate((("H1", "shared"), ("H2", "advnorm_batch"))):
            x, y = P["separate"], P[b]
            dfun = lambda u, v: A.agg_iqm(u) - A.agg_iqm(v)  # noqa: E731
            pt, ci, _ = A.interval_estimate(jax.random.fold_in(K("h_diff"), j), (x, y), dfun, B_AGG)
            pp, pci, _ = A.interval_estimate(jax.random.fold_in(K("h_poi"), j), (x, y), A.poi, B_POI)
            mfun = lambda u, v: A.agg_mean(u) - A.agg_mean(v)  # noqa: E731
            mp, mci, _ = A.interval_estimate(jax.random.fold_in(K("h_meandiff"), j), (x, y), mfun, 10_000 // q)
            tx, ty = pool["separate"].mean(1), pool[b].mean(1)
            w = st.ttest_ind(tx, ty, equal_var=False)
            fam[h] = dict(comparison=f"separate vs {b}", iqm_diff=float(pt[0]), iqm_diff_ci=[float(ci[0, 0]), float(ci[1, 0])],
                          poi=float(pp[0]), poi_ci=[float(pci[0, 0]), float(pci[1, 0])],
                          mean_diff=float(mp[0]), mean_diff_ci=[float(mci[0, 0]), float(mci[1, 0])],
                          taskmean_sep=[float(tx.mean()), float(tx.std(ddof=1) / math.sqrt(len(tx)))],
                          taskmean_other=[float(ty.mean()), float(ty.std(ddof=1) / math.sqrt(len(ty)))],
                          welch_t=float(w.statistic), welch_p=float(w.pvalue), n=[len(tx), len(ty)])
            pv[h] = float(w.pvalue)
        hh = holm(pv)
        for h in fam:
            f = fam[h]
            f["p_holm"] = hh[h]["p_holm"]
            excl = f["iqm_diff_ci"][0] > 0 or f["iqm_diff_ci"][1] < 0
            same = np.sign(f["iqm_diff"]) == np.sign(f["welch_t"])
            f["decision"] = (f"separate {'outperforms' if f['iqm_diff'] > 0 else 'underperforms'} {f['comparison'].split(' vs ')[1]}"
                             if (f["p_holm"] < 0.05 and excl and same) else "no detectable difference at n = 100")
        out["primary"] = fam
        metric("primary", seconds=T.lap(), **{h: fam[h]["decision"] for h in fam})

    # ------------------------------------------------------------------ few-run study
    if run("fewrun") and len(pool) == 3:
        fr = {}
        for j, b in enumerate(("shared", "advnorm_batch")):
            X, Y = P["separate"], P[b]
            full = float(A.agg_iqm(X) - A.agg_iqm(Y))
            sgn = np.sign(full)
            for n in (3, 5, 10, 20, 50):
                t0 = time.time()
                r = fewrun(jax.random.fold_in(jax.random.fold_in(K("fewrun"), j), n), X, Y, n, 10_000 // q, FR_INNER)  # full-run count (declared: 2,000)
                lo_pos, hi_neg = r[:, 1] > 0, r[:, 2] < 0
                fr[f"separate-{b}/n{n}"] = dict(
                    pool_iqm_diff=full, reps=len(r),
                    point_same_sign=float(np.mean(np.sign(r[:, 0]) == sgn)),
                    ci_excludes0_same_sign=float(np.mean(lo_pos if sgn > 0 else hi_neg)),
                    ci_excludes0_wrong_sign=float(np.mean(hi_neg if sgn > 0 else lo_pos)),
                    welch_p_lt_05=float(np.mean(r[:, 3] < 0.05)),
                    welch_p_lt_05_same_sign=float(np.mean((r[:, 3] < 0.05) & (np.sign(r[:, 0]) == sgn))),
                    ci_covers_pool_diff=float(np.mean((r[:, 1] <= full) & (full <= r[:, 2]))),
                    mean_ci_width=float(np.mean(r[:, 2] - r[:, 1])), point_sd=float(np.std(r[:, 0])),
                    seconds=time.time() - t0)
                info("fewrun", cmp=b, n=n, **{k: round(v, 4) for k, v in fr[f"separate-{b}/n{n}"].items() if isinstance(v, float)})
        out["fewrun"] = fr
        metric("fewrun", seconds=T.lap())

    # ------------------------------------------------------------------ coverage (Fig 6 procedure)
    if run("coverage"):
        cov = {}
        for ai, arm in enumerate(ARMS):
            if arm not in P:
                continue
            for si, (nm, fn) in enumerate((("iqm", A.agg_iqm), ("mean", A.agg_mean))):
                for k in (3, 5, 10, 20, 50):
                    t0 = time.time()
                    res, truth, _ = A.coverage(jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(K("coverage"), ai), si), k),
                                               P[arm], fn, k, SETS, INNER)
                    cov[f"{arm}/{nm}/K{k}"] = dict(truth=truth, seconds=time.time() - t0, finite_pool_caveat=k > 25,
                                                   **{m: dict(coverage=v["coverage"], se=v["se"], width=v["mean_width"],
                                                              undefined=v["undefined"]) for m, v in res.items()})
                    info("coverage", arm=arm, stat=nm, K=k, pct=round(res["percentile"]["coverage"], 4),
                         basic=round(res["basic"]["coverage"], 4), bc=round(res["bc"]["coverage"], 4), bca=round(res["bca"]["coverage"], 4),
                         s=round(time.time() - t0, 1))
        out["coverage"] = cov
        metric("coverage", seconds=T.lap())

    # ------------------------------------------------------------------ sampling distributions (Fig 2/3/A.17), lift (Fig 4), MSE (A.16)
    if run("sampdist"):
        sd, lift, mse = {}, {}, {}
        NG = (3, 5, 10, 25, 50, 100)
        for ai, arm in enumerate(ARMS):
            if arm not in P:
                continue
            for si, (nm, fn) in enumerate((("iqm", A.agg_iqm), ("pooled_median", pooled_median), ("mean", A.agg_mean))):
                truth = float(fn(P[arm]))
                for n in NG:
                    v = np.asarray(A.sampling_distribution(jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(K("sampdist"), ai), si), n),
                                                           P[arm], fn, n, SUBREPS))[:, 0]
                    sd[f"{arm}/{nm}/N{n}"] = dict(truth=truth, mean=float(v.mean()), bias=float(v.mean() - truth),
                                                  bias_se=float(v.std() / math.sqrt(len(v))), sd=float(v.std()),
                                                  q025=float(np.quantile(v, .025)), q975=float(np.quantile(v, .975)))
                for n in NG:
                    L = np.asarray(A.lift_distribution(jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(K("lift"), ai), si), n),
                                                       P[arm], jnp.asarray([0.0, 0.10, 0.25]), fn, n, SUBREPS))
                    lift[f"{arm}/{nm}/N{n}"] = {f"{l:.2f}": [float(np.quantile(L[:, i], .025)), float(np.quantile(L[:, i], .975))]
                                                for i, l in enumerate((0.0, 0.10, 0.25))}
            alphas = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5)
            m, b_, v_ = A.trimmed_mse(jax.random.fold_in(K("mse"), ai), P[arm], alphas, MSE["n"], MSE["value"] // q)
            mse[arm] = dict(alphas=list(alphas), mse=np.asarray(m).tolist(), bias=np.asarray(b_).tolist(), var=np.asarray(v_).tolist())
        out["sampling"] = sd
        out["lift"] = lift
        out["trimmed_mse_N10"] = mse
        metric("sampdist", seconds=T.lap())

    # ------------------------------------------------------------------ ablation A2: stratification removed
    if run("ablate"):
        ab = {}
        for ai, arm in enumerate(ARMS):
            if arm not in P:
                continue
            for si, (nm, fn) in enumerate((("iqm", A.agg_iqm), ("mean", A.agg_mean))):
                for k in (3, 10):
                    ab[f"{arm}/{nm}/K{k}"] = coverage_ablation(jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(K("ablate_pool"), ai), si), k),
                                                               P[arm], fn, k, SETS // 2, INNER)  # 1,000 x 1,000 (full-run count)
        final, _, _ = D.load_dopamine_ale(final_only=True, fetch=False)
        ale = {}
        for j, (a, x) in enumerate(sorted(final.items())):
            x = jnp.asarray(x)
            for si, (nm, fn) in enumerate((("iqm", A.agg_iqm), ("median", A.agg_median), ("mean", A.agg_mean))):
                bk = jax.random.split(jax.random.fold_in(jax.random.fold_in(K("ale_pool"), j), si), B_AGG)
                ds = np.asarray(jax.lax.map(lambda b: fn(A.resample(b, x)), bk, batch_size=5000))
                dp = np.asarray(jax.lax.map(lambda b: fn(pooled_resample(b, x)), bk, batch_size=5000))
                w = lambda d: float(np.percentile(d, 97.5) - np.percentile(d, 2.5))  # noqa: E731
                ale[f"{a}/{nm}"] = dict(stratified_width=w(ds), pooled_width=w(dp), ratio=w(dp) / w(ds))
            taus = jnp.linspace(0.0, 8.0, 81)
            bk = jax.random.split(jax.random.fold_in(K("ale_profiles"), j), B_POI)
            pr = np.asarray(jax.lax.map(lambda b: A.score_profile(A.resample(b, x), taus), bk, batch_size=500))
            pa = np.asarray(jax.lax.map(lambda b: A.avg_score_profile(A.resample(b, x), taus), bk, batch_size=500))
            ale[f"{a}/profile"] = dict(run_score_mean_band=float((np.percentile(pr, 97.5, 0) - np.percentile(pr, 2.5, 0)).mean()),
                                       avg_score_mean_band=float((np.percentile(pa, 97.5, 0) - np.percentile(pa, 2.5, 0)).mean()),
                                       run_score_mean_sd=float(pr.std(0).mean()), avg_score_mean_sd=float(pa.std(0).mean()))
        out["ablation_stratification_e6"] = ab
        out["ablation_ale"] = ale
        metric("ablate", seconds=T.lap())

    # ------------------------------------------------------------------ compute cost per estimator (CPU, ALE matrix)
    if run("timing"):
        final, _, _ = D.load_dopamine_ale(final_only=True, fetch=False)
        x = jnp.asarray(final["Rainbow"])
        tm = {}
        for nm, fn in (("median", A.agg_median), ("iqm", A.agg_iqm), ("mean", A.agg_mean), ("og", A.agg_og)):
            f = jax.jit(lambda kk, fn=fn: A.bootstrap_dist(kk, (x,), fn, 10_000))
            jax.block_until_ready(f(jax.random.PRNGKey(0)))
            t0 = time.time()
            jax.block_until_ready(f(jax.random.PRNGKey(1)))
            tm[nm] = dict(seconds_per_10k_reps=time.time() - t0, shape=list(x.shape),
                          flops_note="mean/og O(NM); median O(M log M) after task means; iqm sort O(NM log NM)")
        out["estimator_timing_cpu"] = tm
        metric("timing", **{k: round(v["seconds_per_10k_reps"], 3) for k, v in tm.items()})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_json(args.out, out)
    info("done", out=args.out)


if __name__ == "__main__":
    main()
