"""the review pass (+ review) sensitivity analyses on the full run E6 pool. Nothing here replaces a
pre-declared result; every block is a LABELLED sensitivity beside it. Writes runs/full/e6_sens.json.

    JAX_PLATFORMS=cpu python scripts/full_e6_sens.py [--only a,b]

Blocks
  trim      effective trim fraction of IQM mode 'a' (floor(K/4) per end) at K = M N, for E6 (M = 2), the paper's
            Atari-100k (M = 26) and ALE (M = 55).
  paired    common random numbers: the arms share PRNG keys per seed (common random numbers), so a paired analysis is reported
            beside the pre-declared two-sample one: Pearson r, paired t, Wilcoxon signed-rank on per-seed task-mean
            scores, and a PAIRED stratified bootstrap of the IQM difference (same run index for both arms per task
            and replicate; 50,000 reps). Sampled and greedy evaluation pools.
  greedy    H1 / H2 on the greedy-evaluation pool with the same estimators as the primary family (IQM-difference
            CI 50,000 reps, P(X > Y) 2,000 reps, Welch) and the few-run study (n = 3, 5, 10, 20) on it.
  repl      few-run study with per-task subsampling WITH replacement (the paper's Fig 2 / 3 / 4 subsampling),
            n = 3, 5, 10, 20, 50, 10,000 studies x 1,000 inner reps (full-run counts).
  iqm_c     IQM mode 'c' (exact 25 % fractional trim) in place of mode 'a' for every N-dependent IQM result whose
            K = 2N is not a multiple of 4 (N = 3, 5, 25): few-run n = 3, 5; coverage K = 3, 5; sampling distribution
            / bias N = 3, 5, 25; lift N = 3, 5, 25. SAME keys as runs/full/e6.json (full-run counts), so the
            difference isolates the estimator.

Key tree: branch_key(0, EXP_IDS['e6']) -> fold_in(STAT id) exactly as full_e6.py for the iqm_c block; new blocks
use STAT ids 20+ (logged).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, write_json  # noqa: E402
from scipy import stats as st  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
import full_e6 as G  # noqa: E402  (STAT ids, welch_t, pooled_median; module import runs no analysis)

STAT_NEW = dict(paired=20, greedy_diff=21, greedy_poi=22, greedy_fewrun=23, repl_fewrun=24)
iqm_c = lambda x: A.agg_iqm(x, "c")  # noqa: E731


def fewrun_gen(key, X, Y, n, reps, inner, stat, replace):
    """full_e6.fewrun with the IQM function and the subsampling mode as parameters (same key use)."""
    def one(k):
        sx = A.subsample(jax.random.fold_in(k, 0), X, n, replace=replace)
        sy = A.subsample(jax.random.fold_in(k, 1), Y, n, replace=replace)
        diff = stat(sx) - stat(sy)
        bk = jax.random.split(jax.random.fold_in(k, 2), inner)
        dd = jax.vmap(lambda bb: (lambda r: stat(r[0]) - stat(r[1]))(A.resample_many(bb, (sx, sy))))(bk)
        _, _, p = G.welch_t(jnp.mean(sx, 1), jnp.mean(sy, 1))
        return jnp.stack([diff, jnp.percentile(dd, 2.5), jnp.percentile(dd, 97.5), p])
    return np.asarray(jax.lax.map(jax.jit(one), jax.random.split(key, reps), batch_size=max(1, min(reps, 20_000 // max(inner, 1) * 10))))


def summarise_fewrun(r, full):
    sgn = np.sign(full)
    lo_pos, hi_neg = r[:, 1] > 0, r[:, 2] < 0
    se = lambda p: math.sqrt(p * (1 - p) / len(r))  # noqa: E731
    out = dict(pool_iqm_diff=full, reps=len(r),
               point_same_sign=float(np.mean(np.sign(r[:, 0]) == sgn)),
               ci_excludes0_same_sign=float(np.mean(lo_pos if sgn > 0 else hi_neg)),
               ci_excludes0_wrong_sign=float(np.mean(hi_neg if sgn > 0 else lo_pos)),
               welch_p_lt_05=float(np.mean(r[:, 3] < 0.05)),
               ci_covers_pool_diff=float(np.mean((r[:, 1] <= full) & (full <= r[:, 2]))),
               mean_ci_width=float(np.mean(r[:, 2] - r[:, 1])), point_sd=float(np.std(r[:, 0])))
    out["mc_se"] = {k: se(out[k]) for k in ("point_same_sign", "ci_excludes0_same_sign", "ci_excludes0_wrong_sign",
                                             "welch_p_lt_05", "ci_covers_pool_diff")}
    return out


def paired_iqm_ci(key, X, Y, B):
    """Paired stratified bootstrap: per replicate one (N, M) run-index draw applied to BOTH arms (seed pairing)."""
    def one(k):
        idx = A.stratified_indices(k, X.shape)
        return A.agg_iqm(A.gather_runs(X, idx)) - A.agg_iqm(A.gather_runs(Y, idx))
    d = np.asarray(jax.lax.map(one, jax.random.split(key, B), batch_size=5000))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "full", "e6_sens.json"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    check_device()
    only = set(filter(None, args.only.split(",")))
    run = lambda b: not only or b in only  # noqa: E731
    root = os.path.join(REPO, "runs", "full")
    pool, seeds = D.load_e6_pool(root, envs=G.TASKS, arms=G.ARMS)
    gpool, gseeds = D.load_e6_pool(root, envs=G.TASKS, arms=G.ARMS, key="eval_greedy_mean")
    assert all(seeds[a] == list(range(1, 101)) and gseeds[a] == seeds[a] for a in G.ARMS), "pools must be seeds 1..100, aligned"
    P = {a: jnp.asarray(pool[a]) for a in G.ARMS}
    PG = {a: jnp.asarray(gpool[a]) for a in G.ARMS}
    key = A.branch_key(0, A.EXP_IDS["e6"])
    K = lambda s: jax.random.fold_in(key, G.STAT[s] if s in G.STAT else STAT_NEW[s])  # noqa: E731
    T = Timer()
    out = dict(stat_ids=dict(G.STAT, **STAT_NEW), master_seed=0, exp=A.EXP_IDS["e6"],
               note="labelled sensitivities (the review pass); none replaces a pre-declared result")
    CMP = (("H1", "shared"), ("H2", "advnorm_batch"))

    if run("trim"):
        tr = {}
        for lab, M, Ns in (("E6 M=2", 2, (3, 5, 10, 20, 25, 50, 100)), ("paper Atari-100k M=26", 26, (3, 5, 10, 20, 50, 100)),
                           ("ALE M=55", 55, (3, 5))):
            tr[lab] = {f"N{N}": dict(K=M * N, cut_each_end=(M * N) // 4, kept=M * N - 2 * ((M * N) // 4),
                                     trim_fraction=((M * N) // 4) / (M * N)) for N in Ns}
        out["iqm_mode_a_effective_trim"] = tr
        metric("trim", **{f"E6_N{N}": round(v["trim_fraction"], 4) for N, v in ((k[1:], v) for k, v in tr["E6 M=2"].items())})

    if run("paired"):
        pr = {}
        for evname, PP, pp in (("sampled", P, pool), ("greedy", PG, gpool)):
            for j, (h, b) in enumerate(CMP):
                tx, ty = pp["separate"].mean(1), pp[b].mean(1)
                pt = st.ttest_rel(tx, ty)
                wi = st.wilcoxon(tx, ty)
                r = st.pearsonr(tx, ty)
                ci = paired_iqm_ci(jax.random.fold_in(jax.random.fold_in(K("paired"), j), 0 if evname == "sampled" else 1),
                                   PP["separate"], PP[b], 50_000)
                per_task_r = [float(st.pearsonr(pp["separate"][:, m], pp[b][:, m])[0]) for m in range(len(G.TASKS))]
                pr[f"{evname}/{h}"] = dict(comparison=f"separate vs {b}", pearson_r_taskmean=float(r[0]), pearson_p=float(r[1]),
                                           pearson_r_per_task=per_task_r,
                                           paired_t=float(pt.statistic), paired_p=float(pt.pvalue), wilcoxon_p=float(wi.pvalue),
                                           welch_p=float(st.ttest_ind(tx, ty, equal_var=False).pvalue),
                                           iqm_diff=float(A.agg_iqm(PP["separate"]) - A.agg_iqm(PP[b])), paired_iqm_diff_ci=ci)
                info("paired", ev=evname, h=h, r=round(float(r[0]), 3), paired_p=float(pt.pvalue), wilcoxon_p=float(wi.pvalue), ci=ci)
        out["paired"] = pr
        metric("paired", seconds=T.lap())

    if run("greedy"):
        gr = {}
        for j, (h, b) in enumerate(CMP):
            x, y = PG["separate"], PG[b]
            dfun = lambda u, v: A.agg_iqm(u) - A.agg_iqm(v)  # noqa: E731
            pt, ci, _ = A.interval_estimate(jax.random.fold_in(K("greedy_diff"), j), (x, y), dfun, 50_000)
            pp_, pci, _ = A.interval_estimate(jax.random.fold_in(K("greedy_poi"), j), (x, y), A.poi, 2_000)
            tx, ty = gpool["separate"].mean(1), gpool[b].mean(1)
            w = st.ttest_ind(tx, ty, equal_var=False)
            fr = {}
            for n in (3, 5, 10, 20):
                r = fewrun_gen(jax.random.fold_in(jax.random.fold_in(K("greedy_fewrun"), j), n), x, y, n, 10_000, 1_000, A.agg_iqm, False)
                fr[f"n{n}"] = summarise_fewrun(r, float(pt[0]))
                info("greedy fewrun", h=h, n=n, excl_same=fr[f"n{n}"]["ci_excludes0_same_sign"], excl_wrong=fr[f"n{n}"]["ci_excludes0_wrong_sign"])
            gr[h] = dict(comparison=f"separate vs {b}", iqm_by_arm=[float(A.agg_iqm(x)), float(A.agg_iqm(y))],
                         iqm_diff=float(pt[0]), iqm_diff_ci=[float(ci[0, 0]), float(ci[1, 0])],
                         poi=float(pp_[0]), poi_ci=[float(pci[0, 0]), float(pci[1, 0])],
                         welch_t=float(w.statistic), welch_p=float(w.pvalue), fewrun=fr)
        out["greedy"] = gr
        metric("greedy", seconds=T.lap())

    if run("repl"):
        rp = {}
        for j, (h, b) in enumerate(CMP):
            X, Y = P["separate"], P[b]
            full = float(A.agg_iqm(X) - A.agg_iqm(Y))
            for n in (3, 5, 10, 20, 50):
                t0 = time.time()
                r = fewrun_gen(jax.random.fold_in(jax.random.fold_in(K("repl_fewrun"), j), n), X, Y, n, 10_000, 1_000, A.agg_iqm, True)
                rp[f"separate-{b}/n{n}"] = dict(summarise_fewrun(r, full), seconds=time.time() - t0)
                info("repl fewrun", cmp=b, n=n, **{k: round(v, 4) for k, v in rp[f"separate-{b}/n{n}"].items() if isinstance(v, float)})
        out["fewrun_with_replacement"] = rp
        metric("repl", seconds=T.lap())

    if run("iqm_c"):
        ic = dict(fewrun={}, coverage={}, sampling={}, lift={})
        for j, (h, b) in enumerate(CMP):
            X, Y = P["separate"], P[b]
            full_a = float(A.agg_iqm(X) - A.agg_iqm(Y))
            full_c = float(iqm_c(X) - iqm_c(Y))
            for n in (3, 5):   # same key as full_e6.py
                r = fewrun_gen(jax.random.fold_in(jax.random.fold_in(K("fewrun"), j), n), X, Y, n, 10_000, 1_000, iqm_c, False)
                ic["fewrun"][f"separate-{b}/n{n}"] = dict(summarise_fewrun(r, full_c), pool_iqm_diff_mode_a=full_a)
                info("iqm_c fewrun", cmp=b, n=n, excl_same=ic["fewrun"][f"separate-{b}/n{n}"]["ci_excludes0_same_sign"],
                     excl_wrong=ic["fewrun"][f"separate-{b}/n{n}"]["ci_excludes0_wrong_sign"])
        for ai, arm in enumerate(G.ARMS):
            for k in (3, 5):   # same key as full_e6.py coverage block (si = 0 is the IQM)
                res, truth, _ = A.coverage(jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(K("coverage"), ai), 0), k),
                                           P[arm], iqm_c, k, 2_000, 1_000)
                ic["coverage"][f"{arm}/iqm_c/K{k}"] = dict(truth=truth, **{m: dict(coverage=v["coverage"], se=v["se"], width=v["mean_width"],
                                                                                    undefined=v["undefined"]) for m, v in res.items()})
            truth = float(iqm_c(P[arm]))
            for n in (3, 5, 25):   # same key as full_e6.py sampdist block (si = 0 is the IQM)
                v = np.asarray(A.sampling_distribution(jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(K("sampdist"), ai), 0), n),
                                                       P[arm], iqm_c, n, 100_000))[:, 0]
                ic["sampling"][f"{arm}/iqm_c/N{n}"] = dict(truth=truth, mean=float(v.mean()), bias=float(v.mean() - truth),
                                                           bias_se=float(v.std() / math.sqrt(len(v))), sd=float(v.std()),
                                                           q025=float(np.quantile(v, .025)), q975=float(np.quantile(v, .975)))
                L = np.asarray(A.lift_distribution(jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(K("lift"), ai), 0), n),
                                                   P[arm], jnp.asarray([0.0, 0.10, 0.25]), iqm_c, n, 100_000))
                ic["lift"][f"{arm}/iqm_c/N{n}"] = {f"{l:.2f}": [float(np.quantile(L[:, i], .025)), float(np.quantile(L[:, i], .975))]
                                                   for i, l in enumerate((0.0, 0.10, 0.25))}
            info("iqm_c arm", arm=arm, **{f"cov_pct_K{k}": round(ic["coverage"][f"{arm}/iqm_c/K{k}"]["percentile"]["coverage"], 4) for k in (3, 5)},
                 **{f"bias_N{n}": round(ic["sampling"][f"{arm}/iqm_c/N{n}"]["bias"], 5) for n in (3, 5, 25)})
        out["iqm_mode_c"] = ic
        metric("iqm_c", seconds=T.lap())

    write_json(args.out, out)
    info("done", out=args.out)


if __name__ == "__main__":
    main()
