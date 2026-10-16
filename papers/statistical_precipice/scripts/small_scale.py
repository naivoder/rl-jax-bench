"""small-scale run (small scale, all arms): (A) the E0 differential test result; (B) the paper's Atari-200M aggregates on the
public Dopamine 5-run data (Fig 9 rows DQN (Nature), DQN (Adam), C51, IQN, Rainbow) with 95 % stratified-bootstrap
CIs, P(X>Y) and the IQN-vs-Rainbow profile crossing (Fig 10); (C) the E6 20-seed pool (3 arms x 2 tasks): aggregates,
P(X>Y), Welch t, and the few-run procedures at reduced counts (sampling distributions, conclusion rates, coverage).
Writes runs/small/metrics.json.

    python scripts/small_scale.py
"""
import json
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

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src import eval as E  # noqa: E402

# Fig 9 read-offs the README; read precision median/IQM +-0.03, mean +-0.15, OG +-0.005)
FIG9 = {"DQN (Nature)": dict(median=0.63, iqm=0.69, mean=2.1, og=0.425),
        "DQN (Adam)": dict(median=1.03, iqm=1.28, mean=3.0, og=0.279),
        "C51": dict(median=1.08, iqm=1.27, mean=3.05, og=0.275),
        "IQN": dict(median=1.24, iqm=1.75, mean=4.0, og=0.207),
        "Rainbow": dict(median=1.50, iqm=1.71, mean=3.7, og=0.218)}
READ_PREC = dict(median=0.03, iqm=0.03, mean=0.15, og=0.005)


def judge(ours, paper, prec):
    rel = abs(ours - paper) / abs(paper)
    if abs(ours - paper) <= prec or rel <= 0.05:
        return "MATCH", rel
    return ("CLOSE" if rel <= 0.20 else "DIVERGENT"), rel


def main():
    check_device()
    T = Timer()
    T_all = Timer()
    out = {}
    root = A.branch_key(0, A.EXP_IDS["small"])
    # ------------------------------------------------------------------ A: E0
    p = os.path.join(REPO, "runs", "e0", "difftest.json")
    e0 = json.load(open(p)) if os.path.exists(p) else {}
    out["e0"] = {k: e0.get(k) for k in ("all_pass", "points_pass", "replay_pass", "indist_pass", "bca_single_pass",
                                         "task_pass", "poi_profile_pass", "rank_pass", "studentized_pass", "reps",
                                         "points_max_abs_diff", "replay_max_abs_diff", "endpoint_checks",
                                         "endpoint_abs_z_quantiles", "ks_min_holm_p", "oracle_sha256")}
    # ------------------------------------------------------------------ B: ALE (Dopamine)
    ale, games, prov = D.load_dopamine_ale(fetch=True)
    out["ale_provenance"] = dict(commit=D.DOPAMINE_COMMIT, n_games=len(games),
                                 sha256={g: v["sha256"] for g, v in prov.items()},
                                 runs_per_agent={k: int(v.shape[0]) for k, v in ale.items()})
    tab = E.aggregate_table(jax.random.fold_in(root, 1), ale, B=50_000)
    rows = []
    for alg, r in tab.items():
        for s in A.AGG_NAMES:
            v, verdict = r[s], judge(r[s][0], FIG9[alg][s], READ_PREC[s])
            rows.append(dict(alg=alg, stat=s, ours=v[0], lo=v[1], hi=v[2], paper=FIG9[alg][s], verdict=verdict[0],
                             rel_err=verdict[1], order_of_magnitude_ok=bool(0.1 <= v[0] / FIG9[alg][s] <= 10)))
    out["ale_aggregates"] = tab
    out["ale_vs_fig9"] = rows
    # oracle points on the identical ALE arrays (rliable)
    op = os.path.join(DATA_ROOT, "e0", "oracle.npz")
    if os.path.exists(op):
        o = np.load(op)
        if "ale_names" in o.files:
            names = [str(n) for n in o["ale_names"]]
            out["ale_oracle_point_max_abs_diff"] = float(max(
                np.abs(np.asarray(A.aggregates4(jnp.asarray(ale[n]))) - o[f"ale_{j}_pt"]).max() for j, n in enumerate(names)))
            out["ale_oracle_ci_diff"] = {n: (np.array([[tab[n][s][1] for s in A.AGG_NAMES], [tab[n][s][2] for s in A.AGG_NAMES]])
                                             - o[f"ale_{j}_ci"]).tolist() for j, n in enumerate(names)}
            out["ale_oracle_poi_max_abs_diff"] = float(max(
                abs(float(A.poi(jnp.asarray(ale[a]), jnp.asarray(ale[b]))) - float(o[f"ale_poi_{j}_{k}"][0]))
                for j, a in enumerate(names) for k, b in enumerate(names) if a != b))
    out["ale_poi"] = E.poi_table(jax.random.fold_in(root, 2), ale,
                                 [("IQN", "Rainbow"), ("Rainbow", "C51"), ("C51", "DQN (Adam)"), ("DQN (Adam)", "DQN (Nature)")],
                                 B=2000)
    taus = np.linspace(0.0, 8.0, 81)
    prof = {k: np.asarray(A.score_profile(jnp.asarray(v), taus)) for k, v in ale.items()}
    d = prof["IQN"] - prof["Rainbow"]
    out["ale_profile_iqn_minus_rainbow"] = dict(taus=taus.tolist(), diff=d.tolist(),
                                                below2_mean=float(d[taus < 2].mean()), above2_mean=float(d[taus >= 2].mean()))
    info("ale done", t=T.lap())
    # few-run on ALE (pool = 5 runs): m-out-of-n sampling distribution of median vs IQM at N = 3, 5
    fr = {}
    for alg in ("IQN", "Rainbow"):
        pool = jnp.asarray(ale[alg])
        for N in (3, 5):
            for nm, fn in (("median", A.agg_median), ("iqm", A.agg_iqm)):
                v = np.asarray(A.sampling_distribution(jax.random.fold_in(root, 100 + N), pool, fn, N, 20_000))[:, 0]
                fr[f"{alg}/{nm}/N{N}"] = dict(mean=float(v.mean()), sd=float(v.std()), q025=float(np.quantile(v, .025)),
                                              q975=float(np.quantile(v, .975)), full=float(fn(pool)))
    out["ale_fewrun"] = fr
    # ------------------------------------------------------------------ C: E6 pool
    pool, seeds = D.load_e6_pool(os.path.join(REPO, "runs", "small"))
    out["e6_seeds"] = {k: len(v) for k, v in seeds.items()}
    if len(pool) == 3:
        raw, _ = D.load_e6_pool(os.path.join(REPO, "runs", "small"), normalised=False)
        out["e6_raw_mean_se"] = {a: {t: [float(raw[a][:, j].mean()), float(raw[a][:, j].std(ddof=1) / np.sqrt(len(raw[a])))]
                                     for j, t in enumerate(("CartPole-v1", "Acrobot-v1"))} for a in raw}
        rawg, _ = D.load_e6_pool(os.path.join(REPO, "runs", "small"), key="eval_greedy_mean", normalised=False)
        out["e6_raw_greedy_mean"] = {a: rawg[a].mean(0).tolist() for a in rawg}
        rawf, _ = D.load_e6_pool(os.path.join(REPO, "runs", "small"), key="final_window", normalised=False)
        out["e6_raw_final_window_mean"] = {a: rawf[a].mean(0).tolist() for a in rawf}
        out["e6_aggregates"] = E.aggregate_table(jax.random.fold_in(root, 3), pool, B=50_000)
        out["e6_poi"] = E.poi_table(jax.random.fold_in(root, 4), pool, [("separate", "shared"), ("separate", "advnorm_batch")],
                                    B=2000)
        welch = {}
        for b in ("shared", "advnorm_batch"):
            x, y = pool["separate"].mean(1), pool[b].mean(1)
            t = st.ttest_ind(x, y, equal_var=False)
            welch[f"separate vs {b}"] = dict(diff=float(x.mean() - y.mean()), t=float(t.statistic), p=float(t.pvalue))
        out["e6_welch_taskmean"] = welch
        # few-run conclusion rates: draw n of the 20 seeds per arm WITHOUT replacement, independent per arm,
        # 2,000 repetitions; IQM difference CI (stratified bootstrap, 1,000 reps) excludes 0 in the pool's direction?
        conc = {}
        for b in ("shared", "advnorm_batch"):
            X, Y = jnp.asarray(pool["separate"]), jnp.asarray(pool[b])
            full = float(A.agg_iqm(X) - A.agg_iqm(Y))
            for n in (3, 5, 10):
                @jax.jit
                def one(k):
                    sx = A.subsample(jax.random.fold_in(k, 0), X, n, replace=False)
                    sy = A.subsample(jax.random.fold_in(k, 1), Y, n, replace=False)
                    diff = A.agg_iqm(sx) - A.agg_iqm(sy)
                    bk = jax.random.split(jax.random.fold_in(k, 2), 1000)
                    dd = jax.vmap(lambda bb: (lambda r: A.agg_iqm(r[0]) - A.agg_iqm(r[1]))(A.resample_many(bb, (sx, sy))))(bk)
                    lo, hi = jnp.percentile(dd, 2.5), jnp.percentile(dd, 97.5)
                    return jnp.stack([diff, lo, hi])
                r = np.asarray(jax.lax.map(one, jax.random.split(jax.random.fold_in(root, 200 + n), 2000), batch_size=200))
                sign = np.sign(full)
                conc[f"separate-{b}/n{n}"] = dict(pool_iqm_diff=full, point_same_sign=float(np.mean(np.sign(r[:, 0]) == sign)),
                                                  ci_excludes0_same_sign=float(np.mean((r[:, 1] > 0) if sign > 0 else (r[:, 2] < 0))),
                                                  ci_excludes0_opposite=float(np.mean((r[:, 2] < 0) if sign > 0 else (r[:, 1] > 0))),
                                                  ci_covers_pool_diff=float(np.mean((r[:, 1] <= full) & (full <= r[:, 2]))),
                                                  mean_ci_width=float(np.mean(r[:, 2] - r[:, 1])))
        out["e6_conclusion_rates"] = conc
        cov = {}
        for nm, fn in (("iqm", A.agg_iqm), ("mean", A.agg_mean), ("median", A.agg_median)):
            for K in (3, 5, 10):
                res, truth, _ = A.coverage(jax.random.fold_in(root, 300 + K), jnp.asarray(pool["separate"]), fn, K,
                                           sets=1000, inner_B=1000)
                cov[f"{nm}/K{K}"] = dict(truth=truth, **{m: dict(coverage=v["coverage"], se=v["se"], width=v["mean_width"],
                                                               undefined=v["undefined"]) for m, v in res.items()})
        out["e6_coverage_pool20_separate"] = cov
        info("e6 done", t=T.lap())
    else:
        warn("E6 pool incomplete", arms=list(pool))
    # ------------------------------------------------------------------ verdict
    ale_ok = all(r["order_of_magnitude_ok"] for r in rows)
    e6_ok = len(pool) == 3 and all(len(v) == 20 for v in seeds.values())
    direction = None
    if e6_ok:
        direction = out["e6_aggregates"]["separate"]["iqm"][0] > out["e6_aggregates"]["shared"]["iqm"][0]
    out["flags"] = dict(e0=bool(out["e0"].get("all_pass")), ale_order_of_magnitude=ale_ok, e6_complete=e6_ok,
                        e6_separate_beats_shared_direction=bool(direction))
    out["PASS"] = all(out["flags"].values())
    out["seconds"] = T_all.lap()
    write_json(os.path.join(REPO, "runs", "small", "metrics.json"), out)
    metric("small", PASS=out["PASS"], **out["flags"])


if __name__ == "__main__":
    main()
