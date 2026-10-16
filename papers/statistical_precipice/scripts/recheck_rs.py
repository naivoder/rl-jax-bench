"""recheck: the E6 primary family re-run with returns scaled for EVERY arm
(`reward_scaling=true`, the 37-details recipe K8 for shared trunks; the ppo_implementation_details reproduction). Labelled sensitivity: the
declared H1 / H2 decisions are not replaced.

  before : runs/full/<task>/<arm>/seed_<n>          (declared pool, CleanRL defaults, seeds 1-100, CPU)
  after  : runs/recheck_rs/<task>/<arm>/seed_<n>    (reward_scaling=true, SAME seeds 1-100, CPU)
  fresh  : runs/recheck_rs_fresh/<task>/<arm>/...   (reward_scaling=true, FRESH seeds 101-200, separate + shared, GPU)

Same estimators and counts as the primary family (IQM-difference CI 50,000 reps, P(X>Y) 2,000, mean difference
10,000, Welch on task-mean scores, Holm over H1/H2), fresh key branch STAT 22; plus a seed-paired before/after test
per (arm, task) and the distribution summaries that  /  are about.

    JAX_PLATFORMS=cpu python scripts/recheck_rs.py
"""
from __future__ import annotations

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
from scipy import stats as st  # noqa: E402
from rl_jax_bench.stats import holm  # noqa: E402

from src import algorithm as A  # noqa: E402
from src.data import E6_BOUNDS  # noqa: E402

TASKS = ("CartPole-v1", "Acrobot-v1")
SA_STAT = 22
B_AGG, B_POI, B_MEAN = 50_000, 2_000, 10_000


def load(root, arm, seeds, key="eval_sampled_mean"):
    raw = np.array([[json.load(open(os.path.join(REPO, "runs", root, t, arm, f"seed_{s}", "metrics.json")))["summary"][key]
                     for t in TASKS] for s in seeds])
    lo = np.array([E6_BOUNDS[t][0] for t in TASKS]); hi = np.array([E6_BOUNDS[t][1] for t in TASKS])
    return raw, (raw - lo) / (hi - lo)


def family(key, pool, pairs):
    fam, pv = {}, {}
    for j, (h, b) in enumerate(pairs):
        x, y = jnp.asarray(pool["separate"]), jnp.asarray(pool[b])
        pt, ci, _ = A.interval_estimate(jax.random.fold_in(key, 10 + j), (x, y), lambda u, v: A.agg_iqm(u) - A.agg_iqm(v), B_AGG)
        pp, pci, _ = A.interval_estimate(jax.random.fold_in(key, 20 + j), (x, y), A.poi, B_POI)
        mp, mci, _ = A.interval_estimate(jax.random.fold_in(key, 30 + j), (x, y), lambda u, v: A.agg_mean(u) - A.agg_mean(v), B_MEAN)
        tx, ty = pool["separate"].mean(1), pool[b].mean(1)
        w = st.ttest_ind(tx, ty, equal_var=False)
        fam[h] = dict(comparison=f"separate vs {b}", iqm_sep=float(A.agg_iqm(x)), iqm_other=float(A.agg_iqm(y)),
                      iqm_diff=float(pt[0]), iqm_diff_ci=[float(ci[0, 0]), float(ci[1, 0])],
                      poi=float(pp[0]), poi_ci=[float(pci[0, 0]), float(pci[1, 0])], mean_diff=float(mp[0]),
                      mean_diff_ci=[float(mci[0, 0]), float(mci[1, 0])],
                      taskmean_sep=[float(tx.mean()), float(tx.std(ddof=1) / math.sqrt(len(tx)))],
                      taskmean_other=[float(ty.mean()), float(ty.std(ddof=1) / math.sqrt(len(ty)))],
                      welch_t=float(w.statistic) if np.isfinite(w.statistic) else None,
                      welch_p=float(w.pvalue) if np.isfinite(w.pvalue) else 1.0, n=[len(tx), len(ty)])
        pv[h] = fam[h]["welch_p"]
    hh = holm(pv)
    for h, f in fam.items():
        f["p_holm"] = hh[h]["p_holm"]
        excl = f["iqm_diff_ci"][0] > 0 or f["iqm_diff_ci"][1] < 0
        same = f["welch_t"] is not None and np.sign(f["iqm_diff"]) == np.sign(f["welch_t"])
        f["decision"] = (f"separate {'outperforms' if f['iqm_diff'] > 0 else 'underperforms'} {f['comparison'].split(' vs ')[1]}"
                         if (f["p_holm"] < 0.05 and excl and same) else "no detectable difference")
    return fam


def summary(raw, rawg):
    return {t: dict(sampled_mean=float(raw[:, j].mean()), sampled_se=float(raw[:, j].std(ddof=1) / math.sqrt(len(raw))),
                    greedy_mean=float(rawg[:, j].mean()), n_at_or_below_m400=int((raw[:, j] <= -400).sum()) if t == "Acrobot-v1" else None,
                    n_at_500=int((raw[:, j] >= 499.999).sum()) if t == "CartPole-v1" else None,
                    skew=float(st.skew(raw[:, j])) if raw[:, j].std() > 0 else 0.0)
            for j, t in enumerate(TASKS)}


def main():
    key = A.branch_key(0, A.EXP_IDS["e6"], 0, SA_STAT)
    s100, s_fresh = list(range(1, 101)), list(range(101, 201))
    out = dict(stat_branch=SA_STAT, counts=dict(agg=B_AGG, poi=B_POI, mean=B_MEAN))
    arms = ("separate", "shared", "advnorm_batch")
    pools, desc = {}, {}
    for tag, root, seeds, arms_ in (("before", "full", s100, arms), ("after", "recheck_rs", s100, arms),
                                    ("fresh", "recheck_rs_fresh", s_fresh, ("separate", "shared"))):
        for ev in ("eval_sampled_mean", "eval_greedy_mean"):
            for a in arms_:
                raw, nrm = load(root, a, seeds, ev)
                pools[(tag, ev, a)] = (raw, nrm)
        desc[tag] = {a: summary(pools[(tag, "eval_sampled_mean", a)][0], pools[(tag, "eval_greedy_mean", a)][0]) for a in arms_}
    out["describe"] = desc
    fams = {}
    for i, (tag, pairs) in enumerate((("before", (("H1", "shared"), ("H2", "advnorm_batch"))),
                                      ("after", (("H1", "shared"), ("H2", "advnorm_batch"))),
                                      ("fresh", (("H1", "shared"),)))):
        for k, ev in enumerate(("eval_sampled_mean", "eval_greedy_mean")):
            pool = {a: pools[(tag, ev, a)][1] for a in ("separate",) + tuple(b for _, b in pairs)}
            fams[f"{tag}/{ev}"] = family(jax.random.fold_in(key, 100 * i + k), pool, pairs)
            print(f"[INFO] {tag:6s} {ev:17s} " + " | ".join(
                f"{h}: IQM diff {f['iqm_diff']:+.4f} [{f['iqm_diff_ci'][0]:+.4f}, {f['iqm_diff_ci'][1]:+.4f}] P {f['poi']:.3f} "
                f"Welch p {f['welch_p']:.2g} -> {f['decision']}" for h, f in fams[f"{tag}/{ev}"].items()), flush=True)
    out["families"] = fams
    paired = {}
    for a in arms:
        rb, nb = pools[("before", "eval_sampled_mean", a)]
        ra, na = pools[("after", "eval_sampled_mean", a)]
        for j, t in enumerate(TASKS + ("task-mean",)):
            xb = nb.mean(1) if t == "task-mean" else nb[:, j]
            xa = na.mean(1) if t == "task-mean" else na[:, j]
            d = xa - xb
            tt = st.ttest_rel(xa, xb)
            wil = st.wilcoxon(xa, xb) if np.any(d != 0) else None
            paired[f"{a}/{t}"] = dict(before=[float(xb.mean()), float(xb.std(ddof=1) / 10)], after=[float(xa.mean()), float(xa.std(ddof=1) / 10)],
                                      delta=float(d.mean()), delta_se=float(d.std(ddof=1) / 10), n=100,
                                      paired_t=float(tt.statistic) if np.isfinite(tt.statistic) else None,
                                      paired_p=float(tt.pvalue) if np.isfinite(tt.pvalue) else None,
                                      wilcoxon_p=float(wil.pvalue) if wil is not None else None)
    out["paired_before_after_normalised"] = paired
    path = os.path.join(REPO, "runs", "full", "recheck_rs.json")
    json.dump(out, open(path, "w"), indent=1)
    for k, v in paired.items():
        print(f"[INFO] paired {k:28s} before {v['before'][0]:.4f} after {v['after'][0]:.4f} delta {v['delta']:+.4f} +- {v['delta_se']:.4f} p {v['paired_p']}")
    print("[INFO] wrote", path)


if __name__ == "__main__":
    main()
