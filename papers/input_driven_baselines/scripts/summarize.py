"""Aggregate runs/<exp>/<arm>/seed_<n>/metrics.json: mean +- SE over seeds, paired tests vs the state arm.

    python scripts/summarize.py runs/small [runs/small_maml_literal:maml_literal ...]
"""
import json
import os
import sys

import numpy as np
from scipy import stats

from rl_jax_bench.stats import boot_ci as _boot_ci  # noqa: E402  (shared seed-level statistics)


def load(root, rename=None):
    out = {}
    for arm in sorted(os.listdir(root)):
        d = os.path.join(root, arm)
        if not os.path.isdir(d):
            continue
        for sd in sorted(os.listdir(d)):
            p = os.path.join(d, sd, "metrics.json")
            if sd.startswith("seed_") and os.path.exists(p):
                with open(p) as f:
                    out.setdefault(rename or arm, {})[int(sd[5:])] = json.load(f)
    return out


def boot_ci(diff, n=10_000, seed=0):
    return tuple(_boot_ci(np.asarray(diff), n=n, seed=seed))


def main(args):
    runs = {}
    for a in args:
        root, _, name = a.partition(":")
        runs.update(load(root, name or None))
    table = {}
    for arm, seeds in runs.items():
        last5 = np.array([seeds[s]["final"]["test_mean_last5"] for s in sorted(seeds)])
        last = np.array([seeds[s]["final"]["test_mean_last"] for s in sorted(seeds)])
        h = {s: seeds[s]["history"] for s in seeds}
        vpg = np.array([np.nanmean([r["var_pg"] for r in h[s][-50:]]) for s in sorted(seeds)])
        bmse = np.array([np.mean([r["baseline_mse"] for r in h[s][-50:]]) for s in sorted(seeds)])
        table[arm] = {"seeds": sorted(seeds), "last5": last5.tolist(), "last5_mean": float(last5.mean()),
                      "last5_se": float(last5.std(ddof=1) / np.sqrt(len(last5))) if len(last5) > 1 else None,
                      "last_mean": float(last.mean()),
                      #: in-batch Var(PG) / MSE are NOT comparable across arms (shared-input batches for
                      # multi/maml remove the input-driven spread); the matched-policy probe is the variance result
                      "var_pg_last50__in_batch_NOT_comparable_across_arms": vpg.tolist(),
                      "baseline_mse_last50__in_batch_NOT_comparable_across_arms": bmse.tolist(),
                      "steady_env_steps_per_s": float(np.mean([seeds[s]["timing"]["steady_env_steps_per_s"] for s in seeds])),
                      "first_compile_s": float(np.mean([seeds[s]["timing"]["first_compile_s"] for s in seeds]))}
        print(f"{arm:14s} last5 {last5.mean():9.2f} +- {table[arm]['last5_se'] or 0:6.2f} (SE, n={len(last5)})  "
              f"per-seed {np.round(last5, 1).tolist()}  in-batch (not cross-arm comparable) baseline_mse {bmse.mean():9.1f}  VarPG {vpg.mean():.3g}")
    if "state" in table:
        for arm in table:
            if arm == "state":
                continue
            common = sorted(set(table[arm]["seeds"]) & set(table["state"]["seeds"]))
            a = np.array([runs[arm][s]["final"]["test_mean_last5"] for s in common])
            b = np.array([runs["state"][s]["final"]["test_mean_last5"] for s in common])
            d = a - b
            t = stats.ttest_rel(a, b) if len(d) > 1 else None
            table[arm]["paired_vs_state"] = {"diff_mean": float(d.mean()), "diff_per_seed": d.tolist(),
                                             "t_p": float(t.pvalue) if t else None, "boot95": boot_ci(d)}
            print(f"  {arm} - state: {d.mean():+.2f}  per-seed {np.round(d, 1).tolist()}  paired-t p={t.pvalue if t else float('nan'):.3g}  "
                  f"boot95 {np.round(boot_ci(d), 1).tolist()}")
    return table


def write_per_seed(args, exp_name):
    """runs/seed_<n>/metrics.json: per-seed raw final metrics of every arm of this experiment (merged in)."""
    runs = {}
    for a in args:
        root, _, name = a.partition(":")
        runs.update(load(root, name or None))
    seeds = sorted({s for v in runs.values() for s in v})
    for s in seeds:
        path = os.path.join("runs", f"seed_{s}", "metrics.json")
        blob = json.load(open(path)) if os.path.exists(path) else {}
        blob[exp_name] = {arm: {"final": v[s]["final"], "evals": v[s]["evals"], "git_sha": v[s]["git_sha"],
                                "timing": v[s]["timing"]} for arm, v in runs.items() if s in v}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(blob, f, indent=1)


if __name__ == "__main__":
    tab = main(sys.argv[1:])
    root0 = sys.argv[1].partition(":")[0]
    with open(os.path.join(root0, "summary.json"), "w") as f:
        json.dump(tab, f, indent=1)
    write_per_seed(sys.argv[1:], os.path.basename(os.path.normpath(root0)))
