"""full-scale, official-draws protocol (App. A.7.1 / Table A.5 / Fig. 5e): for every arm run at the authors' hp draws
(scripts/run_grid.py --draws), compute RobustReward@K over our draws (unbiased U-statistic, src/eval.py) and the same
statistic over the authors' runs AT THE SAME DRAWS (their per-run TSV), plus the per-draw paired difference
ours - theirs (paired t + bootstrap CI; the draw is the unit: same lr / alpha / stage-split, same seed index).

Usage: rr_report.py <tag> <tsv_task> [metric=ret] ; writes runs/<tag>/rr_report.json and prints a table."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import official_tsv, robust_reward_at_k  # noqa: E402
from rl_jax_bench.stats import paired_compare  # noqa: E402

TSV_NAME = {"advisor": "advisor", "ppo": "ppo", "bc": "bc", "dagger": "dagger", "bc_tf1": "bc_teacher_forcing",
            "bc_ppo": "bc_with_ppo", "bc_then_ppo": "bc_then_ppo", "dagger_then_ppo": "dagger_then_ppo",
            "bc_tf1_then_ppo": "bc_teacher_forcing_then_ppo", "dagger_then_advisor": "dagger_then_advisor",
            "bc_tf1_then_advisor": "bc_teacher_forcing_then_advisor"}
# Table A.5 RobustReward@10 (50 draws), exact entries
TABLE_A5 = {"PoisonedDoors": {"ppo": 0, "bc": -0.60, "dagger": -0.59, "bc_tf1": -0.62, "bc_ppo": -0.59,
                              "bc_then_ppo": -0.17, "dagger_then_ppo": -0.45, "bc_tf1_then_ppo": -0.50,
                              "advisor": 1.0, "bc_tf1_then_advisor": -0.13, "dagger_then_advisor": -0.10},
            "AskForHelpLavaCrossingOnce": {"ppo": 0, "bc": 0, "dagger": 0, "bc_tf1": 0, "bc_ppo": 0,
                                           "bc_then_ppo": 0, "dagger_then_ppo": 0, "bc_tf1_then_ppo": 0,
                                           "advisor": 0.80, "bc_tf1_then_advisor": 0, "dagger_then_advisor": 0}}


def main(tag, task, metric="ret"):
    S = json.load(open(os.path.join("runs", tag, "summary.json")))["runs"]
    S = [r for r in S if r.get("draw") is not None]
    tsv = official_tsv(task)
    theirs = {}
    for r in tsv:
        theirs[(r["exp_type"], int(r["seed"]))] = float(r["reward"][-1])
    out = {}
    for arm in TSV_NAME:
        rows = sorted([r for r in S if r["arm"] == arm], key=lambda r: r["draw"])
        if not rows:
            continue
        draws = [r["draw"] for r in rows]
        ours = np.array([r["final"][metric] for r in rows])
        th = np.array([theirs[(TSV_NAME[arm], d)] for d in draws])
        th_all = np.array([v for (m, d), v in theirs.items() if m == TSV_NAME[arm]])
        n = len(ours)
        rr = {K: dict(ours=robust_reward_at_k(ours, K), theirs_same_draws=robust_reward_at_k(th, K),
                      theirs_all50=robust_reward_at_k(th_all, K)) for K in range(1, n + 1)}
        try:
            with np.errstate(all="ignore"):
                pc = paired_compare(ours, th)
        except ValueError:      # e.g. Wilcoxon on all-zero differences
            dd = ours - th
            pc = dict(diff_mean=float(dd.mean()), diff_se=float(dd.std(ddof=1) / np.sqrt(n)), t_p=float("nan"))
        out[arm] = dict(draws=draws, ours=ours.round(4).tolist(), theirs=th.round(4).tolist(), n=n,
                        mean_ours=float(ours.mean()), se_ours=float(ours.std(ddof=1) / np.sqrt(n)),
                        mean_theirs=float(th.mean()), paired_ours_minus_theirs=pc,
                        rr=rr, table_a5_rr10=TABLE_A5[task][arm])
        K = min(10, n)
        print(f"{arm:20s} n={n:2d} RR@{K} ours {rr[K]['ours']:+.3f} | theirs same draws {rr[K]['theirs_same_draws']:+.3f}"
              f" | theirs all 50 {rr[K]['theirs_all50']:+.3f} | Table A.5 {TABLE_A5[task][arm]:+.2f} || per-draw mean "
              f"ours {ours.mean():+.3f} theirs {th.mean():+.3f} diff {pc['diff_mean']:+.3f}+-{pc['diff_se']:.3f} "
              f"p={pc['t_p']:.2g}")
    json.dump(out, open(os.path.join("runs", tag, "rr_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4]))
