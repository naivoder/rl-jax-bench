"""Compare our 2D-LH per-seed final lengths with the authors' per-run TSV (25 runs/cell): median, IQR, judgment."""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import OFFICIAL_HP_RUNS, judge  # noqa: E402

D = os.path.join(OFFICIAL_HP_RUNS, "lighthouse", "pairwise_imitation_results")
MAP = {"bc": "bc", "ppo": "ppo", "bc_then_ppo": "bc_then_ppo", "advisor": "advisor_ppo", "dagger": "dagger",
       "bc_tf1": "bc_teacher_forcing", "bc_ppo": "bc_and_ppo", "bc_tf1_then_ppo": "bc_teacher_forcing_then_ppo",
       "dagger_then_ppo": "dagger_then_ppo"}


def official(arm, i, j):
    rows = list(csv.DictReader(open(os.path.join(D, f"{MAP[arm]}__2_15.tsv")), delimiter="\t"))
    out = []
    for r in rows:
        vi = float(r["view_radius"])
        vj = r["expert_view_radius"]
        if vi != i:
            continue
        if arm != "ppo" and (vj == "" or float(vj) != j):
            continue
        out.append((float(r["avg_ep_length"]), float(r["train_steps"]), float(r["reached_near_optimal"])))
    return np.array(out)


def main(tag):
    rep = json.load(open(os.path.join("runs", tag, "report.json")))
    res = {}
    for key, row in rep.items():
        cell, arm = key.split("/")
        i, j = (int(x[1:]) for x in cell.split("_"))
        o = official(arm, i, j)
        if len(o) == 0:
            continue
        q1, med, q3 = np.percentile(o[:, 0], [25, 50, 75])
        ours = np.array(row["len"]["per_seed"])
        om = float(np.median(ours))
        res[key] = dict(ours_median=om, ours_mean=float(ours.mean()), ours_per_seed=ours.tolist(),
                        official_median=float(med), official_iqr=[float(q1), float(q3)], official_n=len(o),
                        official_near_opt_1_1=float(o[:, 2].mean()), ours_near_opt_1_1=row["near_opt_1.1"]["mean"],
                        judgment=judge(om, float(med), (q1, q3)))
        print(f"{key:22s} ours median {om:7.1f} | official median {med:7.1f} IQR [{q1:6.1f},{q3:6.1f}] n={len(o)} "
              f"| near-opt ours {row['near_opt_1.1']['mean']:.2f} official {o[:, 2].mean():.2f} -> {res[key]['judgment']}")
    json.dump(res, open(os.path.join("runs", tag, "official_comparison.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
