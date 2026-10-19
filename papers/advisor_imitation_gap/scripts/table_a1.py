"""2D-LH Table A.1 analogue under three definitions, ours vs the authors' TSV, i = j in {1, 3, 5}:
  near_1.1  : final test avg length < 1.1 x optimal   (the official TSV column reached_near_optimal,
              lighthouse_scripts/save_pairwise_imitation_data.py: test ep_length < optimal * 1.1)
  near_1.05 : final test avg length < 1.05 x optimal  (the paper caption's 'within 5 % of optimal' on the test metric)
  es_stop   : training stopped early, i.e. the official StopIfNearOptimal criterion (TRAINING-episode mean length
              < 1.05 x optimal, es_tol 0.05) fired before 300k steps: 'converged ... with a budget of 300,000 steps'
Writes runs/report/table_a1.json."""
import csv
import json
import os

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import OFFICIAL_HP_RUNS  # noqa: E402

LH_TSV = os.path.join(OFFICIAL_HP_RUNS, "lighthouse", "pairwise_imitation_results")
MAP = {"advisor": "advisor_ppo", "bc": "bc", "dagger": "dagger"}
PAPER = {"advisor": [1, 1, 1], "bc": [1, 0.72, 0.52], "dagger": [0.88, 0.56, 0.24]}   # Table A.1, i = 1, 3, 5


def tsv(arm, i):
    rows = [r for r in csv.DictReader(open(os.path.join(LH_TSV, f"{MAP[arm]}__2_15.tsv")), delimiter="\t")
            if float(r["view_radius"]) == i and r["expert_view_radius"] != "" and float(r["expert_view_radius"]) == i]
    L = np.array([float(r["avg_ep_length"]) for r in rows]); opt = np.array([float(r["optimal"]) for r in rows])
    ts = np.array([float(r["train_steps"]) for r in rows]); rno = np.array([float(r["reached_near_optimal"]) for r in rows])
    assert np.array_equal(rno, (L < 1.1 * opt).astype(float))
    return dict(n=len(rows), near_1_1=float(np.mean(L < 1.1 * opt)), near_1_05=float(np.mean(L < 1.05 * opt)),
                es_stop=float(np.mean(ts < 300000)), opt=float(opt[0]))


def ours(tag, arm, i):
    S = json.load(open(os.path.join("runs", tag, "summary.json")))["runs"]
    rows = [r for r in S if r["cell"] == f"i{i}_j{i}" and r["arm"] == arm and (arm != "advisor" or r["alpha"] == 20)]
    L = np.array([r["final"]["len"] for r in rows]); opt = rows[0]["opt_len"]
    ts = np.array([r["train_steps"] for r in rows])
    return dict(n=len(rows), near_1_1=float(np.mean(L < 1.1 * opt)), near_1_05=float(np.mean(L < 1.05 * opt)),
                es_stop=float(np.mean(ts < 300000)), opt=float(opt))


out = {}
for arm in ("advisor", "bc", "dagger"):
    tag = "full_lh_dagger_og" if arm == "dagger" else "full_lh_og"
    for k, i in enumerate((1, 3, 5)):
        o, t = ours(tag, arm, i), tsv(arm, i)
        out[f"{arm}/i{i}"] = dict(paper=PAPER[arm][k], ours=o, tsv=t)
        print(f"{arm:8s} i={i} paper {PAPER[arm][k]:.2f} | TSV near1.1 {t['near_1_1']:.2f} near1.05 {t['near_1_05']:.2f} "
              f"es_stop {t['es_stop']:.2f} (n {t['n']}) | ours near1.1 {o['near_1_1']:.2f} near1.05 {o['near_1_05']:.2f} "
              f"es_stop {o['es_stop']:.2f} (n {o['n']})")
os.makedirs("runs/report", exist_ok=True)
json.dump(out, open("runs/report/table_a1.json", "w"), indent=1)
