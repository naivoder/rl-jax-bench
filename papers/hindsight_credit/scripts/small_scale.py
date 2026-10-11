"""Small-scale run of every experiment (E1-E8), all arms, every PG lr, 20 runs (5x fewer than the
paper's 100) on a separate seed (2; evaluation uses seed 0, PG tuning seed 999).

    uv run python -m scripts.small_scale [--n_runs 20] [--seed 2]

Writes runs/small/seed_<seed>/metrics.json (per-run raw curves) and runs/small/summary.json (means ± SE next
to the paper's read-off numbers). Small-scale criteria (declared here, before the run):
  (i)   every arm finite, every curve starts at the analytic uniform-policy value;
  (ii)  sanity: every paper headline number and our mean are within an order of magnitude of each other
        on the normalised scale (|ours - paper| <= R, R = optimum - start of that task; for E2, R = oracle);
  (iii) direction is visible: the sign of (State HCA - PG) and (Return HCA - PG) at the headline point is
        reported against the paper's sign (informative; not a pass criterion at n = 20).
"""
from __future__ import annotations

import argparse
import os
import sys

import src  # noqa: F401
import jax
import numpy as np

from src.train import load_config, provenance, run_fig3r, run_learning_exp
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, write_json

from rl_jax_bench.stats import mean_se  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# paper read-off numbers; 'final' = last curve point, 'avg' = curve mean (A-AVG)
PAPER = {
    "E1": {"stat": "final", "point": 0, "state_hca": -0.10, "pg": -0.18, "return_hca": -0.43, "start": -0.9631, "opt": -0.1},
    "E3": {"stat": "final", "point": 0, "state_hca": 0.68, "pg": 0.05, "return_hca": 0.83, "start": 0.0, "opt": 1.0},
    "E4": {"stat": "final", "point": 0, "state_hca": 0.84, "pg": 0.80, "return_hca": 0.80, "start": 0.0, "opt": 1.0},
    "E5": {"stat": "avg", "points": {0: (0.79, 0.84, 0.80), 1: (0.79, 0.83, 0.80), 2: (0.78, 0.67, 0.72),
                                     3: (0.77, 0.56, 0.56), 4: (0.71, 0.47, 0.54), 5: (0.66, 0.25, 0.47)},
           "start": 0.0, "opt": 1.0},
    "E6": {"stat": "final", "point": 0, "state_hca": 1.845, "pg": 1.73, "return_hca": 1.77, "start": 1.5, "opt": 1.9},
    "E7": {"stat": "final", "point": 0, "state_hca": 1.495, "pg": 1.715, "return_hca": 1.765, "start": 1.5, "opt": 1.9},
    "E8": {"stat": "avg", "points": {0: (1.58, 1.85, 1.97), 1: (1.56, 1.76, 1.81), 2: (1.535, 1.66, 1.68),
                                     3: (1.50, 1.58, 1.59), 4: (1.515, 1.52, 1.52)},
           "start": 1.5, "opt": None},
}
E2_PAPER = {"oracle": [1.07, 1.24, 1.45, 1.72, 2.04, 2.46, 2.96], "state_hca": [0.45, 0.62, 0.84, 1.05, 1.56, 1.96, 2.75],
            "return_hca": [0.21, 0.41, 0.65, 0.89, 1.42, 1.90, 2.78], "pg": [0.17, 0.42, 0.69, 0.83, 0.98, 0.82, 0.55]}
E8_EPS = [0.0, 0.1, 0.2, 0.3, 0.4]


def stat_of(curve, stat):
    return curve[:, -1] if stat == "final" else curve.mean(1)


def summarise_learning(exp, res):
    pp = PAPER[exp]
    rows = []
    for si, rec in enumerate(res["settings"]):
        ms = rec["methods"]
        pg_names = [k for k in ms if k.startswith("pg_lr")]
        best_pg = max(pg_names, key=lambda k: ms[k]["curve"].mean())
        start_ok = all(np.allclose(ms[k]["curve"][:, 0], ms[pg_names[0]]["curve"][0, 0]) for k in ms)
        finite = all(np.all(np.isfinite(ms[k]["curve"])) for k in ms)
        if "points" in pp:
            if si not in pp["points"]:
                continue
            tgt = dict(zip(("state_hca", "pg", "return_hca"), pp["points"][si]))
        else:
            tgt = {k: pp[k] for k in ("state_hca", "pg", "return_hca")}
        opt = pp["opt"] if pp["opt"] is not None else 2 - E8_EPS[si]
        R = opt - pp["start"]
        row = {"setting": rec["env_kwargs"], "stat": pp["stat"], "best_pg_lr_small": best_pg, "start_ok": bool(start_ok),
               "finite": bool(finite), "R": R, "arms": {}}
        for arm, key in (("state_hca", "state_hca"), ("return_hca", "return_hca"), ("pg", best_pg)):
            x = stat_of(ms[key]["curve"], pp["stat"])
            m, se = mean_se(x)
            row["arms"][arm] = {"mean": m, "se": se, "paper": tgt[arm], "norm_err": abs(m - tgt[arm]) / abs(R),
                                "within_order_of_magnitude": abs(m - tgt[arm]) <= abs(R)}
        for k in pg_names:
            row["arms"][k] = {"mean": float(stat_of(ms[k]["curve"], pp["stat"]).mean())}
        for arm in ("state_hca", "return_hca"):
            ours = row["arms"][arm]["mean"] - row["arms"]["pg"]["mean"]
            paper = tgt[arm] - tgt["pg"]
            row[f"{arm}_minus_pg"] = {"ours": ours, "paper": paper, "same_sign": bool(np.sign(ours) == np.sign(paper))}
        rows.append(row)
        metric(f"small {exp}", setting=si, **{f"{a}": round(row['arms'][a]['mean'], 4) for a in ("state_hca", "return_hca", "pg")},
               best_pg=best_pg)
    return rows


def summarise_fig3r(res):
    rows = []
    for i, pt in enumerate(res["points"]):
        R = pt["oracle"]
        row = {"p_long": pt["p_long"], "oracle": pt["oracle"], "oracle_paper": E2_PAPER["oracle"][i], "arms": {}}
        for m in ("state_hca", "return_hca", "pg"):
            mm, se = mean_se(pt["methods"][m]["online"])
            mf, _ = mean_se(pt["methods"][m]["final"])
            row["arms"][m] = {"mean": mm, "se": se, "final_h_mean": mf, "paper": E2_PAPER[m][i],
                              "norm_err": abs(mm - E2_PAPER[m][i]) / R,
                              "within_order_of_magnitude": abs(mm - E2_PAPER[m][i]) <= R}
        rows.append(row)
        metric("small E2", p_long=round(pt["p_long"], 3), oracle=pt["oracle"],
               **{m: round(row["arms"][m]["mean"], 3) for m in row["arms"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    cfg = load_config()
    check_device()
    t = Timer()
    raw, summ = {"provenance": provenance(cfg), "config": cfg, "experiments": {}}, {}
    for exp in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"):
        info("small experiment", exp=exp)
        if exp == "E2":
            res = run_fig3r(cfg, args.n_runs, args.seed)
            summ[exp] = summarise_fig3r(res)
        else:
            res = run_learning_exp(cfg, exp, args.n_runs, args.seed)
            summ[exp] = summarise_learning(exp, res)
        raw["experiments"][exp] = res
    wall = t.lap()
    raw["wall_s"] = wall
    out_dir = os.path.join(ROOT, "runs", "small", f"seed_{args.seed}")
    write_json(os.path.join(out_dir, "metrics.json"), raw)
    checks = {"finite_and_start": all(r.get("finite", True) and r.get("start_ok", True) for e in summ.values() for r in e),
              "order_of_magnitude": all(a.get("within_order_of_magnitude", True) for e in summ.values() for r in e
                                        for a in r["arms"].values())}
    summary = {"n_runs": args.n_runs, "seed": args.seed, "wall_s": wall, "checks": checks,
               "pass": all(checks.values()), "experiments": summ}
    write_json(os.path.join(ROOT, "runs", "small", "summary.json"), summary)
    info("small", passed=summary["pass"], wall_s=wall, **checks)


if __name__ == "__main__":
    main()
