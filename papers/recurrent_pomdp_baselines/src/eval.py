"""Evaluation metrics, the authors'-log reference reader, MATCH/CLOSE/DIVERGENT judgement and paired statistics.

The evaluation rollouts themselves are ``Runner._evaluate`` in train.py (deterministic policy, Pendulum 20 episodes,
meta envs the 20 eval tasks, metric = mean total (meta-)episode return == official ``metrics/return_eval_total``).

    python -m src.eval summarize runs/small/P_sac_separate runs/small/P_sac_shared ...
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

from rl_jax_bench.stats import holm, mean_se, paired_compare  # noqa: E402,F401

AUTHOR_LOGS = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), "numeric_results", "extracted", "logs_txt")


def final_metric(steps, rets, total_steps, lo_frac=0.8):
    """mean eval metric over evaluations with env_step >= 0.8 * T (Appendix C, 'last 20 %')."""
    steps, rets = np.asarray(steps, float), np.asarray(rets, float)
    sel = steps >= lo_frac * total_steps
    return float(rets[sel].mean()) if sel.any() else float("nan")


# The paper's own final-performance pipeline (recheck), per environment as in the official scripts/eval.sh:
# merge_csv.py --start_x/--interval_x, plot_csv.py --window_size (Pendulum-V: 2000/1000/3, PointRobotSparse: 60000/15000/10).
PAPER_METRIC_ARGS = {"pendulum": dict(start_x=2000, interval_x=1000, window=3),
                     "semicircle": dict(start_x=60000, interval_x=15000, window=10)}


def paper_metric(steps, rets, start_x, interval_x, window, max_x=None, ratio=0.8):
    """Port of the authors' metric (official scripts/merge_csv.py + scripts/plot_csv.py get_run_down), one run:
    1. np.interp the eval curve onto the grid start_x, start_x + interval_x, ..., <= the run's last eval step, rounded to
       2 decimals (merge_csv); 2. keep grid points <= max_x (plot_csv --max_x, applied BEFORE smoothing);
    3. trailing rolling mean with min_periods=1 (plot_csv --window_size); 4. mean of points [int(ratio * len):]
    (get_run_down). Table 7 = mean and sd (ddof 1) of this value over seeds. It reproduces Table 7 Semi-Circle
    94.2 +- 0.6 from the authors' logs; ``final_metric`` (raw mean of evals at >= 0.8 T, our pre-declared primary
    metric) is kept unchanged beside it."""
    x, y = np.asarray(steps, float), np.asarray(rets, float)
    g = np.arange(start_x, x.max() + 1, interval_x)
    yi = np.around(np.interp(g, x, y), 2)
    if max_x is not None:
        yi = yi[g <= max_x]
    sm = np.array([yi[max(0, i - window + 1):i + 1].mean() for i in range(len(yi))])
    return float(sm[int(ratio * len(sm)):].mean())


def read_progress(path):
    rows = list(csv.DictReader(open(path)))
    pts = [(float(r["z/env_steps"]), float(r["metrics/return_eval_total"])) for r in rows
           if r.get("metrics/return_eval_total") not in (None, "")]
    return np.array(pts)


def author_runs(rel_glob):
    """per-seed eval curves of one authors'-log cell, e.g. 'pomdp/Pendulum/V/sac_lstm/len--1/*/*/*/oar/*/progress.csv'."""
    return [read_progress(f) for f in sorted(glob.glob(os.path.join(AUTHOR_LOGS, rel_glob)))]


def author_final(rel_glob, total_steps=None, window=None):
    """per-seed authors' final metric: last 20 % of each run's own max env step, or an explicit [lo, hi] window."""
    out = []
    for c in author_runs(rel_glob):
        if window is not None:
            sel = (c[:, 0] >= window[0]) & (c[:, 0] <= window[1])
            out.append(float(c[sel, 1].mean()))
        else:
            out.append(final_metric(c[:, 0], c[:, 1], total_steps or c[:, 0].max()))
    return out


def judge(ours: float, ref: float, ref_spread: float | None = None, lo: float | None = None,
          hi: float | None = None) -> str:
    """MATCH (<= 5 % RAW relative error, or inside the reference's +-1 sd spread) / CLOSE (<= 20 % raw) / DIVERGENT.

    ``lo``/``hi`` are accepted for backward compatibility and IGNORED: since the review the normalised scale
    (R - lo) / (hi - lo) is reported beside the judgement (``judge_detail``) but never decides it -- it made e.g.
    -200 vs -170 (17.6 % raw, ~10 authors' se) a MATCH on Pendulum."""
    if ref_spread is not None and abs(ours - ref) <= ref_spread:
        return "MATCH"
    rel = abs(ours - ref) / max(abs(ref), 1e-12)
    return "MATCH" if rel <= 0.05 else "CLOSE" if rel <= 0.20 else "DIVERGENT"


def judge_detail(ours: float, ref_vals, lo: float | None = None, hi: float | None = None) -> dict:
    """judgement against a per-seed reference sample plus everything a reader needs to weigh it: raw relative error,
    the gap in units of the reference's sd and se, the normalised-scale error (reported only), and whether a MATCH
    rests on the spread rule alone (raw error > 5 %)."""
    ref_vals = np.asarray(ref_vals, float)
    rm = float(ref_vals.mean())
    rsd = float(ref_vals.std(ddof=1)) if len(ref_vals) > 1 else None
    rse = rsd / np.sqrt(len(ref_vals)) if rsd is not None else None
    raw = 100.0 * (ours - rm) / max(abs(rm), 1e-12)
    v = judge(ours, rm, rsd)
    out = {"ref_mean": rm, "ref_sd": rsd, "ref_se": rse, "raw_rel_pct": raw,
           "gap_in_ref_sd": (ours - rm) / rsd if rsd else None, "gap_in_ref_se": (ours - rm) / rse if rse else None,
           "verdict": v, "match_by_spread_only": bool(v == "MATCH" and abs(raw) > 5.0)}
    if lo is not None and hi is not None:
        out["norm_rel_pct_reported_only"] = 100.0 * ((ours - lo) / (hi - lo) - (rm - lo) / (hi - lo))
    return out


def final_value(m: dict) -> float:
    """the per-seed final metric of a metrics.json: ``final_last20`` (paper horizon) or ``final_last20_of_run``
    (run shorter than the paper's horizon) -- callers that compare with the paper must check
    ``m['final_metric_key']``."""
    return m["final_last20"] if "final_last20" in m else m["final_last20_of_run"]


def load_run(run_dir):
    """{variant: {seed: metrics dict}} from runs/<...>/<variant>/seed_<s>/metrics.json."""
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "*", "seed_*", "metrics.json"))):
        m = json.load(open(f))
        out.setdefault(m["variant"], {})[m["seed"]] = m
    return out


def summarize(run_dirs):
    rows = []
    for d in run_dirs:
        for v, seeds in load_run(d).items():
            vals = [final_value(seeds[s]) for s in sorted(seeds)]
            m, se = mean_se(vals)
            rows.append({"run": os.path.basename(d.rstrip("/")), "variant": v, "n": len(vals), "mean": m, "se": se,
                         "per_seed": vals})
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "summarize":
        for r in summarize(sys.argv[2:]):
            print(json.dumps(r))
