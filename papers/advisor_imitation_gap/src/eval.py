"""Evaluation statistics: RobustReward@K (paper Eq. 4, App. A.9), per-seed summaries,
and the authors' per-run TSV reader used as the exact comparator."""
from __future__ import annotations

import json
import math
import os
from typing import Sequence

import numpy as np

from src.common import OFFICIAL_ROOT

OFFICIAL_HP_RUNS = os.path.join(OFFICIAL_ROOT, "advisor", "hp_runs")  # the authors' per-run TSVs


def robust_reward_at_k(values: Sequence[float], k: int) -> float:
    """Unbiased U-statistic for E[max of K iid draws]: sum_{i>=K} C(i-1, K-1)/C(n, K) v_(i) (sorted ascending)."""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= K <= n, got K={k}, n={n}")
    w = np.array([math.comb(i - 1, k - 1) for i in range(1, n + 1)], dtype=float) / math.comb(n, k)
    return float(np.dot(w, v))


def official_tsv(task: str):
    """Rows of the authors' random_hp_search_runs_<task>.tsv as a list of dicts (strings converted when numeric)."""
    import csv
    p = os.path.join(OFFICIAL_HP_RUNS, f"random_hp_search_runs_{task}.tsv")
    rows = []
    with open(p) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out = {}
            for k, x in r.items():
                try:
                    out[k] = float(x)
                except (TypeError, ValueError):
                    try:
                        out[k] = json.loads(x) if x and x[0] in "[{" else x
                    except ValueError:
                        out[k] = x
            rows.append(out)
    return rows


def mean_se(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"))


def judge(ours: float, target: float, spread: tuple | None = None) -> str:
    """Judgment rule (fixed before the runs): MATCH within 5 % rel. (or inside spread); near-zero targets use |diff| <= 0.05 / 0.2."""
    if spread is not None and spread[0] <= ours <= spread[1]:
        return "MATCH"
    d = abs(ours - target)
    if abs(target) < 0.1:
        return "MATCH" if d <= 0.05 else ("CLOSE" if d <= 0.2 else "DIVERGENT")
    rel = d / abs(target)
    return "MATCH" if rel <= 0.05 else ("CLOSE" if rel <= 0.2 else "DIVERGENT")
