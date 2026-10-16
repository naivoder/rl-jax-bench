"""Scores and the paper's analysis statistics (paper §2, footnote 4; official notebook ``percentile_ci``), plus the
one-factor experiment paired analysis (README) and the per-seed metrics aggregator.

  python -m src.eval --tag small            # aggregate runs/<tag>/<env>/seed_<n>/metrics.json -> summary + tables
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.stats import binom

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ scores (§2)
def seed_score(curve) -> float:
    """Per-seed score = mean of the evaluation returns at 100k..1M (step-0 point excluded)."""
    return float(np.mean(np.asarray(curve, float)[1:]))


def config_score(seed_scores) -> float:
    """Configuration score = median over its seeds (paper §2)."""
    return float(np.median(np.asarray(seed_scores, float)))


# ------------------------------------------------------------------ conditional 95th percentile + CI (footnote 4)
def p95(x) -> float:
    return float(np.percentile(np.asarray(x, float), 95))           # linear interpolation (official)


def percentile_ci(x, convention: str = "official"):
    """Binomial order-statistic CI for the 95th percentile.
    official: lo, hi = binom.interval(0.95, n, 0.95) used as 0-based indices into the ascending sort (1-based ranks
              lo+1, hi+1), hi clamped to n-1 (notebook percentile_ci).
    lh:       1-based ranks l = F^-1(0.025), h = F^-1(0.975) (README)."""
    xs = np.sort(np.asarray(x, float))
    n = len(xs)
    lo, hi = binom.interval(0.95, n, 0.95)
    lo, hi = int(lo), int(hi)
    if convention == "official":
        return float(xs[min(lo, n - 1)]), float(xs[min(hi, n - 1)])
    return float(xs[max(lo - 1, 0)]), float(xs[min(hi - 1, n - 1)])


def ci_ranks(n: int, convention: str = "official"):
    """1-based ascending ranks (l, h) of the CI ends (after clamping)."""
    lo, hi = binom.interval(0.95, n, 0.95)
    lo, hi = int(lo), int(hi)
    if convention == "official":
        return min(lo + 1, n), min(hi + 1, n)
    return max(lo, 1), min(hi, n)


def coverage_exact(n: int, convention: str = "official", q: float = 0.95) -> float:
    """P(X_(l) <= xi_q <= X_(h)) for continuous i.i.d. scores: with K = #{X_i <= xi} ~ Bin(n, q),
    X_(l) <= xi <=> K >= l and X_(h) >= xi <=> K <= h-1 (README; ties have probability 0)."""
    l, h = ci_ranks(n, convention)
    return float(binom.cdf(h - 1, n, q) - binom.cdf(l - 1, n, q))


def top_share(values, scores):
    """Official top set = the top floor((n-1)/20)+1 configurations; returns {value: share}."""
    values, scores = np.asarray(values), np.asarray(scores, float)
    m = (len(scores) - 1) // 20 + 1
    top = values[np.argsort(-scores)[:m]]
    return {v: float(np.mean(top == v)) for v in np.unique(values)}


# ------------------------------------------------------------------ one-factor experiment aggregation
def load_lanes(tag: str):
    lanes = []
    for f in sorted(glob.glob(os.path.join(REPO, "runs", tag, "*", "seed_*", "metrics.json"))):
        with open(f) as fh:
            lanes += json.load(fh)["lanes"]
    return lanes


def table(tag: str, ref: str = "REC"):
    from rl_jax_bench.stats import paired_compare
    lanes = load_lanes(tag)
    envs = sorted({l["env"] for l in lanes})
    rows = []
    for e in envs:
        L = [l for l in lanes if l["env"] == e]
        arms = list(dict.fromkeys(l["arm"] for l in L))
        sc = {a: {l["seed"]: l["score"] for l in L if l["arm"] == a} for a in arms}
        for a in arms:
            v = np.array(list(sc[a].values()), float)
            row = {"env": e, "arm": a, "n": len(v), "mean": float(np.nanmean(v)),
                   "se": float(np.nanstd(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"),
                   "final_mean": float(np.mean([l["final"] for l in L if l["arm"] == a]))}
            if a != ref and ref in sc:
                seeds = sorted(set(sc[a]) & set(sc[ref]))
                x = [sc[a][s] for s in seeds]
                y = [sc[ref][s] for s in seeds]
                row["ratio"] = float(np.mean(x) / np.mean(y)) if np.mean(y) != 0 else float("nan")
                if len(seeds) > 1:
                    pc = paired_compare(x, y)
                    row.update({"diff": pc["diff_mean"], "p_t": pc["t_p"], "boot95": pc["boot95"], "wins": pc["wins"],
                                "rel": pc["rel_improvement"], "rel_boot95": pc["rel_boot95"]})
            rows.append(row)
    return rows


def aggregate_seed_files(tag: str):
    """runs/seed_<n>/metrics.json (required layout): every env/arm lane of seed n from run <tag>."""
    by_seed = {}
    for l in load_lanes(tag):
        by_seed.setdefault(l["seed"], []).append({k: l[k] for k in ("env", "arm", "score", "final", "det_final",
                                                                       "curve_steps", "curve", "nonfinite")})
    for s, ls in by_seed.items():
        p = os.path.join(REPO, "runs", f"seed_{s}", "metrics.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        old = {}
        if os.path.exists(p):
            with open(p) as f:
                old = json.load(f)
        old[tag] = ls
        with open(p, "w") as f:
            json.dump(old, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    rows = table(a.tag)
    for r in rows:
        print("[METRIC] arm " + " ".join(f"{k}={(round(v, 4) if isinstance(v, float) else v)}" for k, v in r.items()))
    with open(os.path.join(REPO, "runs", a.tag, "table.json"), "w") as f:
        json.dump(rows, f, indent=1)
    aggregate_seed_files(a.tag)


if __name__ == "__main__":
    main()
