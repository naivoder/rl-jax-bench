"""Reproduction verdicts (numpy only, no jax).

    from src import verdicts as V

Rule: MATCH = within 5 % relative of the paper value, CLOSE = within 20 %, DIVERGENT beyond.
Three refinements for comparing Monte-Carlo means with numbers read off a paper's plot:

* ``judge`` also flags a judgment as *precision-limited* when moving the paper value by its read-off precision
  (e.g. +-10 regret units off a figure) would change the category.
* ``mc_limited`` flags a judgment that flips when our mean moves by +-k standard errors (Monte-Carlo noise).
* ``perrun_reading`` answers a different question from the mean-based judgment: is the paper value consistent
  with ONE of our runs (inside our 5-95 % per-run band)?  Use it next to the mean judgment whenever the paper's
  number is probably a single plotted realisation (no replication count stated, curves with sharp corners).

``loglog_slope_boot`` fits a rate exponent (slope of log mean vs log x) with a seed-bootstrap CI; the CI is
Monte-Carlo error only, not line misfit, so report local slopes too when the regime may be in transition.

Extracted from scripts/report.py (numerically identical: runs/full/report.json regenerates byte-for-byte
through these functions).

Two ordering-claim helpers (not used by this paper's scripts):

* ``ordering_verdict`` labels one pre-declared "arm A outperforms arm B" claim from its paired mean difference and its
  Holm-adjusted p value: REPRODUCED / REVERSED / NOT REPRODUCED (n.s.).
* ``fresh_seed_revision`` is a revision rule declared BEFORE any re-run: a re-read changes a primary
  verdict only when the decisive seed's verdict under the re-read differs from the primary verdict AND every fresh
  (never-before-run) seed gives that same new verdict.  It returns the revised verdict, or None (no revision).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["judge_category", "judge", "mc_limited", "perrun_reading", "loglog_slope_boot", "ordering_verdict",
           "fresh_seed_revision"]


def judge_category(rel: float, match: float = 0.05, close: float = 0.20) -> str:
    """MATCH / CLOSE / DIVERGENT for an absolute relative difference |ours - paper| / paper."""
    return "MATCH" if rel <= match else "CLOSE" if rel <= close else "DIVERGENT"


def judge(ours: float, paper: float, precision: float = 0.0, match: float = 0.05,
          close: float = 0.20) -> tuple[str, bool]:
    """(judgment of ours against paper, precision_limited).

    precision_limited is True when the judgment against paper - precision or paper + precision differs.
    """
    def cat(pv):
        return judge_category(abs(ours - pv) / pv, match, close)

    j = cat(paper)
    return j, {cat(paper - precision), cat(paper + precision)} != {j}


def mc_limited(ours: float, paper: float, se: float, k: float = 2.0, match: float = 0.05,
               close: float = 0.20) -> bool:
    """True when the judgment changes as ours moves by -k*se, 0, +k*se (Monte-Carlo-limited cell)."""
    j = judge_category(abs(ours - paper) / paper, match, close)
    return {judge_category(abs(ours + d - paper) / paper, match, close) for d in (-k * se, 0.0, k * se)} != {j}


def perrun_reading(dist: Sequence[float], paper: float, precision: float = 0.0,
                   band: Sequence[float] = (5, 95)) -> dict:
    """Where the paper value falls in OUR per-run distribution (for single-realisation targets).

    Returns {"reading": CONSISTENT | OUTSIDE (below p5) | OUTSIDE (above p95), "band": [lo, hi],
    "pct": percentile of the paper value among our runs, "precision_limited": bool}.
    """
    dist = np.asarray(dist)
    lo, hi = np.percentile(dist, band)
    pct = float((dist < paper).mean() * 100)

    def rd(v):
        return "CONSISTENT" if lo <= v <= hi else ("OUTSIDE (below p5)" if v < lo else "OUTSIDE (above p95)")

    r = rd(paper)
    return {"reading": r, "band": [float(lo), float(hi)], "pct": pct,
            "precision_limited": {rd(paper - precision), rd(paper + precision)} != {r}}


def loglog_slope_boot(xs: Sequence[float], seed_means, n_boot: int = 10_000,
                      rng_seed: int = 0) -> tuple[float, list[float]]:
    """OLS slope of log(mean over seeds) on log x, with a 95 % CI from resampling seeds.

    seed_means has shape (n_points, n_seeds); the same seed resample is used at every point.
    """
    lx = np.log(np.asarray(xs, float))
    Y = np.asarray(seed_means, float)
    est = np.polyfit(lx, np.log(Y.mean(1)), 1)[0]
    rng = np.random.default_rng(rng_seed)
    ns = Y.shape[1]
    bs = [np.polyfit(lx, np.log(Y[:, rng.integers(0, ns, ns)].mean(1)), 1)[0] for _ in range(n_boot)]
    return float(est), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def ordering_verdict(diff: float, p_holm: float, alpha: float = 0.05) -> str:
    """Verdict of a pre-declared ordering claim (diff = claimed-better arm minus the other, paired mean)."""
    if p_holm < alpha:
        return "REPRODUCED" if diff > 0 else "REVERSED"
    return "NOT REPRODUCED (n.s.)"


def fresh_seed_revision(primary: str, decisive: str, fresh: Sequence[str]) -> str | None:
    """Revised verdict under a re-read, or None.

    primary: the verdict as originally reported; decisive: the verdict under the re-read on the decisive seed (the
    seed the primary verdict was computed on); fresh: the verdicts under the re-read on seeds never run before the rule
    was declared.  Revises only if decisive != primary and every fresh verdict equals decisive (and there is at least
    one fresh seed).
    """
    fresh = list(fresh)
    if fresh and decisive != primary and all(v == decisive for v in fresh):
        return decisive
    return None
