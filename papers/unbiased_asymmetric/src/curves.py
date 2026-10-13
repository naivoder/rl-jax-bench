"""Learning-curve comparison helpers (numpy + scipy, no jax), extracted from src/eval.py and pinned against it by
tests/test_shared_curves.py. Three hazards they address:

* Time-to-threshold on two curves sampled on different grids (a 101-vertex PDF curve vs a 400-point run log):
  the first VERTEX at or above the threshold is biased late by up to one grid interval on the coarser curve.
  ``first_crossing`` interpolates linearly between the bracketing vertices, the same estimator for both curves.
* Runs vmapped in one program and stopped at a common WALL CLOCK reach different env steps when an update collects a
  fixed number of episodes (runs that learned have shorter episodes and stop earlier). Comparing each run's own last
  window then compares learned runs at earlier steps. ``common_window`` / ``common_step_values`` evaluate every run
  over the same window (t_c - W, t_c], t_c = the smallest stop step over all runs of all compared arms.
* Bimodal cells (some runs stuck at a trivial policy): the mean moves with the stuck draw. ``rate_compare`` tests the
  stuck/solved fraction with Fisher's exact test, and ``welch_p`` gives the unpaired test beside any nominal pairing.

    import curves   # with src/ on the import path
    t_star = curves.first_crossing(t, y, thr)
    lo, hi = curves.common_window([run_a_t, run_b_t], W)
    vals = curves.common_step_values(list_of_(t, y), W)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def first_crossing(t, y, thr: float, interpolate: bool = True) -> float:
    """First t at which the polyline (t, y) reaches ``thr`` (y >= thr); NaN if never.

    With ``interpolate`` (default) the crossing is placed linearly between the last vertex below and the first vertex
    at or above ``thr``; ``interpolate=False`` returns the first such vertex (the biased legacy estimator)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    idx = np.nonzero(y >= thr)[0]
    if not len(idx):
        return float("nan")
    i = int(idx[0])
    if not interpolate or i == 0:
        return float(t[i])
    return float(t[i - 1] + (thr - y[i - 1]) / (y[i] - y[i - 1]) * (t[i] - t[i - 1]))


def window_mean(t, y, t0: float, t1: float) -> float:
    """Mean of y over the points with t in (t0, t1]; NaN if the window holds no point."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    sel = (t > t0) & (t <= t1)
    return float(y[sel].mean()) if sel.any() else float("nan")


def common_window(stop_steps: Sequence, W: float) -> tuple[float, float]:
    """(t_c - W, t_c] with t_c the smallest stop step. ``stop_steps`` may be a flat list of floats or a list of
    per-run time arrays (their last element is the stop step)."""
    ends = [float(np.asarray(s, float).reshape(-1)[-1]) for s in stop_steps]
    if not ends:
        raise ValueError("common_window needs at least one run")
    tc = min(ends)
    return tc - float(W), tc


def common_step_values(curves: Sequence[tuple], W: float, window: tuple[float, float] | None = None) -> list[float]:
    """Each run's window mean over the common window. ``curves`` is a list of (t, y); pass ``window`` to use a
    window computed over a larger set of runs (e.g. both arms of a between-arm test)."""
    lo, hi = window if window is not None else common_window([c[0] for c in curves], W)
    return [window_mean(t, y, lo, hi) for t, y in curves]


def welch_p(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Two-sided Welch t-test p value; None if either side has fewer than 2 values."""
    from scipy import stats as sst
    a, b = [float(x) for x in a], [float(x) for x in b]
    if len(a) < 2 or len(b) < 2:
        return None
    return float(sst.ttest_ind(a, b, equal_var=False).pvalue)


def rate_compare(k_a: int, n_a: int, k_b: int, n_b: int) -> dict:
    """Fisher exact test of two event rates (e.g. stuck runs k of n per implementation or arm)."""
    from scipy import stats as sst
    if not (0 <= k_a <= n_a and 0 <= k_b <= n_b) or n_a == 0 or n_b == 0:
        raise ValueError("need 0 <= k <= n and n > 0 on both sides")
    res = sst.fisher_exact([[k_a, n_a - k_a], [k_b, n_b - k_b]])
    return dict(rate_a=k_a / n_a, rate_b=k_b / n_b, diff=k_a / n_a - k_b / n_b, fisher_p=float(res.pvalue))
