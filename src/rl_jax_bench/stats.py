"""Seed-level statistics shared by every paper reproduction in this repository.

Everything here is numpy/scipy only (no jax), so report scripts can import it without the GPU bootstrap:

    from rl_jax_bench.stats import mean_se, boot_ci, paired_compare, holm

Conventions: uncertainty is the standard error over seeds (std ddof=1 / sqrt n),
never the std; an "outperforms" claim needs a paired test (paired t, plus a t-interval and a bootstrap CI);
families of primary tests get a Holm correction. Bootstrap resampling uses a fixed numpy seed so reports
regenerate byte-identically.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats as _st

__all__ = ["mean_se", "t_interval", "boot_ci", "paired_compare", "holm"]


def mean_se(x: Sequence[float]) -> tuple[float, float | None]:
    """(mean, standard error over the entries); SE is None for n < 2."""
    x = np.asarray(x, dtype=float)
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None
    return float(x.mean()), se


def t_interval(x: Sequence[float], level: float = 0.95) -> list[float]:
    """Student-t confidence interval of the mean of x (n - 1 degrees of freedom)."""
    x = np.asarray(x, dtype=float)
    m, se = mean_se(x)
    if se is None:
        return [m, m]
    tq = float(_st.t.ppf(0.5 + level / 2, len(x) - 1))
    return [m - tq * se, m + tq * se]


def boot_ci(x, n: int = 10_000, seed: int = 0, stat: Callable = np.mean) -> list[float]:
    """Percentile bootstrap 95 % CI of ``stat`` over the entries of x (fixed-seed resampling).

    Percentile bootstraps under-cover at n <= 5; report ``t_interval`` next to it for small n.
    """
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    v = np.array([stat(x[i]) for i in idx]) if stat is not np.mean else x[idx].mean(1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired_compare(a: Sequence[float], b: Sequence[float], *, n_boot: int = 10_000, rel_seed: int = 1) -> dict:
    """Paired comparison of per-seed values a (arm) against b (reference), same seed order.

    Returns the mean paired difference with its SE, a 95 % t-interval (absolute and relative to |mean b|), the
    paired t-test, a bootstrap CI of the difference, the Wilcoxon signed-rank p (n >= 5), the number of wins and
    the relative improvement (mean a - mean b) / |mean b| with a paired-bootstrap CI.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"paired_compare needs equal-length 1-d inputs, got {a.shape} and {b.shape}")
    d = a - b
    t = _st.ttest_rel(a, b)

    def rel(ii):
        return (a[ii].mean() - b[ii].mean()) / abs(b[ii].mean())

    rng = np.random.default_rng(rel_seed)
    rels = [rel(rng.integers(0, len(d), len(d))) for _ in range(n_boot)]
    w = _st.wilcoxon(a, b) if len(d) >= 5 else None
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    tq = float(_st.t.ppf(0.975, len(d) - 1))
    return {"diff_mean": float(d.mean()), "diff_se": se,
            "t95": [float(d.mean() - tq * se), float(d.mean() + tq * se)],
            "rel_t95": [float((d.mean() - tq * se) / abs(b.mean())), float((d.mean() + tq * se) / abs(b.mean()))],
            "diff_per_seed": d.round(3).tolist(), "t": float(t.statistic), "t_p": float(t.pvalue),
            "boot95": boot_ci(d, n=n_boot), "wins": int((d > 0).sum()), "n": len(d),
            "wilcoxon_p": float(w.pvalue) if w is not None else None,
            "rel_improvement": float(rel(np.arange(len(d)))),
            "rel_boot95": [float(np.percentile(rels, 2.5)), float(np.percentile(rels, 97.5))]}


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for a family {name: p} -> {name: {p_raw, p_holm}}."""
    names = sorted(pvals, key=pvals.get)
    m = len(names)
    out, run = {}, 0.0
    for i, n in enumerate(names):
        run = max(run, min(1.0, (m - i) * pvals[n]))
        out[n] = {"p_raw": float(pvals[n]), "p_holm": float(run)}
    return out
