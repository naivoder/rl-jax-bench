"""Host-side (numpy/scipy) metrics and aggregation for the section VII reproduction.

Depends on scipy and on rl_jax_bench.stats (imported lazily).

Metrics (positions only, [p_x, p_y] = state components 0 and 2):
  OSPA  (Schuhmacher-Vo-Vo 2008), order p=2, cut-off c=10 -- the paper's metric (section VII).
  GOSPA (Rahmathullah-Garcia-Fernandez-Svensson 2017), alpha=2, p=2, c=10 with the localisation / missed / false
        decomposition of the official `GOSPA.m` / `ComputeGOSPAerror.m` -- a labelled EXTRA (the official code reports it).
Both use the SAME optimal assignment: LAP on min(||x-y||, c)^p. With S = sum over assigned pairs of that cost:
  OSPA^p  = (S + c^p |n-m|) / max(n, m)            (0 if both empty; c if exactly one is empty)
  GOSPA^p = S + (c^p / alpha) |n-m|                (alpha = 2: pairs at the cut-off count as one miss + one false)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _ev_pairs(est_pos, truth_pos, p, c):
    """-> (S, n_loc_cost, n_cut, m, n): optimal sum of min(d,c)^p over min(m,n) pairs, localisation part (pairs with d<c),
    number of assigned pairs at the cut-off (d >= c)."""
    e = np.asarray(est_pos, dtype=float).reshape(-1, 2)
    t = np.asarray(truth_pos, dtype=float).reshape(-1, 2)
    m, n = len(e), len(t)
    if m == 0 or n == 0:
        return 0.0, 0.0, 0, m, n
    d = cdist(e, t)
    cost = np.minimum(d, c) ** p
    r, cc = linear_sum_assignment(cost)
    dd = d[r, cc]
    loc = float(np.sum(dd[dd < c] ** p))
    return float(cost[r, cc].sum()), loc, int(np.sum(dd >= c)), m, n


def ospa(est_pos, truth_pos, p=2, c=10.0) -> float:
    """OSPA distance between two position sets (arrays [m,2], [n,2]); d(empty, empty) = 0, one empty -> c."""
    S, _, _, m, n = _ev_pairs(est_pos, truth_pos, p, c)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(c)
    return float(((S + c ** p * abs(n - m)) / max(m, n)) ** (1.0 / p))


def gospa(est_pos, truth_pos, p=2, c=10.0, alpha=2.0) -> dict:
    """GOSPA between two position sets. Returns dict(d, loc, missed, false): d = (loc+missed+false)^(1/p) for alpha=2, where
    loc/missed/false are the (un-rooted) p-th-power cost components of the official decomposition. For alpha != 2 the
    decomposition entries are NaN (only valid for alpha=2, as in the official code)."""
    S, loc, ncut, m, n = _ev_pairs(est_pos, truth_pos, p, c)
    dummy = c ** p / alpha
    total = S + dummy * abs(n - m)
    out = dict(d=float(total ** (1.0 / p)))
    if alpha == 2:
        out["loc"] = float(loc)
        out["missed"] = float(dummy * (max(n - m, 0) + ncut))  # truth left unassigned + pairs at the cut-off
        out["false"] = float(dummy * (max(m - n, 0) + ncut))  # estimates left unassigned + pairs at the cut-off
    else:
        out.update(loc=np.nan, missed=np.nan, false=np.nan)
    return out


def evaluate_run(est_mask, est_x, truth, k_axis=None, p=2, c=10.0) -> dict:
    """One run: per-step squared OSPA and squared GOSPA(+decomposition).

    est_mask bool[K,E], est_x [K,E,4]; truth dict(X[nT,K,4], alive[nT,K]); k_axis = step indices to score (default all).
    Returns dict of float64[len(k_axis)]: ospa2, gospa2, g_loc, g_mis, g_fal (g_loc+g_mis+g_fal == gospa2), n_est, n_true.
    """
    est_mask = np.asarray(est_mask, dtype=bool)
    est_x = np.asarray(est_x, dtype=float)
    X = np.asarray(truth["X"], dtype=float)
    alive = np.asarray(truth["alive"], dtype=bool)
    K = est_mask.shape[0]
    ks = np.arange(K) if k_axis is None else np.asarray(k_axis, dtype=int)
    o2, g2, gl, gm, gf, ne, nt = (np.zeros(len(ks)) for _ in range(7))
    dummy = c ** p / 2.0
    for i, k in enumerate(ks):
        e = est_x[k][est_mask[k]][:, [0, 2]]
        t = X[alive[:, k], k][:, [0, 2]]
        S, loc, ncut, m, n = _ev_pairs(e, t, p, c)
        ne[i], nt[i] = m, n
        if m == 0 and n == 0:
            continue
        if m == 0 or n == 0:
            o2[i] = c ** 2
        else:
            o2[i] = ((S + c ** p * abs(n - m)) / max(m, n)) ** (2.0 / p)
        tot = S + dummy * abs(n - m)
        g2[i] = tot ** (2.0 / p)
        if p == 2:  # squared GOSPA == total cost
            gl[i], gm[i], gf[i] = loc, dummy * (max(n - m, 0) + ncut), dummy * (max(m - n, 0) + ncut)
    return dict(ospa2=o2, gospa2=g2, g_loc=gl, g_mis=gm, g_fal=gf, n_est=ne, n_true=nt)


def evaluate_runs(est_mask, est_x, truth, k_axis=None, p=2, c=10.0) -> dict:
    """est_mask[R,K,E], est_x[R,K,E,4] -> same keys as evaluate_run stacked to [R, K]."""
    rs = [evaluate_run(est_mask[r], est_x[r], truth, k_axis, p, c) for r in range(len(est_mask))]
    return {k: np.stack([r_[k] for r_ in rs]) for k in rs[0]}


# ------------------------------------------------------------------------------------- aggregation
def rms_per_step(d2) -> np.ndarray:
    """Figure 4: sqrt(mean over runs of d^2) per step. d2: [R,K]."""
    return np.sqrt(np.mean(np.asarray(d2, dtype=float), axis=0))


def joint_rms(d2) -> float:
    """Table II: sqrt( sum_k sum_run d^2 / (K R) ). d2: [R,K]."""
    return float(np.sqrt(np.mean(np.asarray(d2, dtype=float))))


def run_stat(d2) -> np.ndarray:
    """Per-run statistic s_run = mean_k d^2 (the unit of the paired tests). d2: [R,K] -> [R]."""
    return np.mean(np.asarray(d2, dtype=float), axis=1)


def se_delta(s) -> float:
    """Delta-method SE of the table value sqrt(mean s): sd(s) / (2 sqrt(n) sqrt(mean s))."""
    s = np.asarray(s, dtype=float)
    return float(s.std(ddof=1) / (2.0 * np.sqrt(len(s)) * np.sqrt(s.mean())))


def boot_se_joint(s, n_boot=10_000, seed=0) -> dict:
    """Bootstrap over runs (n_boot resamples, seeded) of the table value sqrt(mean s): SE and percentile 95 % CI."""
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    v = np.sqrt(s[idx].mean(1))
    return dict(se=float(v.std(ddof=1)), ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])


def boot_se_rms_per_step(d2, n_boot=10_000, seed=0, chunk=500) -> np.ndarray:
    """Bootstrap SE (over runs) of the per-step RMS curve. d2 [R,K] -> [K]."""
    d2 = np.asarray(d2, dtype=float)
    R, K = d2.shape
    rng = np.random.default_rng(seed)
    acc = np.zeros(K)
    acc2 = np.zeros(K)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, R, (b, R))
        v = np.sqrt(d2[idx].mean(1))  # [b,K]
        acc += v.sum(0)
        acc2 += (v ** 2).sum(0)
        done += b
    mean = acc / n_boot
    return np.sqrt(np.maximum(acc2 / n_boot - mean ** 2, 0.0) * n_boot / (n_boot - 1))


def table_cell(d2, n_boot=10_000, seed=0) -> dict:
    """Table II cell: value, delta-method SE, bootstrap SE + CI, from d2 [R,K]."""
    s = run_stat(d2)
    b = boot_se_joint(s, n_boot, seed)
    return dict(value=joint_rms(d2), se_delta=se_delta(s), se_boot=b["se"], ci95_boot=b["ci95"], n=len(s))


# ---------------------------------------------------------------- thin wrappers to rl_jax_bench.stats
def _ev_stats():
    from rl_jax_bench import stats
    return stats


def paired_test(s_a, s_b, n_boot=10_000) -> dict:
    """Paired comparison of per-run statistics s (same runs, same measurement sets): stats.paired_compare(a, b); diff = a - b,
    so a negative diff means arm a has the lower error."""
    return _ev_stats().paired_compare(s_a, s_b, n_boot=n_boot)


def boot_ci_stat(x, n=10_000, seed=0):
    return _ev_stats().boot_ci(x, n=n, seed=seed)


def holm_adjust(pvals: dict) -> dict:
    return _ev_stats().holm(pvals)
