"""Evaluation = the paper's metric (official plot.py): undiscounted return of TRAINING episodes (exploration on, no
separate evaluation run), per run the mean of the returns of episodes ENDING in each bin (i*stride, (i+1)*stride],
curve = mean over runs, band = s.e. over runs (code; captions say 90 % CI -- both are reported, D18).
End-point ("final bin") = the run's last bin. Plus the E8 ETTm2 prediction-error summary.
"""
from __future__ import annotations

import sys

import numpy as np

from rl_jax_bench import stats as _stats  # noqa: E402

mean_se = _stats.mean_se
paired_compare = _stats.paired_compare
holm = _stats.holm


def run_bins(e: np.ndarray, total: int, stride: int) -> np.ndarray:
    """Per-run bin means (official avg_return_curve inner loop); NaN for a bin with no finished episode."""
    e = np.asarray(e, float).reshape(-1, 3)
    nb = total // stride
    out = np.full(nb, np.nan)
    for i in range(nb):
        m = (e[:, 0] > i * stride) & (e[:, 0] <= (i + 1) * stride)
        if m.any():
            out[i] = e[m, 1].mean()
    return out


def curve(runs: list[np.ndarray], total: int, stride: int):
    """Official avg_return_curve over runs: (steps, mean, s.e. with ddof 0 as np.std, 90 % t-CI half-width)."""
    b = np.stack([run_bins(e, total, stride) for e in runs])
    steps = np.arange(stride, total + stride, stride)[: b.shape[1]]
    m = b.mean(0)
    se = b.std(0) / np.sqrt(len(runs))
    from scipy import stats as st
    n = len(runs)
    h90 = st.t.ppf(0.95, n - 1) * b.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full_like(m, np.nan)
    return steps, m, se, h90


def figure_stride(points) -> int:
    """Bin stride (env steps) of a digitized paper curve: the figures plot 40 bins per run (official plot.py), so the
    stride is total/40 -- 125k on Figure 4 (5M), 50k on Figures 7/8 (2M). Vertices repeat (path segments) and carry a
    ~1e-4 M calibration offset, so x (in millions) is rounded to 1e-3 before differencing."""
    xs = np.unique(np.round(np.asarray(points, float)[:, 0], 3))
    d = np.unique(np.round(np.diff(xs), 3))
    if len(d) != 1:
        raise ValueError(f"non-uniform figure grid: {d}")
    return int(round(d[0] * 1e6))


def grid_value(points, x_millions: float):
    """Paper value AT a figure grid point (mean of the digitized vertices within stride/10 of x), never an
    interpolation between grid points; None when x is not a grid point. ()"""
    pts = np.asarray(points, float)
    strd = figure_stride(pts) / 1e6
    m = np.abs(pts[:, 0] - float(x_millions)) < strd / 10
    return float(pts[m, 1].mean()) if m.any() else None


def summarize_run(e: np.ndarray, total: int, stride: int) -> dict:
    b = run_bins(e, total, stride)
    finite = b[np.isfinite(b)]
    e = np.asarray(e, float).reshape(-1, 3)
    return dict(final_bin=float(b[-1]) if len(b) else float("nan"),
                last_finite_bin=float(finite[-1]) if len(finite) else float("nan"),
                auc_bins=float(np.nanmean(b)) if len(finite) else float("nan"),
                n_episodes=int(len(e)), bins=b.tolist(), stride=stride,
                mean_len=float(e[:, 2].mean()) if len(e) else float("nan"))


def discounted_returns(c: np.ndarray, gamma: float) -> np.ndarray:
    """Official stream_td.py: G_t = c_t + gamma G_{t+1}, G_T = 0, over the logged (rescaled) cumulants."""
    G = np.empty_like(c)
    g = 0.0
    for t in range(len(c) - 1, -1, -1):
        g = c[t] + gamma * g
        G[t] = g
    return G


def _ett_windows(pred: np.ndarray, G: np.ndarray, window: int) -> dict:
    f, l_ = slice(0, window), slice(len(G) - window, len(G))
    lt = slice(len(G) - window, len(G) - 500)   # drop the last 500 steps, where G is cut by the series end (5/(1-gamma))
    return dict(mse_first=float(np.mean((pred[f] - G[f]) ** 2)), mse_last=float(np.mean((pred[l_] - G[l_]) ** 2)),
                mse_last_excl_tail=float(np.mean((pred[lt] - G[lt]) ** 2)),
                corr_last_excl_tail=float(np.corrcoef(pred[lt], G[lt])[0, 1]),
                corr_last=float(np.corrcoef(pred[l_], G[l_])[0, 1]),
                G_var_last=float(np.var(G[l_])), mean_abs_pred_last=float(np.mean(np.abs(pred[l_]))),
                mean_G_last=float(np.mean(G[l_])))


def ett_summary(pred: np.ndarray, G: np.ndarray, cum: np.ndarray, window: int = 5000) -> dict:
    """MSE of the sigma-rescaled prediction vs the realised discounted return, first / last `window` steps
    (official plots show steps [0, 5000) and [T-5000, T)), plus correlations.

    Alignment . pred[j] is v(S_{j+1}) after the j-th update (official stream_td.py logs
    predict(s') after update_params), and cum[j] is the cumulant of the transition S_j -> S_{j+1}, so
    G[j] = cum[j] + gamma G[j+1] is the return FROM S_j. The return v(S_{j+1}) estimates is G[j+1].
      * keys without suffix  : pred[:-1] vs G[1:]  (correct alignment; the repo's primary E8 metric since the review pass)
      * keys '<k>_official'  : pred[j] vs G[j]     (one step off; = the official plotting, kept for figure overlays
                                                    and for comparison with small-scale / full-scale numbers, which used it)"""
    out = _ett_windows(np.asarray(pred)[:-1], np.asarray(G)[1:], window)
    out.update({f"{k}_official": v for k, v in _ett_windows(pred, G, window).items()})
    return out
