"""Statistics for crossed seeds x worlds designs (one trained policy scored on every world), numpy + scipy only.

Checked against scipy and against src/algorithm.py on synthetic data (tests/test_crossed_stats.py).

Why it exists: rliable's stratified bootstrap resamples runs independently inside every task. When one policy (one
training seed) is scored on all worlds, the seed's main effect sits in every column and that resampling averages it
away (on one crossed design we analysed, its coverage was 0.22-0.56). The unit of replication is the seed. Use
`seed_mean_t_ci` for the declared world block, `interval_family(...)["two_way"]` (S >= 10) for new worlds.

Matrix convention: Y[s, w] = score (or paired difference) of seed s on world w.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st

__all__ = ["anova2", "seed_mean_t_ci", "interval_family", "maxt_signflip", "holm", "clopper_pearson",
           "escape_rate_test", "count_rule_oc", "poi_per_task", "poi_pooled", "run_score_profile"]


def anova2(Y):
    """Crossed two-way random effects, one observation per cell, method of moments (negative components -> 0)."""
    Y = np.asarray(Y, float)
    S, W = Y.shape
    g = Y.mean()
    rm, cm = Y.mean(1), Y.mean(0)
    msr = W * ((rm - g) ** 2).sum() / (S - 1)
    msc = S * ((cm - g) ** 2).sum() / (W - 1)
    res = Y - rm[:, None] - cm[None, :] + g
    mse = (res ** 2).sum() / ((S - 1) * (W - 1))
    s_run = max(0.0, (msr - mse) / W)
    s_w = max(0.0, (msc - mse) / S)
    tot = s_run + s_w + mse
    return {"s2_run": s_run, "s2_world": s_w, "s2_resid": mse,
            "share_run": s_run / tot if tot else None, "share_world": s_w / tot if tot else None,
            "share_resid": mse / tot if tot else None}


def seed_mean_t_ci(D, level=0.95):
    """t interval over per-seed means of D[s, w] (block-conditional estimand: the declared worlds)."""
    D = np.asarray(D, float)
    rm = D.mean(1) if D.ndim == 2 else D
    S = rm.shape[0]
    lo, hi = st.t.interval(level, S - 1, loc=rm.mean(), scale=rm.std(ddof=1) / math.sqrt(S))
    return [float(lo), float(hi)]


def interval_family(D, rng, B=10_000):
    """Four 95 % intervals for the mean of D[s, w]: (i) 'stratified' = rliable's per-world independent run resample
    (mis-specified for crossed designs, printed for comparison only), (ii) 'run_cluster', (iii) 'two_way' (seeds and
    worlds resampled), (iv) 't_run_means'. The draw order equals audit_sp.ci_four, so a shared rng gives equal output."""
    D = np.asarray(D, float)
    S, W = D.shape
    v1 = np.empty(B); v2 = np.empty(B); v3 = np.empty(B)
    cols = np.arange(W)
    for b in range(B):
        ri = rng.integers(0, S, (S, W))
        v1[b] = D[ri, cols[None, :]].mean()
        rr = rng.integers(0, S, S)
        v2[b] = D[rr].mean()
        wc = rng.integers(0, W, W)
        v3[b] = D[np.ix_(rr, wc)].mean()

    def pc(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {"mean": float(D.mean()), "stratified": pc(v1), "run_cluster": pc(v2), "two_way": pc(v3),
           "t_run_means": seed_mean_t_ci(D)}
    for k in ("stratified", "run_cluster", "two_way", "t_run_means"):
        out[k + "_width"] = out[k][1] - out[k][0]
    return out


def _tstat(x):
    n = x.shape[-1]
    return x.mean(-1) / (x.std(-1, ddof=1) / math.sqrt(n))


def maxt_signflip(Dmat, n_mc=None, rng=None, eps=1e-12):
    """Westfall-Young max-|t| sign-flip test over a family of paired-difference rows Dmat[c, w] that share worlds
    (one flip vector per resample, shared across rows). Exact enumeration of all 2^W flips when n_mc is None
    (W <= 20 advised), else n_mc Monte-Carlo flips. Returns {'single': [p_c], 'adjusted': [p_c]} (two-sided)."""
    Dmat = np.atleast_2d(np.asarray(Dmat, float))
    C, W = Dmat.shape
    if n_mc is None:
        if W > 22:
            raise ValueError(f"exact enumeration of 2^{W} flips refused; pass n_mc")
        flips = ((np.arange(2 ** W)[:, None] >> np.arange(W)[None, :]) & 1) * 2.0 - 1.0
    else:
        rng = rng if rng is not None else np.random.default_rng(0)
        flips = rng.choice([-1.0, 1.0], size=(n_mc, W))
    tobs = np.abs(_tstat(Dmat))
    tperm = np.abs(np.stack([_tstat(flips * Dmat[i][None, :]) for i in range(C)]))
    single = [float(np.mean(tperm[i] >= tobs[i] - eps)) for i in range(C)]
    mx = tperm.max(0)
    adjusted = [float(np.mean(mx >= tobs[i] - eps)) for i in range(C)]
    return {"single": single, "adjusted": adjusted, "t_obs": tobs.tolist(), "n_flips": int(flips.shape[0])}


def holm(p):
    """Holm step-down adjusted p-values; p is a dict name -> p."""
    names = sorted(p, key=p.get)
    m, run, out = len(names), 0.0, {}
    for i, n in enumerate(names):
        run = max(run, min(1.0, (m - i) * p[n]))
        out[n] = run
    return out


def clopper_pearson(k, n, level=0.95):
    r = st.binomtest(int(k), int(n)).proportion_ci(confidence_level=level, method="exact")
    return [float(r.low), float(r.high)]


def escape_rate_test(k_treat, n_treat, k_ctrl, n_ctrl, level=0.95):
    """Grade a floor-or-escape recipe as a rate against its control: one-sided Fisher exact p (treat > control),
    two-sided p, and Clopper-Pearson intervals for both rates."""
    tab = [[k_treat, n_treat - k_treat], [k_ctrl, n_ctrl - k_ctrl]]
    return {"fisher_p_greater": float(st.fisher_exact(tab, alternative="greater").pvalue),
            "fisher_p_two_sided": float(st.fisher_exact(tab).pvalue),
            "treat_cp": clopper_pearson(k_treat, n_treat, level), "ctrl_cp": clopper_pearson(k_ctrl, n_ctrl, level)}


def count_rule_oc(n, k_go, p):
    """P(at least k_go of n independent runs succeed | per-run probability p): the operating characteristic of a
    'GO if >= k/n runs pass' rule."""
    return float(st.binom.sf(k_go - 1, n, p))


def poi_per_task(x, y):
    """The paper's P(X > Y): per-task Mann-Whitney (ties 1/2) averaged over tasks; x (Nx, M), y (Ny, M)."""
    x = np.atleast_2d(np.asarray(x, float)); y = np.atleast_2d(np.asarray(y, float))
    gt = (x[:, None, :] > y[None, :, :]).mean((0, 1)); eq = (x[:, None, :] == y[None, :, :]).mean((0, 1))
    return float(np.mean(gt + 0.5 * eq))


def poi_pooled(x, y):
    """Pooled Mann-Whitney over all run-task scores (compares X on task a with Y on task b). NOT the paper's
    estimand; kept to show the difference."""
    xs, ys = np.ravel(x), np.ravel(y)
    d = xs[:, None] - ys[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (xs.size * ys.size))


def run_score_profile(x, taus):
    """The paper's run-score distribution: fraction of all (run, task) scores > tau, for every tau."""
    x = np.ravel(np.asarray(x, float))
    taus = np.asarray(taus, float)
    return (x[None, :] > taus[:, None]).mean(1)
