"""Evaluation procedures built on src/algorithm.py: the Gate-1 ground-truth checks the README, D9, D10, D2.5,
D3, D5.2), aggregate tables with stratified-bootstrap CIs for a dict of score matrices (Fig 7 / 9 / A.29 analogue),
and the few-run study on a run pool (Fig 2-4, 6, A.15-A.17 procedures, used by E1-sub / E6).

Every function takes an explicit PRNG key (or master seed + branch ids) and returns plain dicts for metrics.json.
"""
from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, optimize
from scipy import stats as st

from . import algorithm as A

F64 = jnp.float64


# ------------------------------------------------------------------------------------------------ exact enumeration
def enumerate_bootstrap(x, stat):
    """All N^(N*M) equally likely ordered stratified replicates of x (N, M) -> stat values (D4.2 exact)."""
    x = np.asarray(x, np.float64)
    N, M = x.shape
    idx = np.array(list(itertools.product(range(N), repeat=N * M)), dtype=np.int32).reshape(-1, N, M)
    return np.asarray(jax.vmap(lambda ii: A._as_vec(stat(A.gather_runs(jnp.asarray(x), ii))))(jnp.asarray(idx)))


def closed_form_moments(x, gamma=1.0, tau=0.5):
    """D4.2: E* and Var* of MEAN*, F_hat*(tau), OG* under the stratified bootstrap of x (N, M), plug-in variances."""
    x = np.asarray(x, np.float64)
    N, M = x.shape
    s2 = x.var(axis=0)                                   # ddof 0
    f = (x > tau).mean(axis=0)
    g = np.maximum(gamma - x, 0.0)
    return dict(mean=(x.mean(), (s2 / N).sum() / M ** 2),
                profile=((x > tau).mean(), (f * (1 - f) / N).sum() / M ** 2),
                og=(g.mean(), (g.var(axis=0) / N).sum() / M ** 2))


# ------------------------------------------------------------------------------------------------ toy checks
D9_X = np.array([[0.1, 0.9], [0.3, 0.3], [1.5, 0.6]]).T   # (runs, tasks) layout
D9_Y = np.array([[0.2, 0.9], [0.0, 0.4], [0.6, 0.8]]).T
D9_POINTS = {  # exact values of the README (hand derivations there)
    "mean": (1.85 / 3, 1.45 / 3), "median": (0.5, 0.55), "iqm_a": (0.525, 0.5), "iqm_b": (0.4, 0.4),
    "iqm_c": (0.5, 0.5), "og": (1 - 3.2 / 6, 1 - 2.9 / 6),
    "prof_0.3": (0.5, 4 / 6), "prof_0.5": (0.5, 0.5), "prof_1.0": (1 / 6, 0.0),
    "avgprof_0.3": (2 / 3, 2 / 3), "avgprof_0.5": (1 / 3, 2 / 3), "avgprof_1.0": (1 / 3, 0.0),
    "poi_xy": (0.5, 0.5)}
D9_BOOT = {"mean": (0.616667, 0.0201389), "prof_0.5": (0.5, 0.0138889), "og": (0.466667, 0.0111111),
           "median": (0.53125, 0.0396484), "iqm_a": (0.5546875, 0.0256421), "iqm_c": (0.5385417, 0.0258236)}
D9_PCT = {"mean": (1 / 3, 0.9), "median": (0.3, 0.9), "iqm_a": (0.325, 0.9), "og": (0.266667, 0.666667)}


def _d9_stats():
    return {"mean": A.agg_mean, "median": A.agg_median, "iqm_a": lambda s: A.agg_iqm(s, "a"),
            "iqm_b": lambda s: A.agg_iqm(s, "b"), "iqm_c": lambda s: A.agg_iqm(s, "c"), "og": A.agg_og,
            "prof_0.5": lambda s: A.score_profile(s, [0.5])[0]}


def toys_toy(key, B: int = 200_000) -> dict:
    """D9: point estimates (exact), the 64-atom bootstrap (exact enumeration vs D9 table vs closed forms), our MC
    bootstrap vs the enumeration (4 MC SE on E*, Var*; KS distance <= 0.01), percentile CI on the 64 atoms."""
    out = {"points": {}, "boot": {}, "pct": {}}
    X, Y = jnp.asarray(D9_X), jnp.asarray(D9_Y)
    f = {"mean": A.agg_mean, "median": A.agg_median, "iqm_a": lambda s: A.agg_iqm(s, "a"),
         "iqm_b": lambda s: A.agg_iqm(s, "b"), "iqm_c": lambda s: A.agg_iqm(s, "c"), "og": A.agg_og}
    for t in (0.3, 0.5, 1.0):
        f[f"prof_{t}"] = (lambda tt: lambda s: A.score_profile(s, [tt])[0])(t)
        f[f"avgprof_{t}"] = (lambda tt: lambda s: A.avg_score_profile(s, [tt])[0])(t)
    for k, fn in f.items():
        got = (float(fn(X)), float(fn(Y)))
        out["points"][k] = dict(ours=got, truth=D9_POINTS[k], max_abs_err=max(abs(g - t) for g, t in zip(got, D9_POINTS[k])))
    pxy, pyx = float(A.poi(X, Y)), float(A.poi(Y, X))
    out["points"]["poi_xy"] = dict(ours=(pxy, pyx), truth=D9_POINTS["poi_xy"], max_abs_err=max(abs(pxy - .5), abs(pyx - .5)))
    out["points_pass"] = all(v["max_abs_err"] <= 1e-12 for v in out["points"].values())

    cf = closed_form_moments(D9_X)
    for i_k, (k, fn) in enumerate(_d9_stats().items()):
        if k == "iqm_b":
            continue
        atoms = enumerate_bootstrap(D9_X, fn)[:, 0]
        e_ex, v_ex = float(atoms.mean()), float(atoms.var())
        # stable per-statistic key (was hash(k) % 1000: Python's str hash is salted per process -> not reproducible)
        mc = np.asarray(A.bootstrap_dist(jax.random.fold_in(key, i_k), (X,), fn, B))[:, 0]
        se_e, se_v = math.sqrt(v_ex / B), math.sqrt(max(np.mean((atoms - e_ex) ** 4) - v_ex ** 2, 0) / B)
        # CDF distance at the atoms; values rounded to 1e-12 so 1-ulp evaluation-order differences do not split atoms
        mcr, atr = np.round(mc, 12), np.round(atoms, 12)
        u = np.unique(atr)
        ks = float(np.max(np.abs(np.searchsorted(np.sort(mcr), u, side="right") / B
                                 - np.searchsorted(np.sort(atr), u, side="right") / len(atr))))
        row = dict(n_atoms=len(atoms), exact_E=e_ex, exact_Var=v_ex, d9_table=D9_BOOT.get(k),
                   table_err=max(abs(e_ex - D9_BOOT[k][0]), abs(v_ex - D9_BOOT[k][1])) if k in D9_BOOT else None,
                   mc_E=float(mc.mean()), mc_Var=float(mc.var()), z_E=(float(mc.mean()) - e_ex) / se_e if se_e else 0.0,
                   z_Var=(float(mc.var()) - v_ex) / se_v if se_v else 0.0, ks=ks)
        cfk = {"mean": "mean", "og": "og", "prof_0.5": "profile"}.get(k)
        if cfk:
            row["closed_form"] = cf[cfk]
            row["closed_form_err"] = max(abs(cf[cfk][0] - e_ex), abs(cf[cfk][1] - v_ex))
        row["pass"] = bool((row["table_err"] is None or row["table_err"] <= 5e-7) and abs(row["z_E"]) <= 4
                           and abs(row["z_Var"]) <= 4 and ks <= 0.01 and row.get("closed_form_err", 0) <= 1e-12)
        out["boot"][k] = row
        if k in D9_PCT:
            ci = np.asarray(A.ci_from_dist(jnp.asarray(atoms[:, None]), jnp.asarray([fn(X)]), "percentile"))[:, 0]
            out["pct"][k] = dict(ours=ci.tolist(), truth=D9_PCT[k], err=float(np.abs(ci - np.array(D9_PCT[k])).max()))
    out["boot_pass"] = all(r["pass"] for r in out["boot"].values())
    out["pct_pass"] = all(r["err"] <= 5e-7 for r in out["pct"].values())
    return out


def toys_closed_forms(key, n_cases: int = 30, B: int = 100_000) -> dict:
    """D4.2 property test: (i) exact enumeration on random small (N, M) equals the closed forms to 1e-12;
    (ii) MC bootstrap on random larger (N, M) agrees with the closed forms within 4 MC SE."""
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 31 - 1)))
    rows_exact, rows_mc = [], []
    for c in range(n_cases):
        N, M = [(2, 2), (2, 3), (3, 2)][c % 3]
        x = np.round(rng.normal(0.6, 0.5, (N, M)), 2 if c % 2 else 6)
        cf = closed_form_moments(x)
        for k, fn in (("mean", A.agg_mean), ("og", A.agg_og), ("profile", lambda s: A.score_profile(s, [0.5])[0])):
            atoms = enumerate_bootstrap(x, fn)[:, 0]
            rows_exact.append(max(abs(atoms.mean() - cf[k][0]), abs(atoms.var() - cf[k][1])))
        # bootstrap mean of P(X>Y) = point estimate (independent resampling)
        y = np.round(rng.normal(0.5, 0.5, (N, M)), 1)
        Nn = N
        ix = np.array(list(itertools.product(range(Nn), repeat=Nn * M))).reshape(-1, Nn, M)
        if len(ix) <= 729:
            vals = jax.vmap(lambda a: jax.vmap(lambda b: A.poi(A.gather_runs(jnp.asarray(x), a), A.gather_runs(jnp.asarray(y), b)))(jnp.asarray(ix)))(jnp.asarray(ix))
            rows_exact.append(abs(float(np.asarray(vals).mean()) - float(A.poi(jnp.asarray(x), jnp.asarray(y)))))
    for c in range(8):
        N, M = (3, 5, 10, 25)[c % 4], (6, 26)[c % 2]
        x = rng.gamma(2.0, 0.4, (N, M))
        cf = closed_form_moments(x)
        for k, fn in (("mean", A.agg_mean), ("og", A.agg_og), ("profile", lambda s: A.score_profile(s, [0.5])[0])):
            d = np.asarray(A.bootstrap_dist(jax.random.fold_in(key, 100 * c + len(k)), (jnp.asarray(x),), fn, B))[:, 0]
            se_e = math.sqrt(cf[k][1] / B)
            m4 = np.mean((d - d.mean()) ** 4)
            se_v = math.sqrt(max(m4 - d.var() ** 2, 1e-300) / B)
            rows_mc.append(dict(shape=[N, M], stat=k, z_E=float((d.mean() - cf[k][0]) / se_e) if se_e else 0.0,
                                z_Var=float((d.var() - cf[k][1]) / se_v) if se_v else 0.0))
    return dict(exact_max_abs_err=float(max(rows_exact)), n_exact=len(rows_exact), mc=rows_mc,
                mc_max_abs_z=float(max(max(abs(r["z_E"]), abs(r["z_Var"])) for r in rows_mc)),
                pass_=bool(max(rows_exact) <= 1e-12 and max(max(abs(r["z_E"]), abs(r["z_Var"])) for r in rows_mc) <= 4.5))


def toys_poi_identities(key, n_cases: int = 50) -> dict:
    """D2.5: P(X>Y) + P(Y>X) = 1 exactly; invariance to strictly increasing per-task transforms; P(X>X) = 1/2."""
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 31 - 1)))
    e_sum, e_inv, e_self = 0.0, 0.0, 0.0
    for c in range(n_cases):
        M = int(rng.integers(1, 30))
        x = np.round(rng.normal(0, 1, (int(rng.integers(2, 12)), M)), 1)
        y = np.round(rng.normal(0.2, 1, (int(rng.integers(2, 12)), M)), 1)   # rounding -> ties
        p, q = float(A.poi(jnp.asarray(x), jnp.asarray(y))), float(A.poi(jnp.asarray(y), jnp.asarray(x)))
        e_sum = max(e_sum, abs(p + q - 1))
        a, b = rng.uniform(0.5, 3, M), rng.uniform(-2, 2, M)          # per-task strictly increasing maps
        tx = lambda v: np.exp(a * v) + b + v ** 3  # noqa: E731
        e_inv = max(e_inv, abs(float(A.poi(jnp.asarray(tx(x)), jnp.asarray(tx(y)))) - p))
        e_self = max(e_self, abs(float(A.poi(jnp.asarray(x), jnp.asarray(x))) - 0.5))
    return dict(max_sum_err=e_sum, max_invariance_err=e_inv, max_self_err=e_self,
                pass_=bool(max(e_sum, e_inv, e_self) <= 1e-12))


def toys_profile_unbiased(key, reps: int = 20_000, N: int = 3, legacy: bool = False, avg_Ns=(3, 10, 100),
                           min_dist: float = 0.03) -> dict:
    """D3.1/D3.2 by simulation against a known tail: X_m ~ Beta(a_m, b_m), M = 5. The run-score profile is unbiased
    with Var = (1/M^2) sum F_m(1-F_m)/N.

    The average-score profile (1/M) sum_m 1[xbar_m > tau] estimates "the fraction of tasks on which an algorithm performs
    better than a certain score" (paper §4.2), i.e. its own target T(tau) = (1/M) sum_m 1[mu_m > tau] -- NOT the run-score
    tail F (recheck). Its bias against T is reported pointwise at taus at least ``min_dist`` from every
    task mean (T is a step function there) for N in ``avg_Ns``; the gap to F is a DIFFERENT ESTIMAND (it grows with N) and
    is reported separately as ``avg_gap_to_run_tail``. legacy=True reproduces the pre-fix criterion
    (|z of the gap to F| > 10), which any statistic with a different target would pass."""
    a_m, b_m = np.array([0.5, 1.0, 2.0, 5.0, 0.8]), np.array([0.5, 3.0, 2.0, 1.0, 4.0])
    M = len(a_m)
    taus = np.linspace(0.05, 0.95, 19)
    F_m = np.stack([st.beta.sf(taus, a, b) for a, b in zip(a_m, b_m)])     # (M, T)
    F = F_m.mean(0)
    var = (F_m * (1 - F_m)).sum(0) / (M ** 2 * N)
    x = jax.random.beta(key, jnp.asarray(a_m), jnp.asarray(b_m), (reps, N, M)).astype(F64)
    prof = np.asarray(jax.vmap(lambda s: A.score_profile(s, taus))(x))
    avg = np.asarray(jax.vmap(lambda s: A.avg_score_profile(s, taus))(x))
    z_mean = (prof.mean(0) - F) / np.sqrt(var / reps)
    var_ratio = prof.var(0) / var
    z_avg = (avg.mean(0) - F) / np.sqrt(avg.var(0) / reps + 1e-300)
    run_ok = bool(np.abs(z_mean).max() <= 4.5 and np.all(np.abs(var_ratio - 1) < 0.06))
    legacy_pass = bool(run_ok and np.abs(z_avg).max() > 10)
    out = dict(taus=taus.tolist(), truth=F.tolist(), run_profile_mean=prof.mean(0).tolist(),
               max_abs_z_run=float(np.abs(z_mean).max()), var_ratio_range=[float(var_ratio.min()), float(var_ratio.max())],
               max_abs_z_avg=float(np.abs(z_avg).max()), avg_bias_max=float(np.abs(avg.mean(0) - F).max()),
               legacy_pass=legacy_pass)
    if legacy:
        out["pass_"] = legacy_pass
        return out
    mu = a_m / (a_m + b_m)
    target = (mu[:, None] > taus[None, :]).mean(0)
    away = np.min(np.abs(taus[None, :] - mu[:, None]), 0) >= min_dist
    rows = []
    for i, n in enumerate(avg_Ns):
        xn = jax.random.beta(jax.random.fold_in(key, 100 + i), jnp.asarray(a_m), jnp.asarray(b_m), (reps, n, M)).astype(F64)
        av = np.asarray(jax.vmap(lambda s: A.avg_score_profile(s, taus))(xn))
        b = av.mean(0) - target
        se = np.sqrt(av.var(0) / reps)
        rows.append(dict(N=int(n), max_abs_bias_vs_own_target=float(np.abs(b[away]).max()),
                         bias_vs_own_target=b.tolist(), se=se.tolist(),
                         avg_gap_to_run_tail=float(np.abs(av.mean(0) - F).max())))
    out.update(avg_target=target.tolist(), taus_away_from_task_means=away.tolist(), avg_rows=rows,
               note="avg_bias_max / max_abs_z_avg are the gap to the run-score tail F: a different estimand")
    out["pass_"] = bool(run_ok and rows[-1]["max_abs_bias_vs_own_target"] < rows[0]["max_abs_bias_vs_own_target"])
    return out


def toys_area_identities(key, n_cases: int = 20) -> dict:
    """Fig 8 / D2.3 / D2.4: for x >= 0, mean = int_0^inf F_hat; OG = int_0^gamma (1 - F_hat); IQM_c =
    2 int_{.25}^{.75} Q_hat(u) du (exact piecewise integration of the empirical step functions)."""
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 31 - 1)))
    err = 0.0
    for c in range(n_cases):
        x = rng.gamma(1.5, 0.5, (int(rng.integers(2, 10)), int(rng.integers(2, 20))))
        s = np.sort(x.ravel())
        K = len(s)
        knots = np.concatenate([[0.0], s])
        F_right = 1.0 - np.arange(K) / K          # F_hat on (knots[i], knots[i+1]) = (K - i)/K
        area_mean = np.sum(np.diff(knots) * F_right)
        kg = np.clip(np.concatenate([knots, [np.inf]]), 0, 1.0)            # gamma = 1; last piece (s_K, inf): F = 0
        area_og = np.sum(np.diff(kg) * (1 - np.concatenate([F_right, [0.0]])))
        u = np.linspace(0, 1, K + 1)
        w = np.clip(np.minimum(u[1:], 0.75) - np.maximum(u[:-1], 0.25), 0, None)
        iqm_c_area = 2 * np.sum(w * s)
        err = max(err, abs(area_mean - float(A.agg_mean(jnp.asarray(x)))), abs(area_og - float(A.agg_og(jnp.asarray(x)))),
                  abs(iqm_c_area - float(A.agg_iqm(jnp.asarray(x), "c"))))
    return dict(max_abs_err=err, pass_=bool(err <= 1e-12))


def predicted_mean_coverage(N: int, M: int, level: float = 0.95, R: int | None = None) -> float:
    """D10: percentile CI of MEAN on an equal-variance Gaussian design covers P(|T_nu| < z sqrt((N-1)/N)),
    nu = M (N - 1) (plug-in variance biased by (N-1)/N; sample mean independent of sample variance).

    R (recheck): when the N runs per task are drawn WITHOUT replacement from a finite pool of R runs and
    the truth is the pool statistic (paper Fig 6 procedure), the estimator's variance carries the finite-population
    factor (R-N)/R while the plug-in bootstrap variance does not, so the factor becomes ((N-1)/N) * R/(R-N).
    R=None reproduces the pre-fix (infinite-population) prediction."""
    z = st.norm.ppf(0.5 + level / 2)
    nu = M * (N - 1)
    f = (N - 1) / N
    if R is not None:
        if not N < R:
            raise ValueError("finite-pool prediction needs N < R")
        f *= R / (R - N)
    return float(2 * st.t.cdf(z * math.sqrt(f), nu) - 1)


def fpc_target_coverage(K: int, R: int, level: float = 0.95) -> float:
    """recheck : the coverage that an interval which is EXACT for i.i.d. sampling (width z * S / sqrt(K))
    attains in the without-replacement Fig 6 procedure (truth = statistic of the R-run pool): P(|Z| < z sqrt(R/(R-K))).
    This, not the nominal level, is the honest reference for that procedure (0.95 only as R -> infinity)."""
    z = st.norm.ppf(0.5 + level / 2)
    return float(2 * st.norm.cdf(z * math.sqrt(R / (R - K))) - 1)


def toys_coverage(key, Ns=(3, 10), M: int = 26, sets: int = 2000, inner_B: int = 2000) -> dict:
    """D10 coverage prediction for the percentile CI of MEAN (fresh Gaussian experiments, truth = mu known)."""
    rows = []
    mu = np.linspace(0.0, 1.5, M)
    for N in Ns:
        kN = jax.random.fold_in(key, N)

        @jax.jit
        def one(k):
            x = jnp.asarray(mu)[None, :] + 0.4 * jax.random.normal(jax.random.fold_in(k, 0), (N, M), F64)
            bk = jax.random.split(jax.random.fold_in(k, 1), inner_B)
            d = jax.vmap(lambda b: A.agg_mean(A.resample(b, x)))(bk)
            ci = A.ci_from_dist(d[:, None], A.agg_mean(x)[None], "percentile")[:, 0]
            return (ci[0] <= mu.mean()) & (mu.mean() <= ci[1])

        cov = np.asarray(jax.lax.map(one, jax.random.split(kN, sets), batch_size=100))
        c = float(cov.mean())
        se = math.sqrt(c * (1 - c) / sets)
        pred = predicted_mean_coverage(N, M)
        rows.append(dict(N=N, M=M, sets=sets, inner_B=inner_B, coverage=c, se=se, predicted=pred,
                         z=(c - pred) / se, ok=bool(abs(c - pred) <= 4 * se + 0.01)))
    return dict(rows=rows, pass_=all(r["ok"] for r in rows))


def mixture_iqm_estimand(mu, sigma) -> float:
    """IQM* = 2 int_{.25}^{.75} Q_X(u) du for the equal-weight Gaussian mixture (D1/D10), by quadrature."""
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    cdf = lambda t: np.mean(st.norm.cdf((t - mu) / sigma))  # noqa: E731
    lo, hi = mu.min() - 10 * sigma.max(), mu.max() + 10 * sigma.max()
    Q = lambda u: optimize.brentq(lambda t: cdf(t) - u, lo, hi, xtol=1e-14)  # noqa: E731
    val, _ = integrate.quad(Q, 0.25, 0.75, epsabs=1e-12, epsrel=1e-12, limit=200)
    return 2 * val


def toys_iqm_estimand(key, N: int = 100, M: int = 26, reps: int = 4000) -> dict:
    """D10: E[IQM_a] on K = N M draws of the Gaussian design vs the mixture's IQM* (bias O(1/K))."""
    mu = np.linspace(0.0, 1.5, M)
    sig = np.linspace(0.1, 0.6, M)
    truth = mixture_iqm_estimand(mu, sig)
    x = jnp.asarray(mu)[None, None, :] + jnp.asarray(sig)[None, None, :] * jax.random.normal(key, (reps, N, M), F64)
    v = np.asarray(jax.vmap(A.agg_iqm)(x))
    se = v.std(ddof=1) / math.sqrt(reps)
    return dict(truth=truth, mean=float(v.mean()), se=float(se), z=float((v.mean() - truth) / se),
                pass_=bool(abs(v.mean() - truth) <= 4 * se + 1.0 / (N * M)))


def toys_lift_identity(key) -> dict:
    """D5.2: (1 + L_l) = (1 + l)(1 + L_0) replicate-by-replicate for positively homogeneous stats (median, IQM)."""
    pool = jax.random.gamma(key, 2.0, (100, 26)).astype(F64) * 0.2
    ells = jnp.asarray([0.0, 0.1, 0.25])
    err = 0.0
    for fn in (A.agg_median, A.agg_iqm):
        L = np.asarray(A.lift_distribution(jax.random.fold_in(key, 1), pool, ells, fn, 5, 2000))
        err = max(err, float(np.abs((1 + L[:, 1:]) - (1 + np.asarray(ells[1:]))[None] * (1 + L[:, :1])).max()))
    return dict(max_abs_err=err, pass_=bool(err <= 1e-12))


# ------------------------------------------------------------------------------------------------ analyses
def aggregate_table(key, scores: dict, B: int, method: str = "percentile", task_bootstrap: bool = False) -> dict:
    """{alg: {stat: [point, lo, hi]}} for [median, IQM, mean, OG] with one shared stratified bootstrap per algorithm
    (Fig 7 / 9 / A.29). Algorithms get independent sub-keys fold_in(key, sorted index)."""
    out = {}
    for j, alg in enumerate(sorted(scores)):
        b, ci, _ = A.interval_estimate(jax.random.fold_in(key, j), (jnp.asarray(scores[alg]),), A.aggregates4, B,
                                       method=method, task_bootstrap=task_bootstrap)
        out[alg] = {s: [float(b[i]), float(ci[0, i]), float(ci[1, i])] for i, s in enumerate(A.AGG_NAMES)}
        out[alg]["n_runs"], out[alg]["n_tasks"] = int(scores[alg].shape[0]), int(scores[alg].shape[1])
    return out


def poi_table(key, scores: dict, pairs, B: int) -> dict:
    out = {}
    for j, (x, y) in enumerate(pairs):
        b, ci, _ = A.interval_estimate(jax.random.fold_in(key, j), (jnp.asarray(scores[x]), jnp.asarray(scores[y])), A.poi, B)
        out[f"{x} > {y}"] = [float(b[0]), float(ci[0, 0]), float(ci[1, 0])]
    return out


def toys_coverage_procedure(key, K: int = 3, M: int = 26, pool_runs: int = 4000, sets: int = 2000,
                             inner_B: int = 1000) -> dict:
    """Validates algorithm.coverage (the Fig 6 procedure) itself: on a large equal-variance Gaussian pool, K runs drawn
    without replacement are nearly i.i.d., so percentile coverage of MEAN must match D10's prediction
    P(|T_nu| < z sqrt((K-1)/K)) (finite-population correction K/pool_runs < 0.1 %)."""
    mu = jnp.linspace(0.0, 1.5, M)
    pool = mu[None, :] + 0.4 * jax.random.normal(jax.random.fold_in(key, 0), (pool_runs, M), F64)
    res, truth, _ = A.coverage(jax.random.fold_in(key, 1), pool, A.agg_mean, K, sets, inner_B, methods=("percentile",))
    c, se = res["percentile"]["coverage"], res["percentile"]["se"]
    pred = predicted_mean_coverage(K, M)
    return dict(K=K, M=M, pool_runs=pool_runs, sets=sets, coverage=c, se=se, predicted=pred, z=(c - pred) / se,
                pass_=bool(abs(c - pred) <= 4 * se + 0.01))
