"""The paper's estimators and interval procedures as pure JAX functions (float64), vmapped over bootstrap replicates.

Reference: Agarwal et al., "Deep RL at the Edge of the Statistical Precipice" (NeurIPS 2021); derivation in
the folder README; D-numbers and R-n items below are labels of hand derivations / reconciliation notes that are not shipped

Layout (R5): a score matrix is ``x[n, m]`` = normalised score of run n on task m, shape (N runs, M tasks), as rliable.
A trailing third axis (N, M, T) is allowed where stated (evaluation points over training, R8).

Arms of the comparison (the paper's "proposed" vs "baseline" aggregates, Table 1): MEAN and MEDIAN of task means
(baselines, §2) vs IQM, optimality gap and P(X > Y) (proposed, §4.3); average-score profile (baseline) vs run-score
profile (proposed, §4.2); CI constructions percentile (recommended) vs basic / bc / bca (A.5, Fig 6). Every arm is one
function below and they share the same resampled data inside a bootstrap, so they differ ONLY in the statistic.

PRNG the README: callers pass a key built from ``branch_key(master_seed, exp, alg, stat)``; a bootstrap of B
replicates uses ``split(key, B)``; replicate b resamples argument i of a multi-sample statistic with
``fold_in(key_b, i)`` and, inside ``resample``, draws all (N, M) run indices with one ``randint`` from
``fold_in(fold_in(key_b, i), 0)`` (the task-bootstrap task indices use ``fold_in(fold_in(key_b, i), 1000)``).

float64 is mandatory: importing this module enables ``jax_enable_x64``. Do NOT import it into a process
that trains the vendored PPO (src/ppo8 is float32 and must stay byte-for-byte the ppo_implementation_details reproduction's behaviour).
"""
from __future__ import annotations

import functools
import math
from functools import partial
from typing import Callable, Sequence

import jax

jax.config.update("jax_enable_x64", True)  #  (sources.yaml estimators.dtype)

import jax.numpy as jnp  # noqa: E402
from jax.scipy.special import ndtr, ndtri  # noqa: E402

F64 = jnp.float64

# ------------------------------------------------------------------------------------------------ key tree
EXP_IDS = {"e1": 1, "e2": 2, "e3": 3, "e4": 4, "e5": 5, "e6": 6, "toys": 7, "small": 8, "tests": 9}


def branch_key(master_seed: int, exp: int, alg: int = 0, stat: int = 0) -> jax.Array:
    """root = PRNGKey(master_seed) -> fold_in(EXP_ID) -> fold_in(ALG_ID) -> fold_in(STAT_ID or N)  the README."""
    k = jax.random.PRNGKey(master_seed)
    return jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(k, exp), alg), stat)


# ------------------------------------------------------------------------------------------------ point estimates
def task_means(x):
    """Per-task mean over runs, summing the runs in index order then dividing by N.

    Numerics choice (not a statistical one): this is the order numpy uses for an axis-0 reduction of a C-ordered
    (N, M) array, so task means are bit-identical to rliable's. XLA's tree reduction differs by 1 ulp, which flips
    1[mean_m > tau] when a task mean equals tau exactly (tied scores) and splits bootstrap atoms of the median in two
    (found by the E0 differential test, the READMEsmall-scale run). The division uses exact_div for the same reason."""
    s = x[0]
    for n in range(1, x.shape[0]):
        s = s + x[n]
    return exact_div(s, x.shape[0])


def exact_div(a, n):
    """a / n correctly rounded on every backend: XLA:CPU rewrites division by a constant into multiplication by its
    reciprocal (1-ulp errors in ~1/3 of results); the optimization barrier hides the constant (no-op on GPU).

    Under vmap the barrier's batching rule leaves the constant divisor unbatched outside it and XLA:CPU folds the
    division again (found in the official-code check: 9,142 of 27,500 bootstrap task means 1 ulp off rliable on CPU, which moved bc /
    bca endpoints by one atom through the strict mean(dist < base)). The custom_vmap rule below applies the barrier to
    the full batched array instead, so the result is correctly rounded inside vmap / lax.map / jit as well."""
    return _exact_div_n(int(n))(a)


@functools.lru_cache(maxsize=None)
def _exact_div_n(n: int):
    @jax.custom_batching.custom_vmap
    def div(a):
        return jax.lax.div(a, jax.lax.optimization_barrier(jnp.full_like(a, n)))

    @div.def_vmap
    def _rule(axis_size, in_batched, a):
        return div(a), in_batched[0]

    return div


def agg_mean(x):
    """Mean of task means (D2.1, §2) = rliable aggregate_mean."""
    return jnp.mean(task_means(x), axis=0)


def agg_median(x):
    """Median of task means (D2.2); even M -> midpoint of the two middle task means (R2)."""
    return jnp.median(task_means(x), axis=0)


def _sorted_flat(x):
    return jnp.sort(jnp.ravel(x))


def trimmed_mean(x, alpha: float):
    """alpha-trimmed mean of ALL pooled scores, integer cut g = floor(alpha*K) at each end (scipy trim_mean, R1)."""
    s = _sorted_flat(x)
    K = s.shape[0]
    g = int(math.floor(alpha * K))
    if g >= K - g:
        raise ValueError(f"trim {alpha} leaves no data for K={K}")
    return jnp.mean(s[g:K - g])


def agg_iqm(x, mode: str = "a"):
    """Interquartile mean of pooled runs (§4.3, D2.3).

    mode 'a' (default, official R1): drop floor(K/4) at each end (= scipy trim_mean(.., 0.25, axis=None)).
    mode 'b': keep exactly floor(K/2) order statistics, dropping floor((K - floor(K/2))/2) from the bottom.
    mode 'c': fractional trim of exactly 25 % mass at each end (plug-in of 2*int_{.25}^{.75} Q(u) du).
    """
    s = _sorted_flat(x)
    K = s.shape[0]
    if mode == "a":
        g = K // 4
        return jnp.mean(s[g:K - g])
    if mode == "b":
        k = K // 2
        lo = (K - k) // 2
        return jnp.mean(s[lo:lo + k])
    if mode == "c":
        i = jnp.arange(K, dtype=F64)  # order statistic i+1 covers [i, i+1]
        w = jnp.clip(jnp.minimum(i + 1.0, 0.75 * K) - jnp.maximum(i, 0.25 * K), 0.0, 1.0)
        return jnp.sum(w * s) / (0.5 * K)
    raise ValueError(mode)


def agg_og(x, gamma: float = 1.0):
    """Optimality gap gamma - mean(min(x, gamma)) over all runs (D2.4, R3)."""
    return gamma - jnp.mean(jnp.minimum(x, gamma))


def difficulty_progress(x, frac: float = 0.25):
    """DP-frac: mean of the bottom int(frac*K) pooled scores (A.7, R18)."""
    s = _sorted_flat(x)
    k = int(frac * s.shape[0])
    return jnp.mean(s[:k])


def superhuman_prob(x, thr: float = 1.0):
    """Fraction of pooled runs with score strictly > thr (A.7, R18)."""
    return jnp.mean((x > thr).astype(F64))


def aggregates4(x):
    """[median, IQM, mean, OG] -- the Colab's aggregate vector (Fig 7, 9, A.29); one shared bootstrap."""
    return jnp.stack([agg_median(x), agg_iqm(x), agg_mean(x), agg_og(x)])


AGG_NAMES = ("median", "iqm", "mean", "og")


# ------------------------------------------------------------------------------------------------ profiles
def score_profile(x, taus):
    """Run-score distribution F_X(tau) = mean over all runs and tasks of 1[x > tau] (Eq 1, D3.1)."""
    taus = jnp.asarray(taus, F64)
    flat = jnp.ravel(x)
    return jnp.mean((flat[:, None] > taus[None, :]).astype(F64), axis=0)


def avg_score_profile(x, taus):
    """Average-score distribution: fraction of TASK MEANS > tau (baseline, D3.2)."""
    taus = jnp.asarray(taus, F64)
    tm = task_means(x)
    return jnp.mean((tm[:, None] > taus[None, :]).astype(F64), axis=0)


# ------------------------------------------------------------------------------------------------ P(X > Y)
def poi_per_task(x, y):
    """P_hat(X_m > Y_m) = U_m / (N_x N_y), ties 1/2 (Eq A.2 normalised, D2.5, R4). x (Nx, M), y (Ny, M) -> (M)."""
    gt = (x[:, None, :] > y[None, :, :]).astype(F64)
    eq = (x[:, None, :] == y[None, :, :]).astype(F64)
    return jnp.mean(gt + 0.5 * eq, axis=(0, 1))


def poi(x, y):
    """Average probability of improvement (1/M) sum_m P_hat(X_m > Y_m) (§4.3)."""
    return jnp.mean(poi_per_task(x, y))


# ------------------------------------------------------------------------------------------------ resampling
def stratified_indices(key, shape):
    """Run indices for one stratified replicate: i.i.d. uniform over N runs, independently for every task (and
    every trailing index, R8). shape = x.shape."""
    return jax.random.randint(key, shape, 0, shape[0])


def gather_runs(x, idx, task_idx=None):
    """x*[n, m, ...] = x[idx[n, m, ...], m, ...]  (or task task_idx[m] instead of m under the task bootstrap)."""
    cols = jnp.arange(x.shape[1]) if task_idx is None else task_idx
    if x.ndim == 2:
        return x[idx, cols[None, :]]
    if x.ndim == 3:
        t = jnp.arange(x.shape[2])
        return x[idx, cols[None, :, None], t[None, None, :]]
    raise ValueError("scores must be (N, M) or (N, M, T)")


def resample(key, x, task_bootstrap: bool = False):
    """One stratified-bootstrap replicate of x (D4.1); with task_bootstrap also resample tasks (D4.4, R6)."""
    k_run, k_task = jax.random.fold_in(key, 0), jax.random.fold_in(key, 1000)
    idx = stratified_indices(k_run, x.shape)
    tidx = jax.random.randint(k_task, (x.shape[1],), 0, x.shape[1]) if task_bootstrap else None
    return gather_runs(x, idx, tidx)


def resample_many(key, xs: Sequence, task_bootstrap: bool = False):
    """Independent stratified resampling of each array ('independent sampling'): arg i uses fold_in(key, i).

    task_bootstrap is single-sample only (rliable's StratifiedBootstrap(task_bootstrap=True) takes one array): with two
    or more arrays each would get its OWN task index vector, so a two-sample statistic (P(X>Y), a difference) would
    compare different tasks between X and Y (recheck ). Refused rather than silently wrong."""
    if task_bootstrap and len(xs) > 1:
        raise ValueError("task_bootstrap=True is single-sample only (independent task draws per array would mismatch tasks)")
    return tuple(resample(jax.random.fold_in(key, i), x, task_bootstrap) for i, x in enumerate(xs))


def _as_vec(v):
    return jnp.atleast_1d(jnp.asarray(v, F64))


@partial(jax.jit, static_argnames=("stat", "B", "task_bootstrap", "batch"))
def bootstrap_dist(key, xs: tuple, stat: Callable, B: int, task_bootstrap: bool = False, batch: int = 1000):
    """Bootstrap distribution (B, K) of stat(*xs) under independent stratified resampling of every array in xs."""
    keys = jax.random.split(key, B)

    def one(k):
        return _as_vec(stat(*resample_many(k, xs, task_bootstrap)))

    return jax.lax.map(one, keys, batch_size=min(batch, B))


@partial(jax.jit, static_argnames=("stat", "batch"))
def bootstrap_dist_from_indices(xs: tuple, idxs: tuple, stat: Callable, batch: int = 1000):
    """Same as bootstrap_dist but with externally supplied run indices idxs[i] of shape (B) + xs[i].shape.
    Used to replay the oracle's numpy draws exactly (differential test with a shared RNG stream)."""
    B = idxs[0].shape[0]

    def one(ii):
        return _as_vec(stat(*(gather_runs(x, i) for x, i in zip(xs, ii))))

    return jax.lax.map(one, idxs, batch_size=min(batch, B))


def jackknife(xs: tuple, stat: Callable):
    """Leave-one-run-ROW-out jackknife (all tasks at once, every array, R9). Returns (N, K)."""
    N = xs[0].shape[0]
    if any(x.shape[0] != N for x in xs):
        raise ValueError("BCa needs equal run counts (arch raises the same)")
    keep = jnp.asarray([[j for j in range(N) if j != i] for i in range(N)])
    return jax.vmap(lambda kk: _as_vec(stat(*(x[kk] for x in xs))))(keep)


# ------------------------------------------------------------------------------------------------ intervals
def bca_acceleration(jk, pooled_mean: bool = False):
    """a = sum u^3 / (6 (sum u^2)^1.5), u = mean(jk) - jk (D4.3). pooled_mean=True reproduces arch's
    ``jk_params.mean`` over ALL outputs (R9 quirk; identical for a single output)."""
    mu = jnp.mean(jk) if pooled_mean else jnp.mean(jk, axis=0)
    u = mu - jk
    return jnp.sum(u ** 3, 0) / (6.0 * jnp.sum(u ** 2, 0) ** 1.5)


def ci_from_dist(dist, base, method: str = "percentile", level: float = 0.95, jk=None, pooled_jk_mean: bool = False):
    """Two-sided interval (2, K) from a bootstrap distribution dist (B, K) and the point estimate base (K) (D4.3, R9).

    percentile [q_{a/2}, q_{1-a/2}]; basic [2 base - q_hi, 2 base - q_lo]; bc / bca: z0 = Phi^-1(mean(dist < base))
    (strict <) and endpoints q at 100 Phi(z0 + (z0 + z_p)/(1 - a (z0 + z_p))). Quantiles: linear interpolation.
    bc/bca return NaN where mean(dist < base) is 0 or 1 (arch raises: 'undefined').
    """
    dist = jnp.asarray(dist, F64)
    base = _as_vec(base)
    alpha = (1.0 - level) / 2.0
    zq = ndtri(jnp.asarray([alpha, 1.0 - alpha], F64))
    if method in ("percentile", "basic"):
        pct = jnp.broadcast_to(jnp.asarray([alpha, 1.0 - alpha]) * 100.0, (base.shape[0], 2))
    elif method in ("bc", "bca"):
        p = jnp.mean((dist < base[None, :]).astype(F64), axis=0)
        z0 = ndtri(p)[:, None]
        a = 0.0
        if method == "bca":
            if jk is None:
                raise ValueError("bca needs jackknife values")
            a = bca_acceleration(jk, pooled_jk_mean)[:, None]
        pct = 100.0 * ndtr(z0 + (z0 + zq[None, :]) / (1.0 - a * (z0 + zq[None, :])))
        bad = (p <= 0.0) | (p >= 1.0)
        pct = jnp.where(bad[:, None], jnp.nan, pct)
    else:
        raise ValueError(method)
    lo = jax.vmap(lambda col, q: jnp.percentile(col, q[0], method="linear"))(dist.T, pct)
    hi = jax.vmap(lambda col, q: jnp.percentile(col, q[1], method="linear"))(dist.T, pct)
    if method == "basic":
        lo, hi = 2.0 * base - hi, 2.0 * base - lo
    return jnp.stack([lo, hi])


def interval_estimate(key, xs, stat: Callable, B: int, method: str = "percentile", level: float = 0.95,
                      task_bootstrap: bool = False, pooled_jk_mean: bool = False):
    """Point estimate + CI for stat(*xs), the rliable get_interval_estimates analogue (one shared bootstrap for a
    vector statistic). pooled_jk_mean (only matters for VECTOR stats under bca): False (default) = standard Efron BCa,
    one acceleration per output from that output's own jackknife mean (paper A.5); True reproduces arch's
    ``jk_params.mean`` over ALL outputs (R9 quirk) and is used ONLY by the oracle-parity code (e0_difftest,
    official_compare, tests). the pooled form mixes e.g. the median's jackknife with the OG's."""
    xs = tuple(jnp.asarray(x, F64) for x in (xs if isinstance(xs, (tuple, list)) else (xs,)))
    base = _as_vec(stat(*xs))
    dist = bootstrap_dist(key, xs, stat, B, task_bootstrap)
    jk = jackknife(xs, stat) if method == "bca" else None
    return base, ci_from_dist(dist, base, method, level, jk, pooled_jk_mean), dist


# ------------------------------------------------------------------------------------------------ ranks
@partial(jax.jit, static_argnames=("n",))
def rank_matrix(key, xs: tuple, n: int):
    """Rank distribution (Fig 11c, A.31, R16): per task and draw, one run per algorithm sampled with replacement,
    algorithms ranked by that single score (rank 0 = highest), ties broken uniformly at random.
    Returns (M, A, A): [task, algorithm, rank] = fraction of the n draws.

    This is the OFFICIAL estimand (Colab cell 47, ``get_rank_matrix``: ``subsample_scores_mat(score, num_samples=n,
    replace=True)`` draws n single runs per task, then ``np.lexsort`` ranks those single scores). The Fig 11 caption's
    "200,000 stratified bootstrap re-samples" also admits a per-task-MEAN-of-a-resample reading; that variant is
    ``rank_matrix_resample_means`` (labelled sensitivity). The two estimands differ."""
    A = len(xs)
    M = xs[0].shape[1]
    draws = []
    for a, x in enumerate(xs):
        idx = jax.random.randint(jax.random.fold_in(key, a), (n, M), 0, x.shape[0])
        draws.append(x[idx, jnp.arange(M)[None, :]])
    s = jnp.stack(draws, 0)                                    # (A, n, M)
    r = jax.random.uniform(jax.random.fold_in(key, 10_000), s.shape)
    better = (s[None, :] > s[:, None]).astype(jnp.int32)       # [a, b]: s_b > s_a
    tie_first = ((s[None, :] == s[:, None]) & (r[None, :] < r[:, None])).astype(jnp.int32)
    rank = jnp.sum(better + tie_first, axis=1)                 # (A, n, M)
    onehot = jax.nn.one_hot(rank, A, dtype=F64)                # (A, n, M, A)
    return jnp.transpose(jnp.mean(onehot, axis=1), (1, 0, 2))  # (M, A, A)


@partial(jax.jit, static_argnames=("n",))
def rank_matrix_resample_means(key, xs: tuple, n: int):
    """SENSITIVITY variant of rank_matrix (NOT the official estimand): per draw, a stratified bootstrap
    resample of each algorithm's N runs per task, algorithms ranked by the per-task MEAN of their resample (rank 0 =
    highest), ties broken uniformly at random. Returns (M, A, A) like rank_matrix."""
    A = len(xs)
    means = []
    for a, x in enumerate(xs):
        idx = jax.random.randint(jax.random.fold_in(key, a), (n,) + x.shape, 0, x.shape[0])   # (n, N, M)
        means.append(jnp.mean(x[idx, jnp.arange(x.shape[1])[None, None, :]], axis=1))        # (n, M)
    s = jnp.stack(means, 0)                                    # (A, n, M)
    r = jax.random.uniform(jax.random.fold_in(key, 10_000), s.shape)
    better = (s[None, :] > s[:, None]).astype(jnp.int32)
    tie_first = ((s[None, :] == s[:, None]) & (r[None, :] < r[:, None])).astype(jnp.int32)
    rank = jnp.sum(better + tie_first, axis=1)
    onehot = jax.nn.one_hot(rank, A, dtype=F64)
    return jnp.transpose(jnp.mean(onehot, axis=1), (1, 0, 2))


# ------------------------------------------------------------------------------------------------ studentized (DMC)
def studentized_ci_official(task_means_, task_sds, n: int, level: float = 0.95, scale: float = 1000.0):
    """Fig 11a rows from reported mean +- sd (official Colab form, R14): mean of task means, pooled sd =
    sqrt(mean_m s_m^2), SE = sd / sqrt(n) (no 1/sqrt(M)), t_{n-1}. Returns (mean, [lo, hi]) divided by scale."""
    from scipy import stats as st
    m = float(jnp.mean(jnp.asarray(task_means_, F64)))
    sd = float(jnp.sqrt(jnp.mean(jnp.asarray(task_sds, F64) ** 2)))
    h = sd / math.sqrt(n) * st.t.ppf((1 + level) / 2.0, n - 1)
    return m / scale, [(m - h) / scale, (m + h) / scale]


def studentized_ci_d6(task_means_, task_sds, n: int, level: float = 0.95, scale: float = 1000.0):
    """the README (our correction): SE = sqrt(sum_m s_m^2 / n) / M, Welch-Satterthwaite df."""
    from scipy import stats as st
    mu = jnp.asarray(task_means_, F64)
    s2 = jnp.asarray(task_sds, F64) ** 2
    M = mu.shape[0]
    m = float(jnp.mean(mu))
    se = float(jnp.sqrt(jnp.sum(s2 / n)) / M)
    df = float(jnp.sum(s2 / n) ** 2 / jnp.sum((s2 / n) ** 2 / (n - 1)))
    h = se * st.t.ppf((1 + level) / 2.0, df)
    return m / scale, [(m - h) / scale, (m + h) / scale], df


# ------------------------------------------------------------------------------------------------ few-run procedures
def subsample(key, pool, n: int, replace: bool = True):
    """Per-task subsample of n runs from a pool (R, M): with replacement (default, paper §3) or without."""
    R, M = pool.shape[:2]
    if replace:
        idx = jax.random.randint(key, (n, M), 0, R)
    else:
        u = jax.random.uniform(key, (R, M))
        idx = jnp.argsort(u, axis=0)[:n]
    return pool[idx, jnp.arange(M)[None, :]]


def subsample_rows(key, pool, n: int, replace: bool = False):
    """Whole-row subsample (same run index on every task; Colab batch_subsample_seeds, R17 sensitivity)."""
    R = pool.shape[0]
    if replace:
        rows = jax.random.randint(key, (n,), 0, R)
    else:
        rows = jax.random.permutation(key, R)[:n]
    return pool[rows]


@partial(jax.jit, static_argnames=("stat", "n", "reps", "replace", "batch"))
def sampling_distribution(key, pool, stat: Callable, n: int, reps: int, replace: bool = True, batch: int = 2000):
    """Distribution of stat on reps independent n-run subsamples of the pool (Fig 2 left, Fig 3, A.17; D5.1)."""
    keys = jax.random.split(key, reps)
    return jax.lax.map(lambda k: _as_vec(stat(subsample(k, pool, n, replace))), keys, batch_size=min(batch, reps))


@partial(jax.jit, static_argnames=("stat", "n", "reps", "batch"))
def lift_distribution(key, pool, ells, stat: Callable, n: int, reps: int, batch: int = 2000):
    """Observed lift stat((1+l) B) / stat(A) - 1 for two independent n-subsamples A, B (Fig 4, D5.2).
    Returns (reps, len(ells)); the same subsample pair is used for every l, so the D5.2 identity holds exactly."""
    ells = jnp.asarray(ells, F64)
    keys = jax.random.split(key, reps)

    def one(k):
        a = stat(subsample(jax.random.fold_in(k, 0), pool, n))
        sb = subsample(jax.random.fold_in(k, 1), pool, n)
        # explicit scaling of every score of B (not the homogeneity shortcut), so D5.2's identity is a test
        return jax.vmap(lambda l: stat((1.0 + l) * sb))(ells) / a - 1.0

    return jax.lax.map(one, keys, batch_size=min(batch, reps))


@partial(jax.jit, static_argnames=("stat", "n", "reps", "batch"))
def diff_distribution(key, pool_x, pool_y, stat: Callable, n: int, reps: int, batch: int = 2000):
    """stat(X n-subsample) - stat(Y n-subsample), independent subsamples (Fig A.15, D5.3)."""
    keys = jax.random.split(key, reps)

    def one(k):
        return stat(subsample(jax.random.fold_in(k, 0), pool_x, n)) - stat(subsample(jax.random.fold_in(k, 1), pool_y, n))

    return jax.lax.map(one, keys, batch_size=min(batch, reps))


def coverage(key, pool, stat: Callable, k: int, sets: int, inner_B: int, methods=("percentile", "basic", "bc", "bca"),
             level: float = 0.95, truth=None, set_batch: int = 50):
    """True coverage of stratified-bootstrap CIs built on k runs drawn WITHOUT replacement from the pool (Fig 6, D5.5).

    truth defaults to stat(pool). Returns {method: dict(coverage, se, coverage_conditional, se_conditional,
    mean_width, undefined, sets)} plus the per-set arrays.

    ``coverage`` is UNCONDITIONAL, over all ``sets`` (the paper's Fig 6 / A.19 denominator): a set whose bc / bca
    interval is undefined (mean(dist < base) in {0, 1}) counts as NOT covering.
    ``coverage_conditional`` is over the defined sets only (the earlier behaviour); mean_width is over defined sets.
    """
    pool = jnp.asarray(pool, F64)
    truth = float(_as_vec(stat(pool))[0]) if truth is None else float(truth)
    keys = jax.random.split(key, sets)

    @jax.jit
    def one(kk):
        sub = subsample(jax.random.fold_in(kk, 0), pool, k, replace=False)
        base = _as_vec(stat(sub))
        bkeys = jax.random.split(jax.random.fold_in(kk, 1), inner_B)
        dist = jax.vmap(lambda b: _as_vec(stat(resample(b, sub))))(bkeys)
        jk = jackknife((sub,), stat) if "bca" in methods else None
        out = [ci_from_dist(dist, base, m, level, jk, pooled_jk_mean=True)[:, 0] for m in methods]
        return jnp.stack(out)  # (n_methods, 2)

    cis = jax.lax.map(one, keys, batch_size=min(set_batch, sets))  # (sets, n_methods, 2)
    res = {}
    for j, m in enumerate(methods):
        lo, hi = cis[:, j, 0], cis[:, j, 1]
        ok = jnp.isfinite(lo) & jnp.isfinite(hi)
        cov = jnp.where(ok, (lo <= truth) & (truth <= hi), False)
        n_ok = int(ok.sum())
        cc = float(cov.sum() / max(n_ok, 1))
        cu = float(cov.sum() / sets)
        res[m] = dict(coverage=cu, se=math.sqrt(cu * (1 - cu) / sets),
                      coverage_conditional=cc, se_conditional=math.sqrt(cc * (1 - cc) / max(n_ok, 1)),
                      mean_width=float(jnp.where(ok, hi - lo, 0.0).sum() / max(n_ok, 1)),
                      undefined=int(sets - n_ok), sets=sets)
    return res, truth, cis


def _trimmed_or_median(x, a: float):
    return jnp.median(jnp.ravel(x)) if a >= 0.5 else trimmed_mean(x, a)


@partial(jax.jit, static_argnames=("alphas", "n", "reps", "batch"))
def trimmed_mse(key, pool, alphas: tuple, n: int, reps: int, batch: int = 2000):
    """Fig A.16 (D5.4): MSE of the alpha-trimmed mean of n-run subsamples (with replacement) against the
    same alpha-trimmed mean of the whole pool; alpha = 0.5 means the pooled median. Returns (mse, bias, var)."""
    keys = jax.random.split(key, reps)
    v = jax.lax.map(lambda k: jnp.stack([_trimmed_or_median(subsample(k, pool, n), a) for a in alphas]), keys,
                    batch_size=min(batch, reps))
    truth = jnp.stack([_trimmed_or_median(pool, a) for a in alphas])
    bias = jnp.mean(v, 0) - truth
    var = jnp.var(v, 0)
    return jnp.mean((v - truth) ** 2, 0), bias, var
