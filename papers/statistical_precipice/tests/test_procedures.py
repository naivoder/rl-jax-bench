"""Case-study procedures that the gate scripts use but test_smoke / test_grads did not exercise:
the Fig 6 coverage procedure (vs the D10 prediction, reduced size) and its undefined-interval accounting,
rank_matrix direction / ties and the sensitivity estimand, trimmed_mse and diff_distribution against numpy /
scipy references on the same draws, and subsample(replace=False) / subsample_rows. CPU, a few seconds."""
import numpy as np
from scipy import stats as st


def _A():
    from src import algorithm as A
    return A


def test_coverage_matches_d10_prediction_reduced():
    import jax
    import jax.numpy as jnp
    from src import eval as E
    A = _A()
    k = A.branch_key(0, A.EXP_IDS["tests"], 30)
    M, K, sets = 26, 3, 600
    mu = jnp.linspace(0.0, 1.5, M)
    pool = mu[None, :] + 0.4 * jax.random.normal(k, (4000, M), jnp.float64)   # large pool: K draws ~ i.i.d.
    res, truth, cis = A.coverage(jax.random.fold_in(k, 1), pool, A.agg_mean, K, sets, 400, methods=("percentile",))
    pred = E.predicted_mean_coverage(K, M)
    c, se = res["percentile"]["coverage"], res["percentile"]["se"]
    assert cis.shape == (sets, 1, 2) and res["percentile"]["undefined"] == 0
    assert abs(c - pred) <= 4 * se + 0.01, (c, pred, se)


def test_coverage_counts_undefined_bc_as_not_covering():
    import jax.numpy as jnp
    A = _A()
    k = A.branch_key(0, A.EXP_IDS["tests"], 31)
    pool = jnp.tile(jnp.asarray([[0.25, 0.75]], jnp.float64), (50, 1))     # constant per task (dyadic: sums exact): dist == base
    res, truth, _ = A.coverage(k, pool, A.agg_mean, 3, 20, 50, methods=("percentile", "bc"))
    assert res["percentile"]["coverage"] == 1.0 and res["percentile"]["undefined"] == 0
    assert res["bc"]["undefined"] == 20 and res["bc"]["coverage"] == 0.0     # unconditional (paper denominator)
    assert res["bc"]["coverage_conditional"] == 0.0 and res["bc"]["sets"] == 20


def test_rank_matrix_direction_and_ties():
    import jax.numpy as jnp
    A = _A()
    k = A.branch_key(0, A.EXP_IDS["tests"], 32)
    n = 4000
    # task 0: alg2 > alg1 > alg0 strictly; task 1: alg0 == alg1 (tie) > alg2
    x0 = jnp.asarray([[1.0, 5.0]] * 3)
    x1 = jnp.asarray([[2.0, 5.0]] * 3)
    x2 = jnp.asarray([[3.0, 0.0]] * 3)
    for fn in (A.rank_matrix, A.rank_matrix_resample_means):
        r = np.asarray(fn(k, (x0, x1, x2), n))           # [task, alg, rank], rank 0 = highest
        assert r.shape == (2, 3, 3)
        np.testing.assert_allclose(r[0], [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        assert r[1, 2, 2] == 1.0
        tol = 4 * np.sqrt(0.25 / n)
        assert abs(r[1, 0, 0] - 0.5) < tol and abs(r[1, 1, 0] - 0.5) < tol
        np.testing.assert_allclose(r[1, 0, 0] + r[1, 1, 0], 1.0)


def test_rank_estimands_differ_official_single_run_vs_resample_mean():
    import jax.numpy as jnp
    A = _A()
    k = A.branch_key(0, A.EXP_IDS["tests"], 33)
    n = 20000
    x = jnp.asarray([[0.0], [10.0]])   # runs 0 and 10
    y = jnp.asarray([[4.0], [4.0]])
    single = float(A.rank_matrix(k, (x, y), n)[0, 0, 0])               # P(single run of x > 4) = 1/2
    means = float(A.rank_matrix_resample_means(k, (x, y), n)[0, 0, 0])  # P(mean of 2 resampled runs > 4) = 3/4
    tol = 4 * np.sqrt(0.25 / n)
    assert abs(single - 0.5) < tol and abs(means - 0.75) < tol


def test_trimmed_mse_matches_scipy_on_same_draws():
    import jax
    import jax.numpy as jnp
    A = _A()
    k = A.branch_key(0, A.EXP_IDS["tests"], 34)
    pool = jax.random.normal(k, (40, 3), jnp.float64) ** 3
    alphas, n, reps = (0.0, 0.1, 0.25, 0.5), 5, 200
    mse, bias, var = (np.asarray(v) for v in A.trimmed_mse(jax.random.fold_in(k, 1), pool, alphas, n, reps))
    keys = jax.random.split(jax.random.fold_in(k, 1), reps)
    subs = [np.asarray(A.subsample(kk, pool, n)).ravel() for kk in keys]
    ref = lambda v, a: np.median(v) if a >= 0.5 else st.trim_mean(v, a)  # noqa: E731
    truth = np.array([ref(np.asarray(pool).ravel(), a) for a in alphas])
    v = np.array([[ref(s, a) for a in alphas] for s in subs])
    np.testing.assert_allclose(mse, ((v - truth) ** 2).mean(0), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(bias, v.mean(0) - truth, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(var, v.var(0), rtol=1e-10, atol=1e-14)


def test_diff_distribution_matches_numpy_on_same_draws():
    import jax
    import jax.numpy as jnp
    A = _A()
    k = A.branch_key(0, A.EXP_IDS["tests"], 35)
    px = jax.random.uniform(k, (30, 4), jnp.float64)
    py = jax.random.uniform(jax.random.fold_in(k, 1), (30, 4), jnp.float64) + 0.1
    n, reps = 3, 100
    d = np.asarray(A.diff_distribution(jax.random.fold_in(k, 2), px, py, A.agg_mean, n, reps))
    ref = []
    for kk in jax.random.split(jax.random.fold_in(k, 2), reps):
        sx = np.asarray(A.subsample(jax.random.fold_in(kk, 0), px, n))
        sy = np.asarray(A.subsample(jax.random.fold_in(kk, 1), py, n))
        ref.append(sx.mean(0).mean() - sy.mean(0).mean())
    np.testing.assert_allclose(d, ref, rtol=1e-12, atol=1e-14)


def test_subsample_without_replacement_and_rows():
    import jax
    import jax.numpy as jnp
    A = _A()
    k = A.branch_key(0, A.EXP_IDS["tests"], 36)
    R, M = 12, 5
    pool = jnp.arange(R)[:, None] * 10.0 + jnp.arange(M)[None, :]   # value encodes (run, task)
    for i in range(20):
        s = np.asarray(A.subsample(jax.random.fold_in(k, i), pool, 7, replace=False))
        runs = (s // 10).astype(int)
        assert s.shape == (7, M) and np.all(s % 10 == np.arange(M)[None, :])      # task column preserved
        assert all(len(set(runs[:, m])) == 7 for m in range(M))                   # distinct within each task
        r = np.asarray(A.subsample_rows(jax.random.fold_in(k, 100 + i), pool, 7, replace=False))
        rr = (r // 10).astype(int)
        assert np.all(rr == rr[:, :1]) and len(set(rr[:, 0])) == 7               # whole rows, distinct
    # per-task independence: across draws, two tasks do not always get the same run set
    s = [np.asarray(A.subsample(jax.random.fold_in(k, 200 + i), pool, 3, replace=False)) // 10 for i in range(10)]
    assert any(not np.array_equal(np.sort(q[:, 0]), np.sort(q[:, 1])) for q in s)


def test_task_bootstrap_refuses_two_samples():
    """recheck : a two-sample task bootstrap would mismatch tasks between arrays."""
    import jax
    import jax.numpy as jnp
    import pytest
    A = _A()
    x = jnp.ones((3, 4))
    with pytest.raises(ValueError):
        A.resample_many(jax.random.PRNGKey(0), (x, x), task_bootstrap=True)
    A.resample_many(jax.random.PRNGKey(0), (x,), task_bootstrap=True)     # single sample still fine
    A.resample_many(jax.random.PRNGKey(0), (x, x), task_bootstrap=False)


def test_fpc_prediction_matches_finite_pool_coverage():
    """recheck : with K of R = 100 runs drawn without replacement and truth = pool mean, coverage follows
    the finite-population-corrected D10 prediction, not the infinite-population one (K = 20: 0.961 vs 0.936)."""
    import jax
    import jax.numpy as jnp
    from src import eval as E
    A = _A()
    k = A.branch_key(0, A.EXP_IDS["tests"], 31)
    R, M, K, sets = 100, 2, 20, 1500
    pool = jnp.asarray([0.2, 0.7])[None, :] + 0.1 * jax.random.normal(k, (R, M), jnp.float64)
    res, truth, cis = A.coverage(jax.random.fold_in(k, 1), pool, A.agg_mean, K, sets, 400, methods=("percentile",))
    c, se = res["percentile"]["coverage"], res["percentile"]["se"]
    fpc, inf = E.predicted_mean_coverage(K, M, R=R), E.predicted_mean_coverage(K, M)
    assert abs(c - fpc) <= 3.5 * se + 0.005, (c, fpc, se)
    assert c - inf > 0.01, (c, inf)
    assert abs(E.fpc_target_coverage(10, 100) - 0.9612) < 1e-3 and abs(E.fpc_target_coverage(10, 10 ** 9) - 0.95) < 1e-6
