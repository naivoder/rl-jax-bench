"""Independent checks of src/crossed_stats.py (numpy + scipy).

* max-T single-row p equals scipy's exact paired permutation test; adjusted p >= single p; one-row family: equal.
* seed_mean_t_ci equals scipy.stats.ttest_1samp's confidence interval on the seed means.
* clopper_pearson equals the Beta-quantile closed form; count_rule_oc equals a direct binomial sum.
* poi_per_task equals this repo's verified src.algorithm.poi (rliable-matched); run_score_profile equals
  src.algorithm.score_profile.
* anova2 recovers planted components on a large crossed Gaussian draw.
"""
import math
import os
import sys

import numpy as np
import pytest
from scipy import stats as st

from src import crossed_stats as X  # noqa: E402


def test_maxt_vs_scipy_permutation():
    rng = np.random.default_rng(3)
    D = rng.normal(0.3, 1.0, (3, 10))
    m = X.maxt_signflip(D)
    for i in range(3):
        x = D[i]
        r = st.permutation_test((x,), lambda a: np.mean(a) / (np.std(a, ddof=1) / math.sqrt(len(a))),
                                permutation_type="samples", n_resamples=np.inf, alternative="two-sided")
        assert abs(m["single"][i] - r.pvalue) <= 1e-12
        assert m["adjusted"][i] >= m["single"][i] - 1e-15
    one = X.maxt_signflip(D[:1])
    assert one["adjusted"][0] == one["single"][0]


def test_t_ci_and_binomial_closed_forms():
    rng = np.random.default_rng(5)
    D = rng.normal(0.1, 0.5, (7, 16))
    ci = X.seed_mean_t_ci(D)
    ref = st.ttest_1samp(D.mean(1), 0.0).confidence_interval(0.95)
    assert abs(ci[0] - ref.low) <= 1e-12 and abs(ci[1] - ref.high) <= 1e-12
    k, n = 3, 9
    lo, hi = X.clopper_pearson(k, n)
    assert abs(lo - st.beta.ppf(0.025, k, n - k + 1)) <= 1e-10 and abs(hi - st.beta.ppf(0.975, k + 1, n - k)) <= 1e-10
    p = 0.37
    direct = sum(math.comb(9, j) * p ** j * (1 - p) ** (9 - j) for j in range(6, 10))
    assert abs(X.count_rule_oc(9, 6, p) - direct) <= 1e-12


def test_poi_and_profile_vs_src_algorithm():
    pytest.importorskip("jax")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import jax.numpy as jnp
    from src import algorithm as G
    rng = np.random.default_rng(11)
    x = np.round(rng.normal(0, 1, (5, 4)), 1)
    y = np.round(rng.normal(0.2, 1, (6, 4)), 1)  # rounded: exercises ties
    assert abs(X.poi_per_task(x, y) - float(G.poi(jnp.asarray(x), jnp.asarray(y)))) <= 1e-12
    taus = np.linspace(-2, 2, 9)
    assert np.max(np.abs(X.run_score_profile(x, taus) - np.asarray(G.score_profile(jnp.asarray(x), jnp.asarray(taus))))) <= 1e-12


def test_anova2_recovers_components():
    rng = np.random.default_rng(1)
    S, W = 400, 300
    Y = rng.normal(0, 0.6, S)[:, None] + rng.normal(0, 0.3, W)[None, :] + rng.normal(0, 1.0, (S, W))
    r = X.anova2(Y)
    assert abs(r["s2_run"] - 0.36) < 0.07 and abs(r["s2_world"] - 0.09) < 0.03 and abs(r["s2_resid"] - 1.0) < 0.02
