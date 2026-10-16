"""Parity with rliable@3ccd9f4 from the committed fixture tests/fixtures/oracle_small.npz (built by
scripts/build_fixture.py from the oracle run; numbers only). Point estimates and bootstrap replicates replayed
on the oracle's own np.random indices must agree to 1e-12 (float64)."""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from src import algorithm as A

HERE = os.path.dirname(os.path.abspath(__file__))
F = np.load(os.path.join(HERE, "fixtures", "oracle_small.npz"))
TRIMS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)


def test_points():
    for i in F["meta_points"]:
        x, y = jnp.asarray(F[f"x{i:03d}"]), jnp.asarray(F[f"y{i:03d}"])
        ours = np.array([A.agg_mean(x), A.agg_median(x), A.agg_iqm(x), A.agg_og(x), A.agg_og(x, 0.5)])
        np.testing.assert_allclose(ours, F[f"pt_{i:03d}_agg"], rtol=0, atol=1e-12)
        np.testing.assert_allclose([A.trimmed_mean(x, t) for t in TRIMS], F[f"pt_{i:03d}_trim"], rtol=0, atol=1e-12)
        t = F[f"pt_{i:03d}_taus"]
        np.testing.assert_allclose(A.score_profile(x, t), F[f"pt_{i:03d}_prof"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(A.avg_score_profile(x, t), F[f"pt_{i:03d}_avgprof"], rtol=0, atol=1e-12)
        np.testing.assert_allclose([A.poi(x, y), A.poi(y, x), A.poi(x, x)], F[f"pt_{i:03d}_poi"], rtol=0, atol=1e-12)


def test_replay_stratified_task_and_independent():
    for i in F["meta_small"]:
        x = jnp.asarray(F[f"x{i:03d}"])
        ours = A.bootstrap_dist_from_indices((x,), (jnp.asarray(F[f"rp_{i:03d}_idx"]),), A.aggregates4)
        np.testing.assert_allclose(ours, F[f"rp_{i:03d}_res"], rtol=0, atol=1e-12)
    for i in (0, 10):
        x, y = jnp.asarray(F[f"x{i:03d}"]), jnp.asarray(F[f"y{i:03d}"])
        ours = jax.vmap(lambda a, b: A.aggregates4(A.gather_runs(x, a, b)))(jnp.asarray(F[f"rptask_{i:03d}_idx"]),
                                                                             jnp.asarray(F[f"rptask_{i:03d}_tidx"]))
        np.testing.assert_allclose(ours, F[f"rptask_{i:03d}_res"], rtol=0, atol=1e-12)
        ours = A.bootstrap_dist_from_indices((x, y), (jnp.asarray(F[f"rppoi_{i:03d}_ix"]), jnp.asarray(F[f"rppoi_{i:03d}_iy"])), A.poi)
        np.testing.assert_allclose(ours, F[f"rppoi_{i:03d}_res"], rtol=0, atol=1e-12)


def test_studentized_official_form():
    import yaml
    s = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), "src", "configs", "sources.yaml")))["normalisation"]
    for steps in ("100k", "500k"):
        for alg in ("CURL", "SUNRISE", "CURL-D2RL"):
            m, ci = A.studentized_ci_official(s[f"dmc_reported_means_{steps}"]["value"][alg],
                                              s[f"dmc_reported_stds_{steps}"]["value"][alg], 5 if alg == "SUNRISE" else 10)
            # 1e-10: the oracle venv's scipy 1.13.1 t.ppf is 6e-11 off (mpmath; E0 amendment 1)
            np.testing.assert_allclose([m, *ci], F[f"stud_{steps}_{alg}"], rtol=0, atol=1e-10)
