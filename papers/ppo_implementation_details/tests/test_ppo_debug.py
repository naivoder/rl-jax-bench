"""src/ppo_debug.py against this repo's parity-tested src/algorithm.py and CleanRL's numpy formulas (CPU, < 5 s).

CleanRL @fe8d8a0 ppo.py: old_approx_kl = (-logratio).mean(); approx_kl = ((ratio - 1) - logratio).mean();
clipfracs = ((ratio - 1.0).abs() > clip_coef).float().mean(); explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as alg  # noqa: E402
from src import ppo_debug as pd  # noqa: E402


def _batch(seed=0, n=512):
    rng = np.random.default_rng(seed)
    logratio = rng.normal(0.0, 0.15, n).astype(np.float32)
    returns = rng.normal(1.0, 2.0, n).astype(np.float32)
    values = (returns + rng.normal(0.0, 0.7, n)).astype(np.float32)
    return logratio, values, returns


def test_matches_repo_algorithm_and_cleanrl_formulas():
    logratio, values, returns = _batch()
    k1, k2, k3 = pd.kl_estimators(logratio)
    r1, r2, r3 = alg.kl_estimators(jnp.asarray(logratio))
    np.testing.assert_allclose([k1, k2, k3], [r1, r2, r3], rtol=1e-6, atol=1e-8)
    ratio = np.exp(logratio.astype(np.float64))
    np.testing.assert_allclose(k3, ((ratio - 1) - logratio).mean(), rtol=1e-5)
    # clipfrac as ppo_terms reports it, and as CleanRL computes it
    z = jnp.zeros_like(jnp.asarray(logratio))
    _, aux = alg.ppo_terms(jnp.asarray(logratio), z, z, z, z, z, jnp.ones_like(z), clip_coef=0.2, clip_vloss=False,
                           ent_coef=0.0, vf_coef=0.0, norm_adv_mb=False, ddof=1)
    np.testing.assert_allclose(pd.clipfrac(logratio, 0.2), aux["clipfrac"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(pd.clipfrac(logratio, 0.2), (np.abs(ratio - 1.0) > 0.2).mean(), atol=1e-7)
    np.testing.assert_allclose(pd.ratio_dev(logratio), aux["max_ratio_dev"], rtol=1e-6)
    ev_ref = 1 - np.var(returns - values) / np.var(returns)
    np.testing.assert_allclose(pd.explained_variance(values, returns), ev_ref, rtol=1e-5)
    np.testing.assert_allclose(pd.explained_variance(values, returns),
                               alg.explained_variance(jnp.asarray(values), jnp.asarray(returns)), rtol=1e-5)
    assert np.isnan(float(pd.explained_variance(values, np.ones_like(returns))))


def test_mask_equals_subset_and_is_jittable():
    logratio, values, returns = _batch(1)
    mask = (np.arange(logratio.size) % 3 == 0)
    sub = mask.astype(bool)
    for full, part in [(pd.kl_estimators(logratio, mask), pd.kl_estimators(logratio[sub])),
                       ((pd.clipfrac(logratio, 0.1, mask),), (pd.clipfrac(logratio[sub], 0.1),)),
                       ((pd.ratio_dev(logratio, mask),), (pd.ratio_dev(logratio[sub]),)),
                       ((pd.explained_variance(values, returns, mask),), (pd.explained_variance(values[sub], returns[sub]),))]:
        np.testing.assert_allclose(np.asarray(full, dtype=np.float64), np.asarray(part, dtype=np.float64), rtol=2e-5)
    f = jax.jit(lambda lr, m: pd.clipfrac(lr, 0.2, m))
    np.testing.assert_allclose(f(jnp.asarray(logratio), jnp.asarray(mask)), pd.clipfrac(logratio, 0.2, mask), atol=1e-7)


def test_preflight_clauses_have_power():
    ok = pd.preflight(first_mb_logratio=np.zeros(8, np.float32), approx_kl=0.01, clipfrac_value=0.13, entropy=0.40,
                      entropy_floor=0.362)
    assert ok == {"ratio_is_one": True, "kl_below": True, "actor_moves": True, "entropy_at_floor": True}
    # the CREDIT-B failure modes the post-hoc audit saw: BUNDLE clipfrac 0.000, FIX entropy 0.035 under a 0.362 floor,
    # and a TF32-sized first-minibatch ratio error
    bad = pd.preflight(first_mb_logratio=np.full(8, 8.5e-4, np.float32), approx_kl=0.03, clipfrac_value=0.0,
                       entropy=0.035, entropy_floor=0.362)
    assert bad == {"ratio_is_one": False, "kl_below": False, "actor_moves": False, "entropy_at_floor": False}
