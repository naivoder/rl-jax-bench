"""PPO debug variables and pre-flight checks, with CleanRL's definitions (Huang et al. 2022, detail "Debug variables").

Pure jax.numpy, jit/vmap-safe, dtype-agnostic; numpy arrays are accepted. Every reduction takes an optional ``mask``
(1 = the sample counts, e.g. valid steps of a padded sequence) so the same definition serves flat CleanRL minibatches and masked
recurrent segments. Differential-tested against ``src/algorithm.py`` (itself parity-tested against CleanRL @fe8d8a0 at <= 5.1e-6)
by ``tests/test_ppo_debug.py``.

    kl_estimators(logratio, mask=None)      -> (k1, k2, k3)   k1 = mean(-logr) ("old_approx_kl"), k2 = 0.5 mean(logr^2)
                                                               (baselines' "approxkl"), k3 = mean((r - 1) - logr) (CleanRL)
    clipfrac(logratio, clip_coef, mask=None) -> mean(|r - 1| > clip_coef)                       (CleanRL's definition)
    explained_variance(values, returns, mask=None) -> 1 - Var(R - V) / Var(R), NaN if Var(R) == 0
    ratio_dev(logratio, mask=None)           -> max |r - 1| (the post's "ratio = 1 at the first minibatch" check)
    preflight(...)                           -> dict of named PASS/FAIL clauses for one update (see its docstring)

Notes learned in the reproduction (see the README): the ratio = 1 check needs full
float32 matmuls (TF32 on an RTX 3090 gives |r - 1| ~ 1e-3), and for a Gaussian policy its float32 floor scales with
1/sigma, so pass a tolerance that reflects the policy's precision rather than a fixed 1e-5.
"""
from __future__ import annotations

import jax.numpy as jnp


def _mean(x, mask):
    x = jnp.asarray(x)
    if mask is None:
        return x.mean()
    m = jnp.asarray(mask, dtype=x.dtype)
    return (x * m).sum() / jnp.maximum(m.sum(), 1.0)


def _max(x, mask):
    x = jnp.asarray(x)
    if mask is None:
        return x.max()
    return jnp.where(jnp.asarray(mask, dtype=bool), x, -jnp.inf).max()


def kl_estimators(logratio, mask=None):
    """(k1, k2, k3) of KL(old || new) from logratio = log pi_new - log pi_old on samples drawn from pi_old."""
    logratio = jnp.asarray(logratio)
    ratio = jnp.exp(logratio)
    return (_mean(-logratio, mask), _mean(0.5 * logratio ** 2, mask), _mean((ratio - 1.0) - logratio, mask))


def clipfrac(logratio, clip_coef, mask=None):
    """Fraction of samples whose ratio lies outside [1 - clip_coef, 1 + clip_coef] (CleanRL ``clipfracs``)."""
    r = jnp.exp(jnp.asarray(logratio))
    return _mean((jnp.abs(r - 1.0) > clip_coef).astype(r.dtype), mask)


def explained_variance(values, returns, mask=None):
    """1 - Var(returns - values) / Var(returns) (population variances, CleanRL / baselines); NaN when Var(returns) = 0."""
    values, returns = jnp.asarray(values), jnp.asarray(returns)
    mu_y = _mean(returns, mask)
    var_y = _mean((returns - mu_y) ** 2, mask)
    d = returns - values
    var_d = _mean((d - _mean(d, mask)) ** 2, mask)
    return jnp.where(var_y == 0, jnp.nan, 1.0 - var_d / jnp.where(var_y == 0, 1.0, var_y))


def ratio_dev(logratio, mask=None):
    """max |ratio - 1|; on the first minibatch of the first epoch it must be ~0 (paper's debugging check)."""
    return _max(jnp.abs(jnp.exp(jnp.asarray(logratio)) - 1.0), mask)


def preflight(*, first_mb_logratio=None, approx_kl=None, clipfrac_value=None, entropy=None, entropy_floor=None,
              ratio_tol=1e-5, kl_max=0.02, clipfrac_min=0.01, floor_margin=0.05, mask=None):
    """Named PASS (True) / FAIL (False) clauses for one PPO update; clauses whose inputs are None are omitted.

    * ``ratio_is_one``      max|r - 1| <= ratio_tol on the first minibatch of the first epoch    (the paper's check)
    * ``kl_below``          approx_kl < kl_max (0.02 in the post; it names no estimator, k1 met it more often here)
    * ``actor_moves``       clipfrac >= clipfrac_min                       (OUR extension, not the paper's: a band)
    * ``entropy_at_floor``  entropy >= entropy_floor - floor_margin        (OUR extension for floor-controlled lanes)
    Returns plain Python bools when called eagerly; inside jit the values are traced booleans.
    """
    out = {}
    if first_mb_logratio is not None:
        out["ratio_is_one"] = ratio_dev(first_mb_logratio, mask) <= ratio_tol
    if approx_kl is not None:
        out["kl_below"] = jnp.asarray(approx_kl) < kl_max
    if clipfrac_value is not None:
        out["actor_moves"] = jnp.asarray(clipfrac_value) >= clipfrac_min
    if entropy is not None and entropy_floor is not None:
        out["entropy_at_floor"] = jnp.asarray(entropy) >= jnp.asarray(entropy_floor) - floor_margin
    try:
        return {k: bool(v) for k, v in out.items()}
    except Exception:  # traced (inside jit): leave as arrays
        return out
