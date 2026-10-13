"""Closed-form tests for the two details we cannot run end to end (E5/E6: Gym-muRTS is not installed):
M1 MultiDiscrete log-prob/entropy summed over components, and X4 invalid-action masking (the gradient of the
log-prob and of the entropy w.r.t. a masked logit is zero; Huang & Ontanon 2020, cited by the post).

Run: uv run python -m pytest tests/test_dist_closed_form.py -q
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import model as mdl  # noqa: E402


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def test_multidiscrete_matches_joint_categorical():
    """M1: prod_i p(a^i) is a distribution over the product space; its log-prob and entropy equal the sums of the
    per-component ones. Checked against the explicit joint Categorical over all 3*2*4 = 24 actions."""
    nvec = (3, 2, 4)
    logits = jax.random.normal(jax.random.PRNGKey(0), (sum(nvec),), jnp.float64) * 2.0
    parts = [logits[0:3], logits[3:5], logits[5:9]]
    logps = [np.asarray(jax.nn.log_softmax(p)) for p in parts]
    joint = {a: sum(lp[ai] for lp, ai in zip(logps, a)) for a in itertools.product(*(range(n) for n in nvec))}
    assert abs(sum(np.exp(v) for v in joint.values()) - 1.0) < 1e-12  # a distribution
    joint_ent = -sum(np.exp(v) * v for v in joint.values())
    for a in [(0, 0, 0), (2, 1, 3), (1, 0, 2)]:
        lp, ent = mdl.multidiscrete_log_prob_entropy(logits, nvec, jnp.array(a))
        np.testing.assert_allclose(float(lp), joint[a], atol=1e-12)
        np.testing.assert_allclose(float(ent), joint_ent, atol=1e-12)
    # batched shapes
    B = 5
    bl = jnp.tile(logits, (B, 1))
    acts = jnp.array([[0, 1, 3]] * B)
    lp, ent = mdl.multidiscrete_log_prob_entropy(bl, nvec, acts)
    assert lp.shape == (B,) and ent.shape == (B,)


def test_masking_zeroes_invalid_action_gradients():
    """X4: with invalid actions masked, d log pi(a) / d logit_j = 0 and d H / d logit_j = 0 for every masked j (and
    the masked actions get probability ~0), while the unmasked logits keep a nonzero gradient."""
    logits = jnp.array([0.3, -1.2, 2.0, 0.7, -0.4], jnp.float64)
    mask = jnp.array([True, False, True, False, True])
    a = 2

    def lp_fn(lg):
        return mdl.log_prob(mdl.mask_logits(lg, mask), jnp.array(a), False)

    g_lp = jax.grad(lp_fn)(logits)
    g_h = jax.grad(lambda lg: mdl.masked_entropy(lg, mask))(logits)
    for g in (g_lp, g_h):
        assert np.all(np.asarray(g)[~np.asarray(mask)] == 0.0)
        assert np.all(np.abs(np.asarray(g)[np.asarray(mask)]) > 1e-6)
    p = jax.nn.softmax(mdl.mask_logits(logits, mask))
    assert float(p[~mask].sum()) < 1e-30
    # the masked distribution equals a Categorical over the valid actions only (closed form)
    valid = logits[mask]
    np.testing.assert_allclose(np.asarray(p[mask]), np.asarray(jax.nn.softmax(valid)), atol=1e-12)
    np.testing.assert_allclose(float(mdl.masked_entropy(logits, mask)),
                               float(mdl.entropy(valid, False)), atol=1e-12)
    # control: WITHOUT masking the same logits get a nonzero gradient (the test has power)
    g_un = jax.grad(lambda lg: mdl.log_prob(lg, jnp.array(a), False))(logits)
    assert np.all(np.abs(np.asarray(g_un)[~np.asarray(mask)]) > 1e-6)
