"""Differential test against a labelled SECOND-BEST reference: the CleanRL-differential-tested PPO estimators of the
sibling reproduction papers/ppo_implementation_details (src/algorithm.py; located by PPO_REFERENCE, default
../ppo_implementation_details/src/algorithm.py; the test skips when it is absent). Shared estimators must agree exactly;
the documented differences (paper's 1/(1+eps) lower bound vs 1-eps; keras MSE without 1/2) are checked as such.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from src import algorithm as A  # noqa: E402
import importlib.util  # noqa: E402

import pytest  # noqa: E402

_REF = os.environ.get("PPO_REFERENCE", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                    "ppo_implementation_details", "src", "algorithm.py"))
if not os.path.exists(_REF):
    pytest.skip(f"reference PPO estimators not found at {_REF} (set PPO_REFERENCE)", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("ppo_reference_algorithm", _REF)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def test_gae_matches_reference():
    with jax.enable_x64(True):
        rng = np.random.default_rng(0)
        T, N = 8, 16
        v = jnp.asarray(rng.normal(size=(T + 1, N)))
        r = jnp.asarray(rng.normal(size=(T, N)))
        d = jnp.asarray(rng.random((T, N)) < 0.15)
        tgt, adv = A.gae(v, r, d, 0.99, 0.9)
        # reference convention: dones[t] travels with obs[t] (= "the previous step ended an episode")
        dones_ref = jnp.concatenate([jnp.zeros((1, N)), d[:-1].astype(jnp.float64)], 0)
        adv_r, ret_r = R.gae(r, v[:-1], dones_ref, v[-1], d[-1].astype(jnp.float64), 0.99, 0.9)
        np.testing.assert_allclose(np.asarray(adv), np.asarray(adv_r), atol=1e-12)
        np.testing.assert_allclose(np.asarray(tgt), np.asarray(ret_r), atol=1e-12)


def test_ppo_surrogate_upper_branch_and_lower_bound_difference():
    with jax.enable_x64(True):
        rng = np.random.default_rng(1)
        lr = jnp.asarray(rng.normal(size=4000) * 0.3)
        adv = jnp.asarray(rng.normal(size=4000))
        eps = 0.2
        ours = lambda x, m: jnp.mean(-jnp.minimum(jnp.exp(x) * adv, jnp.clip(jnp.exp(x), 1 / (1 + eps), 1 + eps) * adv) * m)
        ref = lambda x, m: jnp.mean(jnp.maximum(-adv * jnp.exp(x), -adv * jnp.clip(jnp.exp(x), 1 - eps, 1 + eps)) * m)
        rho = np.exp(np.asarray(lr))
        # values agree except where A < 0 and rho < 1/(1+eps) (different constant of the clipped branch)
        same = jnp.asarray(~((rho < 1 / (1 + eps)) & (np.asarray(adv) < 0)))
        m = same.astype(jnp.float64)
        np.testing.assert_allclose(float(ours(lr, m)), float(ref(lr, m)), atol=1e-12)
        g1, g2 = jax.grad(ours)(lr, m), jax.grad(ref)(lr, m)
        np.testing.assert_allclose(np.asarray(g1), np.asarray(g2), atol=1e-12)
        band = (rho > 1 - eps) & (rho < 1 / (1 + eps)) & (np.asarray(adv) < 0)
        assert band.sum() > 0
        gb1 = np.asarray(jax.grad(ours)(lr, jnp.ones(4000)))[band]
        gb2 = np.asarray(jax.grad(ref)(lr, jnp.ones(4000)))[band]
        assert np.all(gb1 == 0.0) and np.all(gb2 != 0.0)     # paper footnote 19: clipped here, standard PPO is not


def test_value_loss_and_adv_norm_and_adam():
    with jax.enable_x64(True):
        rng = np.random.default_rng(2)
        nv, ov, ret = (jnp.asarray(rng.normal(size=64)) for _ in range(3))
        vc = jnp.clip(nv, ov - 0.2, ov + 0.2)
        ours = jnp.mean(jnp.maximum((ret - nv) ** 2, (ret - vc) ** 2))
        assert abs(float(ours) - 2 * float(R.value_loss(nv, ov, ret, 0.2, True))) < 1e-12     # keras MSE has no 1/2
        a = jnp.asarray(rng.normal(size=64))
        am = a - a.mean()
        np.testing.assert_allclose(np.asarray(am / (jnp.sqrt(jnp.mean(am * am)) + 1e-8)),
                                   np.asarray(R.normalize_adv(a, ddof=0)), atol=1e-12)
    # eps-hat Adam: ours (keras) == reference TF1 transform at constant lr
    cfg = A.LaneCfg(*[jnp.float32(0)] * len(A.LaneCfg._fields))._replace(
        lr=jnp.float32(3e-3), lr_horizon_steps=jnp.float32(1e9), lr_end=jnp.float32(1.0), adam_b1=jnp.float32(0.9),
        adam_b2=jnp.float32(0.999), adam_eps=jnp.float32(1e-7))
    p = {"w": jnp.array([0.3, -1.0, 2.0], jnp.float32)}
    ours, ref = A.keras_adam(cfg), optax.chain(R.scale_by_adam_tf1(eps=1e-7), optax.scale(-3e-3))
    so, sr = ours.init(p), ref.init(p)
    po, pr = p, p
    for t in range(6):
        g = {"w": jnp.array([0.1, -0.2, 0.05], jnp.float32) * (t + 1)}
        uo, so = ours.update(g, so)
        ur, sr = ref.update(g, sr)
        po, pr = optax.apply_updates(po, uo), optax.apply_updates(pr, ur)
    np.testing.assert_allclose(np.asarray(po["w"]), np.asarray(pr["w"]), rtol=1e-6)
