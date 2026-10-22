"""Tests for the recheck fixes (2026-10-20; default-off flags): fresh-optimiser / L2 multi-epoch reading of step 14, the
episodic v1/v2 cooperative reward, the log-rate unit variants  and the tolerant pattern metrics."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import algorithm as alg  # noqa: E402
import data as D  # noqa: E402
import eval as E  # noqa: E402

SPEC = D.EnvSpec(name="t", n_min=3, n_max=3, n_channels=2, k_local=2)


def _fns(**kw):
    tc = alg.TrainCfg(lstm=8, episodes=4, horizon=10, iterations=100, **kw)
    return tc, alg.make_train_fns(SPEC, tc)


def test_fresh_opt_first_iteration_equals_default_then_diverges():
    tc0, f0 = _fns()
    tc1, f1 = _fns(fresh_opt=True)
    key = jax.random.PRNGKey(3)
    q0, q1 = f0.init(jax.random.PRNGKey(0)), f1.init(jax.random.PRNGKey(0))
    tk = jax.random.PRNGKey(9)
    a, _ = f0.iterate(q0, 0, tk)
    b, _ = f1.iterate(q1, 0, tk)
    for x, y in zip(jax.tree.leaves(a.params), jax.tree.leaves(b.params)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-5)       # moments are zero at iteration 0 in both
    a2, _ = f0.iterate(a, 1, tk)
    b2, _ = f1.iterate(b, 1, tk)
    d = max(float(jnp.abs(x - y).max()) for x, y in zip(jax.tree.leaves(a2.params), jax.tree.leaves(b2.params)))
    assert d > 1e-6                                                                  # carried vs restarted Adam moments differ


def test_fresh_first_adam_step_is_lr_sign():
    tc, f = _fns(fresh_opt=True)
    q = f.init(jax.random.PRNGKey(0))
    q2, _ = f.iterate(q, 0, jax.random.PRNGKey(9))
    dl = np.concatenate([np.asarray(b - a).ravel() for a, b in zip(jax.tree.leaves(q.params), jax.tree.leaves(q2.params))])
    big = np.abs(dl) > 0
    assert np.abs(dl[big]).max() <= tc.lr * (1 + 1e-3)                              # |step| <= lr per coordinate
    assert (np.abs(dl[big]) > 0.9 * tc.lr).mean() > 0.5                              # most moving coordinates take ~ lr * sign(g)


def test_l2_adds_half_l2_wsq_to_the_loss():
    tc0, f0 = _fns(fresh_opt=True)
    tc1, f1 = _fns(fresh_opt=True, l2=0.1)
    q = f0.init(jax.random.PRNGKey(0))
    _, i0 = f0.iterate(q, 0, jax.random.PRNGKey(9))
    _, i1 = f1.iterate(q, 0, jax.random.PRNGKey(9))
    w2 = sum(float((w ** 2).sum()) for p, w in jax.tree_util.tree_flatten_with_path(q.params)[0] if "bias" not in str(p[-1]).lower())
    np.testing.assert_allclose(float(i1["loss"]) - float(i0["loss"]), 0.5 * 0.1 * w2, rtol=1e-4)
    names = [str(p[-1]) for p, _ in jax.tree_util.tree_flatten_with_path(q.params)[0]]
    assert any("bias" in n.lower() for n in names) and any("bias" not in n.lower() for n in names)


def test_episodic_reward_is_zero_until_T_and_sums_the_shaped_rewards():
    succ = jnp.asarray(np.random.default_rng(0).random((12, 4)) < 0.4)
    active = jnp.asarray([True, True, True, False])
    r = alg.reward_episodic("sum_rate_episodic", succ, active)
    assert float(jnp.abs(r[:-1]).max()) == 0.0
    shaped = sum(alg.reward_vec("sum_rate", succ[t], active, None) for t in range(12))     # sum_t sum_n 1_n(t), for every user
    np.testing.assert_allclose(np.asarray(r[-1]), np.asarray(shaped), atol=1e-6)
    cnt = np.asarray((succ & active[None]).sum(0), np.float64)
    ref = sum(max(np.log(max(c, 1e-30)), alg.LOG_FLOOR_EP) for c, a in zip(cnt, np.asarray(active)) if a)
    rp = alg.reward_episodic("pf_episodic", succ, active)
    np.testing.assert_allclose(float(rp[-1, 0]), ref, rtol=1e-5)
    assert float(jnp.abs(rp[:-1]).max()) == 0.0


def test_episodic_kind_rolls_out_and_trains():
    tc = alg.TrainCfg(reward="sum_rate_episodic", lstm=8, episodes=3, horizon=8, iterations=10, bootstrap_at_T=False)
    f = alg.make_train_fns(SPEC, tc)
    q = f.init(jax.random.PRNGKey(0))
    out = alg.rollout_learned(SPEC, f.net, q.params, D.world_keys(jax.random.PRNGKey(1), 3), D.world_keys(jax.random.PRNGKey(2), 3), 8, 0.0, 5.0, tc.reward)
    r = np.asarray(out["rews"])                                                       # [M, N, T]
    assert np.abs(r[..., :-1]).max() == 0.0
    total = np.asarray(out["success"] & out["active"][:, None, :]).sum((1, 2))
    np.testing.assert_allclose(r[:, 0, -1], total)
    q2, met = f.iterate(q, 0, jax.random.PRNGKey(5))
    assert np.isfinite(float(met["loss"]))


def test_log_rate_variants_unit_shift_and_floor():
    T, N = 6, 3
    rate = jnp.ones((T, N)) * 10.0
    act = jnp.ones((N,), bool)
    v = E.lr_variants(SPEC, rate, act, jnp.asarray(3.0))
    cum_mbit = np.arange(1, T + 1) * 10.0 * SPEC.slot_s
    np.testing.assert_allclose(np.asarray(v["lrv_ln_mbit"]), np.maximum(np.log(0.01), np.log(cum_mbit)), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(v["lrv_ln_bit"]) - np.asarray(v["lrv_ln_mbit"]), np.log(1e6), rtol=1e-4)   # unfloored: pure unit shift
    # a zero-rate user floors the experiment in every convention at that base's floor value
    rate0 = rate.at[:, 2].set(0.0)
    v0 = E.lr_variants(SPEC, rate0, act, jnp.asarray(3.0))
    np.testing.assert_allclose(np.asarray(v0["lrv_ln_bit"]), np.log(0.01), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(v0["lrv_db_mbit"]), 10 * np.log10(0.01), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(v0["lrv_log2_bit"]), np.log2(0.01), rtol=1e-5)


def test_unit_change_can_flip_an_order_of_floored_mixtures():
    """worked example: arm A (30 % floored, unfloored mean 5) vs arm B (never floored, mean 2); a +6.9 unit shift swaps them."""
    floor = np.log(0.01)

    def score(frac, mean, shift):
        return frac * floor + (1 - frac) * (mean + shift)
    assert score(0.3, 5, 0) > score(0.0, 2, 0) or True
    a0, b0 = score(0.3, 5.0, 0.0), score(0.0, 2.0, 0.0)
    a1, b1 = score(0.3, 5.0, np.log(1000)), score(0.0, 2.0, np.log(1000))
    assert (a0 > b0) != (a1 > b1)


def test_rescale_beta_follows_reward_scale_in_the_ramp():
    tc, f = _fns(reward_scale=0.05, rescale_beta=True)
    q = f.init(jax.random.PRNGKey(0))
    _, m0 = f.iterate(q, 0, jax.random.PRNGKey(9))
    _, m1 = f.iterate(q, tc.iterations - 1, jax.random.PRNGKey(9))
    np.testing.assert_allclose(float(m0["beta"]), 1.0 / 0.05, rtol=1e-6)
    np.testing.assert_allclose(float(m1["beta"]), 20.0 / 0.05, rtol=1e-6)
    tc2, f2 = _fns(reward_scale=0.05)                                                # default: beta untouched
    _, m2 = f2.iterate(f2.init(jax.random.PRNGKey(0)), tc2.iterations - 1, jax.random.PRNGKey(9))
    np.testing.assert_allclose(float(m2["beta"]), 20.0, rtol=1e-6)
