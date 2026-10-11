"""Gradient checks.

HCA's policy update is not the gradient of any per-episode loss (it is a counterfactual estimator), so the
finite-difference check is done where gradients exist and the estimator is tested statistically/exactly:
  1. jax.test_util.check_grads on the DP value V_theta(x0) (the metric and the exact-check ground truth);
  2. the exact expectation of the State-HCA update (enumerated trajectories, exact hindsight) equals a central
     finite difference of the DP value (closed-form unbiasedness);
  3. the hand-written hindsight updates (onehot - h) equal autodiff of the cross-entropy log-likelihood, and the
     PG / Return-HCA logit updates equal autodiff of sum_i log pi(A_i|o_i) * stop_grad(advantage_i).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import src  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from src import data as D  # noqa: E402
from src.algorithm import mc_returns, pg_grads, return_bins, return_hca_grads, state_hca_grads, windows, segment_returns  # noqa: E402
from src.eval import enumerate_trajectories, exact_pair_hindsight, start_value, to_jnp_episode, trajectories_to_batch  # noqa: E402
from src.model import init_params, policy  # noqa: E402


def _episode(env, theta, seed):
    pi = policy(theta)
    return D.sample_episode(D.env_arrays(env), env.s0, env.L, lambda o: pi[o], jax.random.PRNGKey(seed))


def test_check_grads_dp_value():
    env = D.shortcut()
    th = 0.5 * jax.random.normal(jax.random.PRNGKey(0), (env.n_obs, 2))
    check_grads(lambda t: start_value(D.env_arrays(env), t, env.s0, env.L), (th,), order=2, modes=("rev",))


def test_expected_state_hca_equals_finite_difference():
    env = D.delayed_effect(3, 2.0)
    th = 0.6 * jax.random.normal(jax.random.PRNGKey(3), (env.n_obs, 2))
    pi = np.asarray(policy(th))
    hp, _ = exact_pair_hindsight(env, pi)
    tab = np.full((env.n_obs, env.n_obs, 2), 0.5)
    for k, v in hp.items():
        tab[k] = v
    with np.errstate(divide="ignore"):
        params = init_params(env.n_obs, 2)._replace(theta=th, psi=jnp.asarray(np.log(tab)))
    probs, batch = trajectories_to_batch(env, enumerate_trajectories(env, pi))
    g = jax.vmap(lambda e: state_hca_grads(params, e, None, 1.0).theta)(to_jnp_episode(batch))
    g_exp = np.asarray(jnp.tensordot(jnp.asarray(probs), g, axes=1))
    f = lambda t: float(start_value(D.env_arrays(env), t, env.s0, env.L))  # noqa: E731
    fd = np.zeros_like(g_exp)
    h = 1e-5
    for idx in np.ndindex(*fd.shape):
        e = np.zeros(fd.shape)
        e[idx] = h
        fd[idx] = (f(th + e) - f(th - e)) / (2 * h)
    np.testing.assert_allclose(g_exp, fd, atol=1e-8)


def test_hindsight_updates_are_cross_entropy_gradients():
    env = D.shortcut()
    k = jax.random.PRNGKey(4)
    p = init_params(env.n_obs, 10)._replace(theta=jax.random.normal(k, (env.n_obs, 2)),
                                            psi=jax.random.normal(jax.random.fold_in(k, 1), (env.n_obs, env.n_obs, 2)),
                                            phi=jax.random.normal(jax.random.fold_in(k, 2), (env.n_obs, 10, 2)))
    centres = jnp.asarray(D.bin_centres(env, 10))
    for seed in range(5):
        ep = _episode(env, p.theta, seed)
        w = windows(ep, None, 1.0)
        o, a = ep.obs, ep.action

        def ll_state(psi):
            logh = jax.nn.log_softmax(psi[o[:, None], o[None, :]], -1)[..., :]
            return jnp.sum(w.hpair * jnp.take_along_axis(logh, jnp.broadcast_to(a[:, None, None], logh.shape[:2] + (1,)), -1)[..., 0])

        np.testing.assert_allclose(np.asarray(state_hca_grads(p, ep, None, 1.0).psi), np.asarray(jax.grad(ll_state)(p.psi)), atol=1e-12)
        b = return_bins(mc_returns(ep, 1.0), centres)

        def ll_ret(phi):
            return jnp.sum(ep.decision * jax.nn.log_softmax(phi[o, b], -1)[jnp.arange(o.shape[0]), a])

        np.testing.assert_allclose(np.asarray(return_hca_grads(p, ep, 1.0, centres).phi), np.asarray(jax.grad(ll_ret)(p.phi)), atol=1e-12)


def test_logit_updates_are_score_function_gradients():
    env = D.delayed_effect(5, 1.0)
    k = jax.random.PRNGKey(5)
    p = init_params(env.n_obs, 3)._replace(theta=jax.random.normal(k, (env.n_obs, 2)),
                                           phi=jax.random.normal(jax.random.fold_in(k, 1), (env.n_obs, 3, 2)),
                                           V=jax.random.normal(jax.random.fold_in(k, 2), (env.n_obs,)))
    centres = jnp.linspace(-3.0, 3.0, 3)
    for seed in range(5):
        ep = _episode(env, p.theta, seed)
        o, a = ep.obs, ep.action
        w = windows(ep, 3, 1.0)
        adv_pg = jax.lax.stop_gradient(segment_returns(ep, w, p.V) - p.V[o])
        Z = mc_returns(ep, 1.0)
        hz = jax.nn.softmax(p.phi[o, return_bins(Z, centres)], -1)
        pi_a = policy(p.theta)[o, a]
        adv_ret = jax.lax.stop_gradient(Z * (1 - pi_a / hz[jnp.arange(o.shape[0]), a]))

        def score(theta, adv):
            return jnp.sum(ep.decision * jax.nn.log_softmax(theta[o], -1)[jnp.arange(o.shape[0]), a] * adv)

        np.testing.assert_allclose(np.asarray(pg_grads(p, ep, 3, 1.0).theta), np.asarray(jax.grad(score)(p.theta, adv_pg)), atol=1e-12)
        np.testing.assert_allclose(np.asarray(return_hca_grads(p, ep, 1.0, centres).theta), np.asarray(jax.grad(score)(p.theta, adv_ret)), atol=1e-12)


def test_sliding_window_state_hca_unbiased_with_exact_v():
    """Sensitivity: the sliding-window n-step State-HCA estimator (each source bootstraps from its own
    step i + n) is exactly unbiased on the fully observed Shortcut, given the exact V and the exact pair hindsight of
    the sliding pairs (enumerated independently in eval.np_pairs). Also the sliding window with n >= L is MC."""
    from src.eval import dp_values, true_policy_gradient
    env = D.shortcut()
    th = 0.7 * jax.random.normal(jax.random.PRNGKey(11), (env.n_obs, 2))
    pi = np.asarray(policy(th))
    V = dp_values(D.env_arrays(env), th, env.L)
    probs, batch = trajectories_to_batch(env, enumerate_trajectories(env, pi))
    eps = to_jnp_episode(batch)
    g_true = np.asarray(true_policy_gradient(D.env_arrays(env), th, env.s0, env.L))
    for T in (2, 3):
        hp, _ = exact_pair_hindsight(env, pi, T, "sliding")
        psi = np.zeros((env.n_obs, env.n_obs, 2))
        for (ox, oy), v in hp.items():
            psi[ox, oy] = np.log(np.maximum(v, 1e-300))
        p = init_params(env.n_obs, 2)._replace(theta=th, psi=jnp.asarray(psi), V=V)
        g = jax.vmap(lambda e: state_hca_grads(p, e, T, 1.0, "official", seg_mode="sliding").theta)(eps)
        g_exp = np.tensordot(probs, np.asarray(g), axes=1)
        np.testing.assert_allclose(g_exp, g_true, atol=1e-8)
        # and the sliding PG (n-step actor-critic with the exact V) is unbiased too
        gp = jax.vmap(lambda e: pg_grads(p, e, T, 1.0, "sliding").theta)(eps)
        np.testing.assert_allclose(np.tensordot(probs, np.asarray(gp), axes=1), g_true, atol=1e-8)
    ep = _episode(env, th, 5)
    p = init_params(env.n_obs, 2)._replace(theta=th)
    a = state_hca_grads(p, ep, env.L, 1.0, seg_mode="sliding").theta
    b = state_hca_grads(p, ep, None, 1.0).theta
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-14)
