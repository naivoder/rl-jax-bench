"""Gradient / estimator correctness on exactly computable toys, small sizes (<~30 s CPU).

The paper's method has gradients, so check_grads is used where a function is differentiable (the tabular-MDP J);
the instrument itself is an ESTIMATOR, so its statistical correctness is tested against closed forms: bandit
moments, the surrogate-landscape identity, the SNR cosine model, g_prac on the 2-state MDP, the GAE/lambda-return
identity and the Returns-MRE toys. scripts/toy_checks.py runs the same functions at full size.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from src import algorithm as A
from src import toys
from src.model import Nets, gauss_kl, gauss_logp
from src.configs.default import algo_config


def test_bandit_moments():
    r = toys.bandit_estimator(jax.random.PRNGKey(1), 200_000, d=1.0, s=1.0)
    assert r["pass_bandit_moments"], r


def test_bandit_landscape_identity():
    r = toys.bandit_landscape(jax.random.PRNGKey(2), 200_000)
    assert r["pass_landscape_identity"], r


def test_cosine_model():
    r = toys.cosine_model(jax.random.PRNGKey(3), d=4000, n_half=2000.0, sizes=(500, 4000), reps=60, N=100_000)
    assert r["pass_cosine_model"], r


def test_mdp_gprac_and_gradJ():
    r = toys.mdp_pipeline(jax.random.PRNGKey(4), n_traj=40_000)
    assert r["pass_mdp"], r


def test_gae_identity():
    r = toys.gae_identity(jax.random.PRNGKey(5), n_traj=8, T=120)
    assert r["pass_gae_identity"], r


def test_mre_toys():
    r = toys.mre_toys(jax.random.PRNGKey(6), N=400_000)
    assert r["pass_mre"], r


def test_check_grads_policy_logp_and_kl():
    """check_grads (finite differences) on the policy log-likelihood and the Gaussian KL used by TRPO."""
    nets = Nets(5, 2, algo_config("trpo", "hopper"))
    p = nets.init(jax.random.PRNGKey(0))[0]
    obs = jax.random.normal(jax.random.PRNGKey(1), (7, 5))
    act = jax.random.normal(jax.random.PRNGKey(2), (7, 2))
    jax.config.update("jax_enable_x64", True)
    try:
        p64, obs64, act64 = (x.astype(jnp.float64) for x in (p, obs, act))
        check_grads(lambda q: jnp.sum(nets.logp(q, obs64, act64)), (p64,), order=2, modes=["rev"])
        m0, l0 = nets.pi_apply(p64, obs64)
        check_grads(lambda q: jnp.mean(gauss_kl(m0, l0, *nets.pi_apply(q, obs64))), (p64 + 0.01,), order=2, modes=["rev"])
    finally:
        jax.config.update("jax_enable_x64", False)


def test_pg_vector_equals_score_times_adv():
    """: grad of mean(ratio * A_hat) at theta_k == mean(psi * A_hat)."""
    nets = Nets(4, 3, algo_config("ppo", "hopper"))
    p = nets.init(jax.random.PRNGKey(0))[0]
    obs = jax.random.normal(jax.random.PRNGKey(1), (64, 4))
    act = jax.random.normal(jax.random.PRNGKey(2), (64, 3))
    adv = jax.random.normal(jax.random.PRNGKey(3), (64,))
    g = A.pg_vector(nets, p, obs, act, adv)
    psi = jax.vmap(lambda o, a: jax.grad(lambda q: nets.logp(q, o[None], a[None])[0])(p))(obs, act)
    g2 = jnp.mean(psi * A.adv_normalize(adv)[:, None], 0)
    np.testing.assert_allclose(np.asarray(g), np.asarray(g2), rtol=2e-4, atol=1e-6)


def test_cg_solves_spd_system():
    M = jax.random.normal(jax.random.PRNGKey(0), (6, 6))
    H = M @ M.T + 6 * jnp.eye(6)
    b = jnp.arange(6.0)
    x = A.cg_solve(lambda v: H @ v, b, 10)
    np.testing.assert_allclose(np.asarray(H @ x), np.asarray(b), atol=1e-3)
