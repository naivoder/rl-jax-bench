"""Gradient / correctness tests.

The method has NO learned parameters and the search (MCTS) is not differentiable, so there is no loss to differentiate.  The smooth
building blocks are still checked with finite differences (jax.test_util.check_grads) so that a wrong closed form shows up:
the synthetic-update covariance (eq. 36), the Lemma-1 bound as a function of (r, P) away from its kinks, the merged Bernoulli
(eqs. 54-56), and the Kalman/PMB update's dependence on the measurement.  The statistical correctness of the estimator (the bound
dominates the Monte-Carlo MSGOSPA; martingale under the exact p(h)) is in test_toy.py.
"""
import numpy as np
import jax, jax.numpy as jnp
from jax.test_util import check_grads
from src import planner as PL
from src import filter as FL
from src import model as M


def _pd_P(rng, n=4):
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


def test_synthetic_update_covariance_grad():
    pc = PL.PC(M.CFG_P)
    P = jnp.asarray(_pd_P(np.random.default_rng(0)))[None]
    f = lambda P_: PL._syn_update_cov(P_, pc)[0]
    check_grads(f, (P,), order=2, modes=["fwd", "rev"], atol=1e-5, rtol=1e-5)


def test_node_cost_and_merge_grad_away_from_kinks():
    cfg = M.CFG_P
    pc = PL.PC(cfg)
    rng = np.random.default_rng(1)
    P = jnp.asarray(_pd_P(rng) * 10.0)
    x = jnp.asarray([5.0, 0.1, 5.0, 0.1])
    pos = jnp.asarray([[0.0, 0.0], [10.0, 30.0]])

    def f(r, P_):
        b = PL.Belief(jnp.stack([r, 0.0]), jnp.stack([x, jnp.zeros(4)]), jnp.stack([P_, jnp.eye(4)]), jnp.asarray(1, jnp.int32))
        c, nb = PL.eval_belief(b, pos, pc, cfg, "GD")
        return c, nb.r[0], nb.P[0]

    # r = 0.3 sits where the min-form is in the "report" regime for the detection hypotheses and "miss" for h=(0,0): no kink crossing in eps
    check_grads(f, (jnp.asarray(0.3), P), order=1, modes=["fwd", "rev"], atol=1e-4, rtol=1e-4)


def test_bernoulli_kld_cost_grad():
    cfg = M.CFG_P
    pc = PL.PC(cfg)
    P = jnp.asarray(_pd_P(np.random.default_rng(2)) * 5.0)
    x = jnp.asarray([5.0, 0.1, 5.0, 0.1])
    pos = jnp.asarray([[0.0, 0.0], [10.0, 30.0]])

    def f(r, P_):
        b = PL.Belief(jnp.stack([r, 0.0]), jnp.stack([x, jnp.zeros(4)]), jnp.stack([P_, jnp.eye(4)]), jnp.asarray(1, jnp.int32))
        return PL.eval_belief(b, pos, pc, cfg, "KLD")[0]

    check_grads(f, (jnp.asarray(0.4), P), order=1, modes=["fwd", "rev"], atol=1e-4, rtol=1e-4)


def test_filter_update_grad_wrt_measurement():
    cfg = M.CFG_P.replace(lbp_mode="converged", lbp_tol=1e-12)
    NB = M.NB_MAX
    P = jnp.asarray(_pd_P(np.random.default_rng(3)))
    fs = FL.FState(jnp.zeros(NB).at[0].set(0.6), jnp.zeros((NB, 4)).at[0].set(jnp.asarray([1.0, 0.0, 1.0, 0.0])),
                   jnp.tile(jnp.eye(4), (NB, 1, 1)).at[0].set(P), jnp.asarray(1, jnp.int32))
    zv = jnp.zeros(M.M_MEAS, bool).at[0].set(True)

    def f(z0):
        z = jnp.zeros((M.M_MEAS, 2)).at[0].set(z0)
        out, _ = FL.update_sensor(fs, z, zv, jnp.asarray([0.0, 0.0]), 0.01, cfg)
        return out.r[0], out.x[0], out.P[0]

    check_grads(f, (jnp.asarray([1.5, 0.5]),), order=1, modes=["fwd"], atol=1e-5, rtol=1e-5)
