"""Gradient/closed-form correctness.  There is NO learned parameter and no loss in this paper (an index policy computed from
deterministic covariance rollouts), so there is no training gradient to check.  What *is* differentiable and what the method
relies on are the TEC map phi (eqs 3-7) and the work/cost metrics F, G for a FIXED action pattern; these are checked against
finite differences (jax.test_util.check_grads) and against hand-derived closed forms.  The threshold (z) dependence of F, G is
piecewise constant (a step function of z), so by design no gradient in z is tested."""
import numpy as np
import jax, jax.numpy as jnp
from jax.test_util import check_grads
import model as M
import algorithm as A


def _bank(qct=4.0, typ="reckless"):
    return M.scalar_bank([typ], [qct], 1.0)


def test_phi_gradient_wrt_state_matches_finite_difference():
    b = _bank()
    for a in (0.0, 1.0):
        f = lambda P: M.phi(b, P.reshape(1, 1, 1), jnp.array([a])).reshape(())
        check_grads(f, (jnp.array(1.3),), order=2, modes=["rev", "fwd"], atol=1e-6, rtol=1e-6)


def test_phi_untracked_gradient_closed_form():
    # D3: phi^0(P) = sum_m u^{0,m}(F_m^2 P + Q_m)  =>  d phi^0 / dP = sum_m u^{0,m} F_m^2 (= 1.258 reckless, 1.234 cautious)
    for typ, expect in [("reckless", 0.9 * 1.21 + 0.1 * 1.69), ("cautious", 0.95 * 1.21 + 0.05 * 1.69)]:
        b = _bank(typ=typ)
        g = jax.grad(lambda P: M.phi(b, P.reshape(1, 1, 1), jnp.zeros(1)).reshape(()))(jnp.array(2.0))
        assert abs(float(g) - expect) < 1e-12


def test_phi_tracked_gradient_closed_form():
    # phi^1(P) = sum_m u^{1,m} R Pbar_m / (Pbar_m + R), Pbar_m = F_m^2 P + Q_m: derivative R^2 F_m^2 / (Pbar_m + R)^2 per mode
    b = _bank()
    P = 1.7
    exp = 0.0
    for u, F, Q in [(0.2, 1.1, 1.0), (0.8, 1.3, 4.0)]:
        pb = F * F * P + Q
        exp += u * 4.0 * F * F / (pb + 2.0) ** 2
    g = jax.grad(lambda P: M.phi(b, P.reshape(1, 1, 1), jnp.ones(1)).reshape(()))(jnp.array(P))
    assert abs(float(g) - exp) < 1e-12


def test_gradient_wrt_process_noise_and_matrix_case():
    def f(q):
        bank = M.scalar_bank(["reckless"], [4.0], 1.0)._replace(Q=jnp.stack([jnp.array([[1.0]]), q.reshape(1, 1)])[None])
        return M.phi(bank, jnp.ones((1, 1, 1)), jnp.ones(1)).reshape(())
    check_grads(f, (jnp.array(4.0),), order=2, modes=["rev", "fwd"], atol=1e-6, rtol=1e-6)
    bm = M.matrix_bank(["reckless"], [3.0], 1.0)
    P0 = jnp.eye(4)[None] * 0.8 + 0.1
    g = lambda P: jnp.sum(M.phi(bm, P.reshape(1, 4, 4), jnp.ones(1)))
    check_grads(g, (P0.reshape(4, 4),), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)


def test_cost_metric_gradient_fixed_actions():
    # F(P,<0,inf>) (never tracked after k = 0) is a smooth function of P (no threshold crossing): finite-difference check.
    b = _bank()
    cfg = A.Cfg(tau=15)
    def F(P):
        v, _ = A.rollout_FG(b, P.reshape(1, 1, 1), jnp.full(1, jnp.inf), jnp.zeros(1), 15, 0.9)
        return v[0]
    check_grads(F, (jnp.array(1.1),), order=2, modes=["rev", "fwd"], atol=1e-5, rtol=1e-6)
    # closed form: passive affine recursion => dF/dP = sum_k beta^k (a^k) with a = sum_m u0_m F_m^2
    a = 0.9 * 1.21 + 0.1 * 1.69
    exp = sum(0.9 ** k * a ** k for k in range(15))
    assert abs(float(jax.grad(F)(jnp.array(1.1))) - exp) < 1e-8 * exp
