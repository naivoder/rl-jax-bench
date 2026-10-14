"""Estimator correctness tests (float64, CPU).

* the expected TRPO surrogate gradient when the actions are drawn from the trainee pi_psi (A2D at beta = 0).
  The batch enumerates the actions with integer multiplicities proportional to pi_psi, so batch means are exact
  expectations. 'paper' (Eq. 27) must give sum_a pi_theta A grad log pi_theta; 'official' (centered) must give that
  minus m (pi_psi - pi_theta) in logit space, m the batch mean of w A. Both scaled by the same std normalizer.
* TRPO freeze: on a near-deterministic policy (min prob ~1e-156) the official KL form gives a non-finite
  Fisher-vector product and the TRPO step is refused; the log-prob fallback takes a finite, accepted step; and on a
  normal policy the fallback changes nothing, bit for bit.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
from src import algorithm as AL  # noqa: E402

PI_PSI = np.array([0.5, 0.25, 0.125, 0.125])
COUNTS = (PI_PSI * 8).astype(int)               # 4, 2, 1, 1 -> batch of 8 = exact expectation under pi_psi
A_ACT = jnp.asarray(np.repeat(np.arange(4), COUNTS))
Q = jnp.asarray([3.0, -1.0, 0.5, 2.0])          # advantage per action at the single state


def _grads(estimator, ent=0.0, reg=False):
    z0 = jnp.asarray([0.3, -0.2, 0.1, 0.0])     # pi_theta logits at the single state (expert)
    lp_e = jax.nn.log_softmax(z0)
    lp_l = jnp.log(jnp.asarray(PI_PSI))
    old = lp_e[A_ACT]
    w = AL.importance_weight(old, lp_l[A_ACT], 0.0)
    adv = w * Q[A_ACT]
    f = lambda z: AL.surrogate(jnp.broadcast_to(z, (8, 4)), A_ACT, old, adv, ent, reg, estimator)
    return jax.grad(f)(z0), jax.nn.softmax(z0), adv


def test_paper_estimator_is_eq27_in_expectation():
    g, pt, adv = _grads('paper')
    sig = jnp.std(adv, ddof=1) + 1e-3
    true = pt * (Q - jnp.sum(pt * Q))            # sum_a pi_theta A grad_z log pi_theta (logit space)
    np.testing.assert_allclose(g * sig, true, atol=1e-9)


def test_official_estimator_has_the_mean_bias():
    g, pt, adv = _grads('official')
    m = adv.mean()
    sig = jnp.std(adv, ddof=1) + 1e-3
    true = pt * (Q - jnp.sum(pt * Q))
    bias = -m * (jnp.asarray(PI_PSI) - pt)       # E_{pi_psi}[-m grad log pi_theta] = -m (pi_psi - pi_theta)
    np.testing.assert_allclose(g * sig, true + bias, atol=1e-9)
    assert float(jnp.abs(bias).max()) > 0.1      # and the bias is not negligible here


def _degenerate(scale):
    # linear policy on 4 one-hot states; logits = scale * (one-hot of a preferred action) -> min prob ~ e^-scale
    X = jnp.eye(4)[jnp.arange(200) % 4]
    W = scale * jnp.eye(4)[jnp.asarray([1, 2, 3, 0])]
    apply = lambda p, x: jax.nn.log_softmax(x @ p['W'], -1)
    A = jnp.asarray(np.random.RandomState(0).randint(0, 4, 200))
    old = jnp.take_along_axis(apply({'W': W}, X), A[:, None], 1)[:, 0]
    adv = jnp.asarray(np.random.RandomState(1).randn(200))
    sel = jnp.arange(20)
    return apply, {'W': W}, X, A, old, adv, sel


def test_kl_fallback_unfreezes_degenerate_policy():
    apply, p, X, A, old, adv, sel = _degenerate(358.0)          # min prob 1.5e-156, the replayed frozen state
    assert float(jnp.exp(apply(p, X)).min()) < 1e-150
    cfg_off = AL.TRPOCfg(max_kl=0.01, ent=0.0, reg_adv=False, kl_fallback=False,
                         loglik_clamp=False)
    cfg_on = cfg_off._replace(kl_fallback=True)
    _, i_off = AL.trpo_step(apply, p, X, A, old, adv, cfg_off, sel, jnp.float64(0.1))
    new, i_on = AL.trpo_step(apply, p, X, A, old, adv, cfg_on, sel, jnp.float64(0.1))
    assert not bool(i_off['accepted']) and float(i_off['damping']) == 0.2      # official: frozen, damping doubled
    assert bool(i_on['kl_fallback']) and bool(i_on['accepted']) and float(i_on['damping']) == 0.1
    assert bool(jnp.all(jnp.isfinite(new['W'])))


def test_kl_fallback_is_bitwise_noop_on_normal_policy():
    apply, p, X, A, old, adv, sel = _degenerate(1.5)
    cfg_off = AL.TRPOCfg(max_kl=0.01, ent=0.0, reg_adv=False, kl_fallback=False,
                         loglik_clamp=False)
    a, ia = jax.jit(lambda: AL.trpo_step(apply, p, X, A, old, adv, cfg_off, sel, jnp.float64(0.1)))()
    b, ib = jax.jit(lambda: AL.trpo_step(apply, p, X, A, old, adv, cfg_off._replace(kl_fallback=True), sel,
                                         jnp.float64(0.1)))()
    assert bool(ia['accepted']) and not bool(ib['kl_fallback'])
    assert np.array_equal(np.asarray(a['W']), np.asarray(b['W']))


def test_kl_logp_equals_kl_probs():
    lp = jax.nn.log_softmax(jnp.asarray(np.random.RandomState(2).randn(16, 4)))
    lq = jax.nn.log_softmax(jnp.asarray(np.random.RandomState(3).randn(16, 4)))
    np.testing.assert_allclose(AL.kl_logp(lp, lq), AL.kl_probs(jnp.exp(lp), jnp.exp(lq)), rtol=1e-12)


def test_entropy_fallback_unfreezes_zero_probability_policy():
    # entropy route of the TRPO freeze: logits 800 apart -> some probabilities are exactly 0 in float64; with the entropy folded into the
    # advantage (main args) the official -sum log(p) p is NaN, so the surrogate and its gradient are NaN.
    apply, p, X, A, old, adv, sel = _degenerate(800.0)
    assert float(jnp.exp(apply(p, X)).min()) == 0.0
    assert not bool(jnp.isfinite(AL.entropy_of(apply(p, X))).all())
    assert bool(jnp.isfinite(AL.entropy_of(apply(p, X), safe=True)).all())
    cfg_off = AL.TRPOCfg(max_kl=0.01, ent=10.0, reg_adv=True, kl_fallback=False,
                         loglik_clamp=False)
    _, i_off = AL.trpo_step(apply, p, X, A, old, adv, cfg_off, sel, jnp.float64(0.1))
    new, i_on = AL.trpo_step(apply, p, X, A, old, adv, cfg_off._replace(kl_fallback=True), sel, jnp.float64(0.1))
    assert not bool(i_off['accepted'])
    assert bool(i_on['accepted']) and bool(jnp.all(jnp.isfinite(new['W'])))


def test_safe_entropy_equals_official_when_positive():
    lp = jax.nn.log_softmax(jnp.asarray(np.random.RandomState(4).randn(16, 4)) * 5)
    assert np.array_equal(np.asarray(AL.entropy_of(lp)), np.asarray(AL.entropy_of(lp, safe=True)))


def test_loglik_clamp_matches_torch_categorical_convention():
    # official get_loglikelihood: torch Categorical(probs).log_prob = log(clamp(p, eps, 1 - eps)), eps = 2.22e-16:
    # value floored at log(eps) = -36.04 and zero gradient for a taken action outside [eps, 1 - eps]
    z = jnp.asarray([[0.0, -50.0, 1.0, 2.0]])
    for a, clamped in ((1, True), (3, False)):
        A = jnp.asarray([a])
        v = AL.action_logp(z, A, True)
        g = jax.grad(lambda zz: AL.action_logp(zz, A, True).sum())(z)
        if clamped:
            np.testing.assert_allclose(float(v[0]), np.log(2.220446049250313e-16), rtol=1e-12)
            assert float(jnp.abs(g).max()) == 0.0
        else:
            np.testing.assert_allclose(v, AL.action_logp(z, A, False), atol=1e-14)
            np.testing.assert_allclose(g, jax.grad(lambda zz: AL.action_logp(zz, A, False).sum())(z), atol=1e-14)


def test_loglik_clamp_degenerate_policy_has_no_gradient():
    # with the official clamp every sample of a policy whose probabilities are all < eps or > 1 - eps carries zero
    # surrogate gradient (official: no step, no NaN), which is why the official code never reaches the TRPO freeze
    apply, p, X, A, old, adv, sel = _degenerate(358.0)
    old_c = AL.action_logp(apply(p, X), A, True)
    cfg = AL.TRPOCfg(max_kl=0.01, ent=0.0, reg_adv=False)
    assert cfg.loglik_clamp
    g = jax.grad(lambda W: AL.surrogate(apply({'W': W}, X), A, old_c, adv, 0.0, False, clamp=True))(p['W'])
    assert float(jnp.abs(g).max()) == 0.0
