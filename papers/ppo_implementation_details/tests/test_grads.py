"""Unit tests: the exactly computable worked examples W1-W12 (float64, atol 1e-6), the W9 bandit
estimator z-test, and finite-difference check_grads on the full PPO loss (feed-forward and LSTM).

Run: uv run python -m pytest tests/test_grads.py -q
"""
from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from src import algorithm as alg  # noqa: E402
from src import data as dt  # noqa: E402
from src import model as mdl  # noqa: E402

ATOL = 1e-6
G, L = 0.9, 0.8  # gamma, lambda (gamma*lambda = 0.72)


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def f64(tree):
    return jax.tree.map(lambda p: p.astype(jnp.float64), tree)


def a64(x):
    return jnp.asarray(x, jnp.float64)


# ------------------------------------------------------------------ W1 GAE
def _w1_inputs():
    # (M, N) = (3, 2): column 0 = env0, column 1 = env1
    r = a64([[1, 1], [1, 0], [1, 2]])
    v = a64([[0.5, 1.0], [0.6, 0.2], [0.7, 0.3]])
    d = a64([[0, 0], [0, 0], [0, 1]])
    return r, v, d, a64([0.8, 5.0]), a64([0.0, 1.0])


def test_w1_gae_done_mask_and_next_done():
    adv, ret = alg.gae(*_w1_inputs(), G, L)
    np.testing.assert_allclose(adv[:, 0], [2.310368, 1.7644, 1.02], atol=ATOL)
    np.testing.assert_allclose(adv[:, 1], [0.036, -0.2, 1.7], atol=ATOL)
    np.testing.assert_allclose(ret[:, 0], [2.810368, 2.3644, 1.72], atol=ATOL)
    np.testing.assert_allclose(ret[:, 1], [1.036, 0.0, 2.0], atol=ATOL)


def test_w1_gae_truncation_correct_variant():
    r, v, d, nv, nd = _w1_inputs()
    truncs = a64([[0, 0], [0, 1], [0, 0]])          # env1 transition 1 was a time-limit truncation
    finals = a64([[0, 0], [0, 0.9], [0, 0]])        # V(o^final) = 0.9
    adv, _ = alg.gae(r, v, d, nv, nd, G, L, truncs, finals)
    np.testing.assert_allclose(adv[:, 1], [0.6192, 0.61, 1.7], atol=ATOL)
    np.testing.assert_allclose(adv[:, 0], [2.310368, 1.7644, 1.02], atol=ATOL)  # untouched env


def test_w1_gae_chain_closed_form():
    M = 5
    adv, _ = alg.gae(jnp.ones((M, 1)), jnp.zeros((M, 1)), jnp.zeros((M, 1)), a64([0.0]), a64([0.0]), G, L)
    expect = [(1 - 0.72 ** (M - t)) / 0.28 for t in range(M)]
    np.testing.assert_allclose(adv[:, 0], expect, atol=ATOL)
    np.testing.assert_allclose(adv[:, 0], [2.88038656, 2.611648, 2.2384, 1.72, 1.0], atol=ATOL)


# ------------------------------------------------------------------ W2 advantage normalization
def test_w2_minibatch_and_batch_normalisation():
    mb = a64([2.310368, 1.7644, 0.036, -0.2])
    np.testing.assert_allclose(alg.normalize_adv(mb, ddof=0), [1.23355254, 0.72819324, -0.87164964, -1.09009613],
                               atol=ATOL)
    np.testing.assert_allclose(alg.normalize_adv(mb, ddof=1), [1.06828784, 0.63063384, -0.75487073, -0.94405095],
                               atol=ATOL)
    full = a64([2.310368, 1.7644, 1.02, 0.036, -0.2, 1.7])
    np.testing.assert_allclose(alg.normalize_adv(full, ddof=0),
                               [1.30793888, 0.71544878, -0.09238178, -1.16022874, -1.41633837, 0.64556123], atol=ATOL)


# ------------------------------------------------------------------ W3 clipped surrogate
def test_w3_clipped_surrogate_value_grad_clipfrac():
    rho, adv = a64([0.7, 0.9, 1.3, 1.5]), a64([1, -1, 1, -1])
    f = lambda lp: alg.clipped_surrogate(jnp.exp(lp), adv, 0.2)  # noqa: E731
    np.testing.assert_allclose(f(jnp.log(rho)), 0.125, atol=ATOL)
    np.testing.assert_allclose(jax.grad(f)(jnp.log(rho)), [-0.175, 0.225, 0.0, 0.375], atol=ATOL)
    z = jnp.zeros(4)
    _, aux = alg.ppo_terms(jnp.log(rho), z, z, z, z, z, adv, clip_coef=0.2, clip_vloss=True, ent_coef=0.0,
                           vf_coef=0.0, norm_adv_mb=False, ddof=1)
    np.testing.assert_allclose(aux["clipfrac"], 0.75, atol=ATOL)
    np.testing.assert_allclose(aux["pg_loss"], 0.125, atol=ATOL)
    # only 1 of 4 samples has zero gradient although clipfrac = 0.75
    assert int((jnp.abs(jax.grad(f)(jnp.log(rho))) < 1e-12).sum()) == 1


# ------------------------------------------------------------------ W4 clipped value loss
def test_w4_clipped_value_loss():
    v_old, ret = a64([1, 1, 1]), a64([2, 0, 0])
    f = lambda v: alg.value_loss(v, v_old, ret, 0.2, True)  # noqa: E731
    v = a64([1.5, 0.9, 0.5])
    np.testing.assert_allclose(f(v), 0.34833333333, atol=ATOL)
    np.testing.assert_allclose(jax.grad(f)(v), [0.0, 0.3, 0.0], atol=ATOL)
    np.testing.assert_allclose(alg.value_loss(v, v_old, ret, 0.2, False), 0.5 * np.mean([0.25, 0.81, 0.25]), atol=ATOL)


# ------------------------------------------------------------------ W5 distributions
def test_w5_entropy_and_logprob():
    np.testing.assert_allclose(mdl.entropy(a64([0.0, math.log(2.0)]), False), 0.636514168, atol=ATOL)
    dist = (a64([0.0, 1.0]), a64([0.0, math.log(0.5)]))
    np.testing.assert_allclose(mdl.log_prob(dist, a64([1.0, 1.0]), True), -1.644729886, atol=ATOL)
    np.testing.assert_allclose(mdl.entropy(dist, True), 2.144729886, atol=ATOL)
    # categorical log-prob closed form
    np.testing.assert_allclose(mdl.log_prob(a64([0.0, math.log(2.0)]), jnp.asarray(1), False), math.log(2 / 3),
                               atol=ATOL)


# ------------------------------------------------------------------ W6 / W7 / W11 optimiser
def test_w6_global_norm_clip():
    g = {"a": a64([3.0]), "b": a64([4.0])}
    out, _ = optax.clip_by_global_norm(0.5).update(g, optax.clip_by_global_norm(0.5).init(g))
    np.testing.assert_allclose([out["a"][0], out["b"][0]], [0.3, 0.4], atol=ATOL)
    # torch convention min(1, c/(||g||+1e-6)) differs by ~6e-8 relative
    torch_like = np.array([3.0, 4.0]) * min(1.0, 0.5 / (5.0 + 1e-6))
    np.testing.assert_allclose(torch_like, [0.29999994, 0.39999992], atol=1e-8)


def test_w7_adam_eps_first_step():
    for eps, expect in [(1e-5, 0.5), (1e-8, 0.999000999)]:
        tx = alg.make_optimizer(1.0, 1e9, eps)
        p = {"w": a64([0.0])}
        upd, _ = tx.update({"w": a64([1e-5])}, tx.init(p), p)
        np.testing.assert_allclose(-upd["w"][0], expect, atol=1e-6)


def test_adam_tf1_epsilon_hat():
    """TF1 AdamOptimizer step = lr*sqrt(1-b2^t)/(1-b1^t) * m/(sqrt(v)+eps), written out by hand,
    equals the bias-corrected form with eps_eff = eps/sqrt(1-b2^t); at t=1 a gradient of 1e-5 (lr 1) moves by
    g/(|g| + eps/sqrt(1-b2)) = 1/(1+31.62) = 0.0307, instead of torch/optax's 0.5 (W7)."""
    b1, b2, eps = 0.9, 0.999, 1e-5
    tx = alg.make_optimizer(1.0, 1e9, eps, "tf1")
    p = {"w": a64([0.0])}
    st = tx.init(p)
    gs = [1e-5, -3e-5, 2e-5, 4e-6, 1e-5]
    m = v = 0.0
    for t, g in enumerate(gs, 1):
        upd, st = tx.update({"w": a64([g])}, st, p)
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        tf1 = math.sqrt(1 - b2 ** t) / (1 - b1 ** t) * m / (math.sqrt(v) + eps)
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        np.testing.assert_allclose(-upd["w"][0], tf1, rtol=1e-6)
        np.testing.assert_allclose(tf1, mh / (math.sqrt(vh) + eps / math.sqrt(1 - b2 ** t)), rtol=1e-9)
        if t == 1:
            np.testing.assert_allclose(-upd["w"][0], 1.0 / (1.0 + 1.0 / math.sqrt(1 - b2)), rtol=1e-6)


def test_w11_lr_schedule():
    lrs = [float(alg.lr_at(jnp.int32(u), 4, 1.0, True)) for u in range(4)]
    np.testing.assert_allclose(lrs, [1.0, 0.75, 0.5, 0.25], atol=1e-7)
    tx = alg.make_optimizer(2.5e-4, 0.5, 1e-5)
    st = alg.set_lr(tx.init({"w": jnp.zeros(1)}), 0.123)
    np.testing.assert_allclose(st.hyperparams["learning_rate"], 0.123, rtol=1e-6)


# ------------------------------------------------------------------ W8 reward scaling (baselines mode)
def test_w8_reward_scaling_baselines_sequence():
    rewards, dones = [1.0, 1.0, 1.0, 0.0, 2.0], [0, 0, 1, 0, 0]  # episode ends after the 3rd reward
    ret, rms = a64([0.0]), dt.rms_init((), jnp.float64)
    var_seq, out = [], []
    for r, d in zip(rewards, dones):
        ret, rms, rt = dt.scale_reward_baselines(ret, rms, a64([r]), jnp.asarray([bool(d)]), 0.99, 1e-8, 10.0)
        var_seq.append(float(rms.var))
        out.append(float(rt[0]))
    np.testing.assert_allclose(var_seq, [0.00019997, 0.24517449, 0.64703111, 1.22527387, 1.02184137], atol=1e-8)
    np.testing.assert_allclose(out, [10.0, 2.01958601, 1.24318974, 0.0, 1.97851001], atol=ATOL)


def test_w8_reward_scaling_gymnasium_differs_on_truncation():
    """The gymnasium-0.29.1 accumulator is not reset on a truncation (and zeroes BEFORE adding on termination)."""
    ret, rms = a64(0.0), dt.rms_init((), jnp.float64)
    for r, term in [(1.0, False), (1.0, True), (1.0, False)]:
        ret, rms, _ = dt.scale_reward_gymnasium029(ret, rms, a64(r), jnp.asarray(term), 0.99, 1e-8, 10.0)
    np.testing.assert_allclose(float(ret), 1.0 * 0.99 + 1.0, atol=ATOL)  # 0 * ... + 1 at the termination, then +1


# ------------------------------------------------------------------ W9 bandit estimator (statistical)
def test_w9_bandit_policy_gradient_expectation():
    n = 200_000
    key = jax.random.PRNGKey(9)
    theta = a64(0.0)
    acts = jax.random.categorical(key, jnp.zeros((n, 2), jnp.float64), axis=-1)
    rew = jnp.where(acts == 0, 1.0, 0.0).astype(jnp.float64)  # arm 0 has logit theta, reward 1

    def per_sample_loss(th, a, adv):
        lp = mdl.log_prob(jnp.stack([th, a64(0.0)]), a, False)
        ratio = jnp.exp(lp - jax.lax.stop_gradient(lp))      # rho == 1 at theta_old = theta
        return alg.clipped_surrogate(ratio[None], adv[None], 0.2)

    g = jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0))(theta, acts, rew)
    mean, var = float(g.mean()), float(g.var())
    z = (mean - (-0.25)) / math.sqrt(var / n)
    assert abs(z) < 3.29, (mean, z)                          # two-sided alpha = 1e-3
    assert abs(var - 0.0625) < 0.002, var                    # per-sample variance p(1-p)^2 - 0.0625


# ------------------------------------------------------------------ W10 KL estimators
def test_w10_kl_estimators():
    k1, k2, k3 = alg.kl_estimators(a64([0.1, -0.2]))
    np.testing.assert_allclose([k1, k2, k3], [0.05, 0.0125, 0.011950836], atol=1e-8)


# ------------------------------------------------------------------ W12 LSTM reset / reconstruction
def test_w12_lstm_reset_and_reconstruction():
    net = mdl.LSTMAgent(n_out=2, hidden=8, lstm_hidden=6, reset=True)
    k = jax.random.PRNGKey(0)
    T, B, d = 5, 3, 4
    x = jax.random.normal(k, (T, B, d), jnp.float64)
    carry0 = net.initial_carry(B)
    carry0 = jax.tree.map(lambda z: z.astype(jnp.float64), carry0)
    params = f64(net.init(k, x, jnp.zeros((T, B)), carry0))
    h_rand = (jax.random.normal(jax.random.PRNGKey(1), (B, 6)), jax.random.normal(jax.random.PRNGKey(2), (B, 6)))
    # done at t=0 => the input state is exactly (0, 0) whatever the carry
    d1 = jnp.zeros((T, B)).at[0].set(1.0)
    l_rand, _, _ = net.apply(params, x, d1, h_rand)
    l_zero, _, _ = net.apply(params, x, d1, carry0)
    np.testing.assert_allclose(l_rand, l_zero, atol=1e-12)
    # sequence pass == step-by-step pass (L5 reconstruction gives rho == 1)
    d2 = jnp.zeros((T, B)).at[2, 1].set(1.0)
    seq, _, c_seq = net.apply(params, x, d2, h_rand)
    c, steps = h_rand, []
    for t in range(T):
        lg, _, c = net.apply(params, x[t:t + 1], d2[t:t + 1], c)
        steps.append(lg[0])
    np.testing.assert_allclose(seq, jnp.stack(steps), atol=1e-12)
    np.testing.assert_allclose(c_seq[0], c[0], atol=1e-12)
    # training from zero instead of the saved h0 changes log-probs before the first in-segment reset
    nz, _, _ = net.apply(params, x, jnp.zeros((T, B)), h_rand)
    z0, _, _ = net.apply(params, x, jnp.zeros((T, B)), carry0)
    assert float(jnp.abs(nz[0] - z0[0]).max()) > 1e-4
    # with reset disabled, a done no longer zeroes the state
    net_nr = mdl.LSTMAgent(n_out=2, hidden=8, lstm_hidden=6, reset=False)
    a_, _, _ = net_nr.apply(params, x, d1, h_rand)
    assert float(jnp.abs(a_[0] - l_zero[0]).max()) > 1e-4


# ------------------------------------------------------------------ finite-difference check of the full loss
def _ppo_loss_ff(net, cont, batch, clip_vloss=True):
    def f(params):
        dist, v = net.apply(params, batch["obs"])
        lp = mdl.log_prob(dist, batch["act"], cont)
        ent = mdl.entropy(dist, cont)
        loss, _ = alg.ppo_terms(lp, batch["old_lp"], ent, v, batch["old_v"], batch["ret"], batch["adv"],
                                clip_coef=0.2, clip_vloss=clip_vloss, ent_coef=0.01, vf_coef=0.5,
                                norm_adv_mb=True, ddof=1)
        return loss
    return f


@pytest.mark.parametrize("arch,cont", [("separate", False), ("shared", False), ("separate", True)])
def test_check_grads_full_ppo_loss(arch, cont):
    k = jax.random.split(jax.random.PRNGKey(3), 8)
    B, d, A = 16, 4, 2
    net = (mdl.SeparateAC if arch == "separate" else mdl.SharedAC)(A, cont, 8)
    obs = jax.random.normal(k[0], (B, d), jnp.float64)
    params = f64(net.init(k[1], obs))
    params = jax.tree.map(lambda p: p + 0.3 * jax.random.normal(k[2], p.shape, p.dtype), params)  # away from init
    act = (jax.random.normal(k[3], (B, A), jnp.float64) if cont else jax.random.randint(k[3], (B,), 0, A))
    dist, v = net.apply(params, obs)
    lp = mdl.log_prob(dist, act, cont)
    batch = dict(obs=obs, act=act, old_lp=lp + 0.15 * jax.random.normal(k[4], (B,), jnp.float64),
                 old_v=v + 0.15 * jax.random.normal(k[5], (B,), jnp.float64),
                 ret=jax.random.normal(k[6], (B,), jnp.float64), adv=jax.random.normal(k[7], (B,), jnp.float64))
    check_grads(_ppo_loss_ff(net, cont, batch), (params,), order=2, modes=["rev"], atol=1e-5, rtol=1e-5)


def test_check_grads_lstm_loss():
    k = jax.random.split(jax.random.PRNGKey(4), 6)
    T, B, d = 4, 2, 3
    net = mdl.LSTMAgent(n_out=2, hidden=5, lstm_hidden=4, reset=True)
    x = jax.random.normal(k[0], (T, B, d), jnp.float64)
    dones = jnp.zeros((T, B)).at[2, 0].set(1.0)
    c0 = (0.5 * jax.random.normal(k[1], (B, 4)), 0.5 * jax.random.normal(k[2], (B, 4)))
    params = f64(net.init(k[3], x, dones, c0))
    act = jax.random.randint(k[4], (T, B), 0, 2)
    adv = jax.random.normal(k[5], (T * B,), jnp.float64)

    logits0, v0, _ = net.apply(params, x, dones, c0)
    old_lp = mdl.log_prob(logits0, act, False).reshape(-1) + 0.1   # fixed constants (not functions of p)
    old_v = v0.reshape(-1) + 0.05

    def f(p):
        logits, v, _ = net.apply(p, x, dones, c0)
        lp = mdl.log_prob(logits, act, False).reshape(-1)
        loss, _ = alg.ppo_terms(lp, old_lp, mdl.entropy(logits, False).reshape(-1), v.reshape(-1), old_v, adv, adv,
                                clip_coef=0.2, clip_vloss=True, ent_coef=0.01, vf_coef=0.5, norm_adv_mb=True, ddof=1)
        return loss
    check_grads(f, (params,), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)
