"""Learning algorithms as pure functions: factorised categorical, PPO clipped surrogate (SB3 semantics), GAE with
time-limit bootstrapping, behaviour cloning (cross-entropy primary, the printed MSE as a descriptive reading) and the
auto-encoder reconstruction loss (README, §7).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


# ----------------------------------------------------------------------------- factorised categorical (MultiDiscrete)
def logp_entropy(logits_az, logits_el, a):
    """log pi(a) = log pi_az(a0) + log pi_el(a1); entropy = H_az + H_el (independent factors)."""
    la, le = jax.nn.log_softmax(logits_az), jax.nn.log_softmax(logits_el)
    lp = jnp.take_along_axis(la, a[..., 0:1], -1)[..., 0] + jnp.take_along_axis(le, a[..., 1:2], -1)[..., 0]
    ent = -jnp.sum(jnp.exp(la) * la, -1) - jnp.sum(jnp.exp(le) * le, -1)
    return lp, ent


def sample_action(key, logits_az, logits_el, deterministic=False):
    if deterministic:
        return jnp.stack([jnp.argmax(logits_az, -1), jnp.argmax(logits_el, -1)], -1).astype(jnp.int32)
    k1, k2 = jax.random.split(key)
    return jnp.stack([jax.random.categorical(k1, logits_az), jax.random.categorical(k2, logits_el)],
                     -1).astype(jnp.int32)


# ----------------------------------------------------------------------------- GAE
def gae(rewards, values, dones, last_value, gamma, lam):
    """SB3 compute_returns_and_advantage. Arrays [T, ...]; dones[t] = 1 if the episode ended at step t (the next value
    is then not bootstrapped; truncation is handled by adding gamma V(final obs) to the reward beforehand, as SB3).
    Returns (advantages, returns = advantages + values)."""
    def body(adv_next, xs):
        r, v, d, v_next = xs
        nonterm = 1.0 - d
        delta = r + gamma * v_next * nonterm - v
        adv = delta + gamma * lam * nonterm * adv_next
        return adv, adv

    v_next = jnp.concatenate([values[1:], last_value[None]], 0)
    _, adv = jax.lax.scan(body, jnp.zeros_like(last_value), (rewards, values, dones, v_next), reverse=True)
    return adv, adv + values


# ----------------------------------------------------------------------------- PPO loss
def ppo_loss_terms(logits_az, logits_el, values, actions, old_logp, adv, returns, clip, normalize_adv=True):
    """SB3 PPO.train for one minibatch: advantage normalisation, clipped surrogate, unclipped value MSE, entropy."""
    if normalize_adv:
        adv = (adv - adv.mean()) / (jnp.std(adv, ddof=1) + 1e-8)   # torch std is unbiased (SB3)
    logp, ent = logp_entropy(logits_az, logits_el, actions)
    ratio = jnp.exp(logp - old_logp)
    pg = -jnp.mean(jnp.minimum(adv * ratio, adv * jnp.clip(ratio, 1 - clip, 1 + clip)))
    vloss = jnp.mean((returns - values) ** 2)
    ent_loss = -jnp.mean(ent)
    clip_frac = jnp.mean((jnp.abs(ratio - 1) > clip).astype(jnp.float32))
    approx_kl = jnp.mean((ratio - 1) - (logp - old_logp))
    return pg, vloss, ent_loss, dict(clip_frac=clip_frac, approx_kl=approx_kl, entropy=-ent_loss)


def ppo_optimizer(lr, max_grad_norm, eps):
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr, eps=eps))


# ----------------------------------------------------------------------------- pre-training losses
def bc_ce_loss(logits_az, logits_el, a, ent_weight):
    """imitation BC: mean NLL of the teacher action - ent_weight * mean entropy ."""
    lp, ent = logp_entropy(logits_az, logits_el, a)
    return -jnp.mean(lp) - ent_weight * jnp.mean(ent)


def bc_mse_loss(logits_az, logits_el, a):
    """The printed 'mean-squared loss between the estimate action a_hat and the true action a', with a_hat ~ pi: the
    exact expectation E_{a_hat ~ pi}[(a_hat - a)^2] summed over the two axes (README (ii))."""
    n = logits_az.shape[-1]
    k = jnp.arange(n, dtype=jnp.float32)
    pa, pe = jax.nn.softmax(logits_az), jax.nn.softmax(logits_el)
    ea = jnp.sum(pa * (k - a[..., 0:1].astype(jnp.float32)) ** 2, -1)
    ee = jnp.sum(pe * (k - a[..., 1:2].astype(jnp.float32)) ** 2, -1)
    return jnp.mean(ea + ee)


def ae_loss(x_hat, x):
    """Per-coordinate MSE ||f_D(f_E(x)) - x||^2 / D (§2.6.2)."""
    return jnp.mean((x_hat - x) ** 2)
