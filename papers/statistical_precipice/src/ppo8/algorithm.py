"""PPO estimators and losses as pure functions (paper C4-C12).

All functions are jit/vmap-safe and dtype-agnostic (float32 in training, float64 in the Gate-1 tests).
Array layout for rollouts: (M, N) = (num_steps, num_envs), dones[t] stored WITH obs[t] (paper C1).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple


# ------------------------------------------------------------------ GAE (paper C5; the README-2)
def gae(rewards, values, dones, next_value, next_done, gamma, lam, truncs=None, final_values=None):
    """Truncated, masked GAE.

    rewards, values, dones: (M, N). next_value, next_done: (N).
    nnt_t = 1 - dones[t+1] (t < M-1), 1 - next_done (t = M-1).
    Official (default, truncs=None): delta_t = r_t + gamma * nnt_t * V_{t+1} - V_t.
    Truncation-correct (OUR EXTENSION, the README: additionally + gamma * truncs_t * V(o^final_{t+1}), where
    truncs_t = 1 iff transition t ended by a time limit (and not a termination).
    Returns (advantages, returns = advantages + values) (TD(lambda) target, the README.
    """
    next_values = jnp.concatenate([values[1:], next_value[None]], axis=0)
    nnt = 1.0 - jnp.concatenate([dones[1:], next_done[None]], axis=0)
    boot = gamma * nnt * next_values
    if truncs is not None:
        boot = boot + gamma * truncs * (1.0 - nnt) * final_values
    deltas = rewards + boot - values

    def body(lastgaelam, x):
        delta, m = x
        a = delta + gamma * lam * m * lastgaelam
        return a, a

    _, adv = jax.lax.scan(body, jnp.zeros_like(next_value), (deltas, nnt), reverse=True)
    return adv, adv + values


# ------------------------------------------------------------------ advantage normalisation (paper C7)
def normalize_adv(adv, ddof: int = 1, eps: float = 1e-8):
    """(A - mean) / (std + 1e-8) over the given set (a minibatch by default). ddof: D1."""
    n = adv.size
    m = adv.mean()
    var = ((adv - m) ** 2).sum() / (n - ddof)
    return (adv - m) / (jnp.sqrt(var) + eps)


# ------------------------------------------------------------------ losses (paper C8, C9, C10)
def clipped_surrogate(ratio, adv, clip_coef):
    """mean(max(-A rho, -A clip(rho, 1-eps, 1+eps)))  the README3.1)."""
    return jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)).mean()


def value_loss(new_v, old_v, returns, clip_coef, clip_vloss: bool):
    """0.5 * mean(max((V-R)^2, (V_old + clip(V-V_old, -eps, eps) - R)^2)) the README5.1; Q3, Q4)."""
    unclipped = (new_v - returns) ** 2
    if not clip_vloss:
        return 0.5 * unclipped.mean()
    v_clipped = old_v + jnp.clip(new_v - old_v, -clip_coef, clip_coef)
    return 0.5 * jnp.maximum(unclipped, (v_clipped - returns) ** 2).mean()


def kl_estimators(logratio):
    """k1 = mean(-logratio) ('old_approx_kl'), k2 = 0.5 mean(logratio^2) (baselines), k3 = mean((r-1) - logratio)."""
    ratio = jnp.exp(logratio)
    return (-logratio).mean(), 0.5 * (logratio ** 2).mean(), ((ratio - 1.0) - logratio).mean()


def ppo_terms(new_logp, old_logp, entropy, new_v, old_v, returns, adv, *, clip_coef, clip_vloss, ent_coef, vf_coef,
              norm_adv_mb: bool, ddof: int):
    """Total loss (paper C10) and the debug variables (paper C12) for one minibatch (all inputs flat)."""
    logratio = new_logp - old_logp
    ratio = jnp.exp(logratio)
    if norm_adv_mb:
        adv = normalize_adv(adv, ddof)
    pg = clipped_surrogate(ratio, adv, clip_coef)
    vl = value_loss(new_v, old_v, returns, clip_coef, clip_vloss)
    ent = entropy.mean()
    loss = pg - ent_coef * ent + vf_coef * vl
    k1, k2, k3 = kl_estimators(jax.lax.stop_gradient(logratio))
    r = jax.lax.stop_gradient(ratio)
    aux = dict(pg_loss=pg, v_loss=vl, entropy=ent, k1=k1, k2=k2, k3=k3,
               clipfrac=(jnp.abs(r - 1.0) > clip_coef).astype(r.dtype).mean(),
               max_ratio_dev=jnp.abs(r - 1.0).max())
    return loss, aux


def explained_variance(values, returns):
    var_y = jnp.var(returns)
    return jnp.where(var_y == 0, jnp.nan, 1.0 - jnp.var(returns - values) / var_y)


# ------------------------------------------------------------------ optimiser (paper C3, C4, C11)
class AdamTF1State(NamedTuple):
    count: jax.Array
    mu: optax.Updates
    nu: optax.Updates


def scale_by_adam_tf1(b1: float = 0.9, b2: float = 0.999, eps: float = 1e-5) -> optax.GradientTransformation:
    """TF1 tf.train.AdamOptimizer (baselines ppo2/model.py:100): eps is added to the UN-bias-corrected sqrt(v)
    ("epsilon hat"): step = sqrt(1-b2^t)/(1-b1^t) * m / (sqrt(v) + eps). In bias-corrected (torch/optax) form this is
    m_hat / (sqrt(v_hat) + eps / sqrt(1-b2^t)), i.e. an effective eps of ~3.2e-4 at t=1 the README. Diagnostic
    flag only (adam_eps_mode: tf1); the default is torch/optax semantics, as CleanRL and the post's code."""
    def init(params):
        z = jax.tree.map(jnp.zeros_like, params)
        return AdamTF1State(jnp.zeros([], jnp.int32), z, jax.tree.map(jnp.zeros_like, params))

    def update(g, state, params=None):
        mu = jax.tree.map(lambda m, x: b1 * m + (1 - b1) * x, state.mu, g)
        nu = jax.tree.map(lambda v, x: b2 * v + (1 - b2) * x * x, state.nu, g)
        def step(m, v):
            t = (state.count + 1).astype(m.dtype)
            return jnp.sqrt(1 - b2 ** t) / (1 - b1 ** t) * m / (jnp.sqrt(v) + eps)

        upd = jax.tree.map(step, mu, nu)
        return upd, AdamTF1State(state.count + 1, mu, nu)

    return optax.GradientTransformation(init, update)


def make_optimizer(lr0: float, max_grad_norm: float, adam_eps: float, eps_mode: str = "torch"):
    """Global-norm clip (paper C11) -> Adam(eps=1e-5) (paper C3); lr injectable per update (paper C4).
    eps_mode "torch" = optax/torch semantics (CleanRL, the post's code); "tf1" = baselines' TF1 semantics (D16)."""
    def chain(learning_rate):
        if eps_mode == "tf1":
            return optax.chain(optax.clip_by_global_norm(max_grad_norm), scale_by_adam_tf1(eps=adam_eps),
                               optax.scale_by_learning_rate(learning_rate))
        assert eps_mode == "torch", eps_mode
        return optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate, eps=adam_eps))
    return optax.inject_hyperparams(chain)(learning_rate=lr0)


def lr_at(update0, num_updates: int, lr0: float, anneal: bool):
    """lr_u = lr0 * (1 - u/U) for 0-based u  ==  CleanRL frac = 1 - (iteration-1)/num_iterations (Q12)."""
    if not anneal:
        return jnp.asarray(lr0, jnp.float32)
    return lr0 * (1.0 - update0.astype(jnp.float32) / num_updates)


def set_lr(opt_state, lr):
    hp = dict(opt_state.hyperparams)
    hp["learning_rate"] = jnp.asarray(lr, hp["learning_rate"].dtype)
    return opt_state._replace(hyperparams=hp)
