"""Policy-gradient losses and the three baseline estimators (§8).

Everything here is a pure function of (params, batch) so it can be jitted, vmapped and
differentiated (tests/test_grads.py runs check_grads on these).

Baseline arms (they differ ONLY in how b_t is produced; the actor loss is shared):
  none b_t = 0
  state b_t = V_phi(omega_t) one net, Adam step on every batch
  multi b_t = V_{phi_i}(omega_t), i = index of the fixed input sequence the rollout used,
         computed with PRE-update phi_i (Alg. 2); then net i takes one Adam step
  maml one meta net phi; per input z the k rollouts are split in halves H1/H2;
         phi^1 = adapt(phi, H1), phi^2 = adapt(phi, H2) (J plain-SGD steps, App. K);
         b = V_{phi^2} on H1 and V_{phi^1} on H2 (cross-fitting, Alg. 1);
         meta step phi <- Adam(grad_phi [L_H2(adapt(phi,H1)) + L_H1(adapt(phi,H2))])
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------- returns
def discounted_returns(rewards, gamma: float):
    """G_t = sum_{t'>=t} gamma^(t'-t) r_t' over the last axis (lambda = 1, no bootstrap)."""
    def body(carry, r):
        g = r + gamma * carry
        return g, g

    r = jnp.moveaxis(rewards, -1, 0)
    _, g = jax.lax.scan(body, jnp.zeros(r.shape[1:], r.dtype), r, reverse=True)
    return jnp.moveaxis(g, 0, -1)


def entropy_coef(it, init: float, final: float, decay_iters: int):
    """Linear decay init -> final over decay_iters iterations, then constant (App. K)."""
    frac = jnp.clip(it.astype(jnp.float32) / float(decay_iters), 0.0, 1.0)
    return init + (final - init) * frac


# ---------------------------------------------------------------- actor
def actor_loss(apply_fn: Callable, params, obs, actions, adv, ent_coef, n_actions: int):
    """A2C: -mean_t[log pi(a_t|o_t) * sg(A_t)] - c_ent * mean_t[H(pi(.|o_t))] / log(K).

    obs [..., d], actions [...] int, adv [...] (treated as constant)."""
    logits = apply_fn(params, obs)
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, actions[..., None], axis=-1)[..., 0]
    ent = -jnp.sum(jnp.exp(logp_all) * logp_all, axis=-1) / jnp.log(float(n_actions))
    pg = -jnp.mean(logp * jax.lax.stop_gradient(adv))
    loss = pg - ent_coef * jnp.mean(ent)
    return loss, {"pg_loss": pg, "entropy": jnp.mean(ent)}


def pg_surrogate(apply_fn: Callable, params, obs, actions, adv):
    """Policy-gradient part only, sum over time: grad = sum_t psi_t A_t (one rollout's estimator)."""
    logits = apply_fn(params, obs)
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], axis=-1)[..., 0]
    return jnp.sum(logp * jax.lax.stop_gradient(adv))


def per_rollout_pg(apply_fn: Callable, params, obs, actions, adv):
    """Flattened per-rollout policy-gradient vectors [E, P] (obs [E,T,d], actions/adv [E,T])."""
    g = jax.vmap(jax.grad(lambda p, o, a, A: pg_surrogate(apply_fn, p, o, a, A)), in_axes=(None, 0, 0, 0))(
        params, obs, actions, adv)
    leaves = jax.tree_util.tree_leaves(g)
    return jnp.concatenate([x.reshape(x.shape[0], -1) for x in leaves], axis=1)


def pg_variance(vecs):
    """: trace of the sample covariance (ddof=1) of per-rollout gradient vectors [E, P]."""
    return jnp.sum(jnp.var(vecs, axis=0, ddof=1))


# ---------------------------------------------------------------- critic
def value_loss(apply_fn: Callable, params, x, G, reduction: str = "mean"):
    """squared error of V_phi(x) against Monte-Carlo returns G.

    reduction: "mean" over every element; "sum" over every element (the paper's Sigma);
    "sum_time" = sum over the time axis (last), mean over rollouts."""
    err = apply_fn(params, x) - G
    sq = err * err
    if reduction == "mean":
        return jnp.mean(sq)
    if reduction == "sum":
        return jnp.sum(sq)
    if reduction == "sum_time":
        return jnp.mean(jnp.sum(sq, axis=-1))
    raise ValueError(reduction)


def sgd_adapt(apply_fn: Callable, params, x, G, steps: int, lr: float, reduction: str, clip: float = 0.0):
    """J plain-SGD steps on L_H[V] (App. K inner loop); differentiable through (for 2nd-order MAML).
    clip > 0 rescales each inner gradient to global norm <= clip (ASSUMED stabiliser, off by default)."""
    def body(p, _):
        g = jax.grad(lambda q: value_loss(apply_fn, q, x, G, reduction))(p)
        if clip > 0:
            gn = optax.tree.norm(g)
            g = jax.tree_util.tree_map(lambda a: a * jnp.minimum(1.0, clip / (gn + 1e-12)), g)
        return jax.tree_util.tree_map(lambda a, b: a - lr * b, p, g), None

    p, _ = jax.lax.scan(body, params, None, length=steps)
    return p


def maml_crossfit(apply_fn: Callable, phi, x1, G1, x2, G2, *, steps: int, lr: float, inner_reduction: str,
                  outer_reduction: str = "mean", first_order: bool = True, clip: float = 0.0):
    """Alg. 1 for ONE input sequence.  x_h [k/2, T, d], G_h [k/2, T].

    Returns (b1, b2, meta_grad, aux): b1 = V_{phi^2}(x1) (baseline for half 1, fitted on half 2),
    b2 = V_{phi^1}(x2); meta_grad = grad_phi [L_H2(adapt(phi,H1)) + L_H1(adapt(phi,H2))]."""
    adapt = lambda p, x, G: sgd_adapt(apply_fn, p, x, G, steps, lr, inner_reduction, clip)
    phi1 = adapt(phi, x1, G1)
    phi2 = adapt(phi, x2, G2)
    b1 = apply_fn(phi2, x1)
    b2 = apply_fn(phi1, x2)

    def meta_loss(p):
        if first_order:  # FOMAML: gradient evaluated at the adapted point, no Hessian term
            q1 = jax.tree_util.tree_map(lambda a, b, c: a + jax.lax.stop_gradient(b - c), p, phi1, phi)
            q2 = jax.tree_util.tree_map(lambda a, b, c: a + jax.lax.stop_gradient(b - c), p, phi2, phi)
        else:
            q1, q2 = adapt(p, x1, G1), adapt(p, x2, G2)
        return value_loss(apply_fn, q1, x2, G2, outer_reduction) + value_loss(apply_fn, q2, x1, G1, outer_reduction)

    loss, grad = jax.value_and_grad(meta_loss)(phi)
    pre = 0.5 * (value_loss(apply_fn, phi, x1, G1, "mean") + value_loss(apply_fn, phi, x2, G2, "mean"))
    post = 0.5 * (jnp.mean((b1 - G1) ** 2) + jnp.mean((b2 - G2) ** 2))
    return b1, b2, grad, {"meta_loss": loss, "mse_unadapted": pre, "mse_crossfit": post}


def tree_select(mask, new, old):
    """Per-net select for stacked params / optimizer states: mask [N] bool."""
    def sel(a, b):
        if a.ndim == 0:
            return a
        m = mask.reshape((-1,) + (1,) * (a.ndim - 1))
        return jnp.where(m, a, b)

    return jax.tree_util.tree_map(sel, new, old)
