"""ADVISOR and every baseline loss as pure functions (paper Eqs. 1-3, Alg. A.1; official advisor_losses.py and
AllenAct v0.5.0 PPO / Imitation / RolloutStorage).

All arms share one rollout/update code path (train.py); an arm is a sequence of stages, each stage a loss MODE:
  'ppo'       : L_PPO = mean(clip surrogate) + 0.5 * mean(clipped value loss) + 0.01 * mean(-entropy)
  'imitation' : masked-mean CE(teacher label, pi)                                  (BC / DAgger / BC^tf=1 stages)
  'bc_ppo'    : imitation + L_PPO, weights 1 and 1 ("BC+PPO static", App. A.5.7)
  'advisor_imitation': masked-mean CE on the main head + masked-mean CE on the aux head (official
                AdvisorImitationStage: the warm-up stage of DAgger->ADV and BC^tf=1->ADV trains BOTH heads)
  'advisor'   : aux CE (masked mean) + mean(-w * log pi(a*)) + mean((1-w) * surrogate)
                + 0.01 * mean((1-w) * (-entropy)) + 0.5 * mean(value loss),
                w = exp(alpha * log pi_aux(a*)) * valid * 1[pi_aux(a*) >= beta], stop-gradient   (Eqs. 2-3)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

MODES = ("ppo", "imitation", "bc_ppo", "advisor", "advisor_imitation")
MODE_ID = {m: i for i, m in enumerate(MODES)}


def gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """AllenAct RolloutStorage.compute_returns with use_gae: arrays [T, B]; dones[t] = episode ended at step t.
    delta_t = r_t + gamma V_{t+1} (1 - d_t) - V_t ; A_t = delta_t + gamma lam (1 - d_t) A_{t+1}; returns = A + V."""
    def body(carry, x):
        adv_next, v_next = carry
        r, v, d = x
        nd = 1.0 - d.astype(r.dtype)
        delta = r + gamma * v_next * nd - v
        adv = delta + gamma * lam * nd * adv_next
        return (adv, v), adv
    _, adv = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value), (rewards, values, dones), reverse=True)
    return adv, adv + values


def normalise(adv):
    """official: (A - mean) / (std + 1e-5) over the whole rollout (torch std is unbiased)."""
    n = adv.size
    m = adv.mean()
    sd = jnp.sqrt(jnp.sum((adv - m) ** 2) / (n - 1))
    return (adv - m) / (sd + 1e-5)


def log_softmax(z):
    return jax.nn.log_softmax(z, axis=-1)


def take(logp, a):
    return jnp.take_along_axis(logp, jnp.clip(a, 0, logp.shape[-1] - 1)[..., None], axis=-1)[..., 0]


def per_step_terms(logits, aux_logits, values, batch, clip):
    """Per-step loss pieces on one minibatch (arrays [T, B])."""
    logp = log_softmax(logits)
    ent = -jnp.sum(jnp.exp(logp) * logp, -1)
    lp_a = take(logp, batch["action"])
    ratio = jnp.exp(lp_a - batch["logp"])
    adv = batch["adv_norm"]
    surr1 = ratio * adv
    surr2 = jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * adv
    action_loss = -jnp.where(surr2 < surr1, surr2, surr1)
    v_clip = batch["value"] + jnp.clip(values - batch["value"], -clip, clip)
    value_loss = 0.5 * jnp.maximum((values - batch["ret"]) ** 2, (v_clip - batch["ret"]) ** 2)
    valid = batch["label_valid"].astype(jnp.float32)
    lp_label = take(logp, batch["label"])
    lp_aux_label = take(log_softmax(aux_logits), batch["label"])
    return dict(action_loss=action_loss, value_loss=value_loss, neg_entropy=-ent, lp_label=lp_label,
                lp_aux_label=lp_aux_label, valid=valid, ratio=ratio)


def advisor_weight(lp_aux_label, valid, alpha, beta):
    """w = exp(alpha * log pi_aux(a*)) * valid * 1[log pi_aux(a*) >= log beta] (beta <= 0: no gate), detached."""
    gate = jnp.where(beta > 0, lp_aux_label >= jnp.log(jnp.maximum(beta, 1e-30)), True)
    return jax.lax.stop_gradient(jnp.exp(alpha * lp_aux_label) * valid * gate)


def masked_mean(x, m):
    return jnp.sum(x * m) / jnp.maximum(jnp.sum(m), 1.0)


def loss_from_terms(p, mode_id, alpha, beta, value_coef=0.5, entropy_coef=0.01):
    """Total loss for a stage mode (traced int mode_id) from per_step_terms output."""
    ppo = p["action_loss"].mean() + value_coef * p["value_loss"].mean() + entropy_coef * p["neg_entropy"].mean()
    imit = -masked_mean(p["lp_label"], p["valid"])
    w = advisor_weight(p["lp_aux_label"], p["valid"], alpha, beta)
    aux_ce = -masked_mean(p["lp_aux_label"], p["valid"])
    adv = (aux_ce + (-(w * p["lp_label"]).mean()) + value_coef * p["value_loss"].mean()
           + ((1.0 - w) * p["action_loss"]).mean() + entropy_coef * ((1.0 - w) * p["neg_entropy"]).mean())
    total = jnp.select([mode_id == 0, mode_id == 1, mode_id == 2, mode_id == 3, mode_id == 4],
                       [ppo, imit, imit + ppo, adv, imit + aux_ce])
    stats = dict(ppo=ppo, imit=imit, aux_ce=aux_ce, w_mean=masked_mean(w, p["valid"]),
                 entropy=-p["neg_entropy"].mean(), value_loss=p["value_loss"].mean())
    return total, stats
