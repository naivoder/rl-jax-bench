"""Recurrent TD3 / SAC / SAC-Discrete with separate or shared RNN encoders, and the Markovian baseline.

All functions are pure and jit/vmap-able.  A gradient step takes explicit noise arrays (``draw_noise``) so the
differential tests can feed the PyTorch oracle's exact noise.

Sequence batch (time-major, official policy_rnn.update layout; README):
    obs  (L+1, B, O)   o[0..L]
    act  (L+1, B, A)   act[0] = 0, act[t] = a_{t-1} (one-hot for discrete)
    rew  (L+1, B, 1)   rew[0] = 0
    term (L+1, B, 1)   term[0] = 0
    mask (L,   B, 1)
Q is read at h[0..L-1] with the stored actions act[1..L]; targets use h[1..L] (the same unroll shifted by one).
Masked loss = sum(mask * err^2) / max(1, sum(mask)) per head (official).

Separate (official ModelFreeOffPolicy_Separate_RNN.forward): critic Adam step on L_Q1 + L_Q2 -> actor Adam step on
L_pi using the UPDATED critic -> Polyak (critic target; TD3 also actor target) -> alpha step.
Shared (official ModelFreeOffPolicy_Shared_RNN.forward): ONE Adam over embedders, RNN, both shortcuts, Q heads and
policy on 0.5 (L_Q1 + L_Q2) + L_pi; targets only for the heads (target uses the ONLINE hidden states, no_grad).
Two diagnostic coefficients (ours, E6; exactly 1/1 = official):
    leak  (lambda): scale of the actor-loss gradient into the Q heads and the obs-action shortcut
    enc   (kappa):  scale of the actor-loss gradient into the encoder (embedders + RNN)
shared_noleak = (0, 1), shared_detach = (0, 0).
Markov (official ModelFreeOffPolicy_MLP.update): critic step -> Polyak -> actor step -> alpha step, transitions.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax

from .model import Nets, categorical, tanh_gauss

sg = jax.lax.stop_gradient
F32 = jnp.float32


def mix(x, c):
    """forward value x, gradient scaled by c (c in {0,1} gives exactly stop_gradient / identity)."""
    return jax.tree_util.tree_map(lambda v: sg(v) + c * (v - sg(v)), x)


def polyak(tgt, src, tau):
    return jax.tree_util.tree_map(lambda t, s: t * (1.0 - tau) + s * tau, tgt, src)


def masked_mean(x, m, nv):
    return jnp.sum(x * m) / nv


def mean_tensor_norm(tree):
    """official utils.helpers.get_grad_norm: arithmetic mean over parameter tensors of each tensor's l2 norm."""
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.mean(jnp.stack([jnp.sqrt(jnp.sum(x ** 2)) for x in leaves]))


def global_norm(tree):
    return optax.tree.norm(tree)


class Agent:
    def __init__(self, cfg: dict, nets: Nets):
        self.cfg, self.nets = cfg, nets
        self.arch = cfg["arch"]            # separate | shared | markov
        self.algo = cfg["algo"]            # td3 | sac | sacd
        self.gamma = float(cfg["gamma"])
        self.tau = float(cfg["tau"])
        self.opt = optax.adam(float(cfg["lr"]), b1=0.9, b2=0.999, eps=1e-8)
        self.alpha_opt = optax.adam(float(cfg.get("alpha_lr", 3e-4)), b1=0.9, b2=0.999, eps=1e-8)
        self.auto_alpha = bool(cfg.get("auto_alpha", True)) and self.algo in ("sac", "sacd")
        self.fixed_alpha = float(cfg.get("entropy_alpha", 0.0)) if self.algo != "td3" else 0.0
        A = nets.act_dim
        if self.algo == "sac":
            self.target_entropy = -float(A)
        elif self.algo == "sacd":
            self.target_entropy = float(cfg.get("target_entropy_ratio", 0.7)) * math.log(A)
        else:
            self.target_entropy = 0.0
        self.expl_noise = float(cfg.get("exploration_noise", 0.1))
        self.tgt_noise = float(cfg.get("target_noise", 0.2))
        self.tgt_clip = float(cfg.get("target_noise_clip", 0.5))

    # ------------------------------------------------------------------ init
    def init(self, key):
        n = self.nets
        if self.arch == "shared":
            p = n.init_shared(key)
            params = {"main": p, "tgt": {"q1": p["q1"], "q2": p["q2"], "pi": p["pi"]}}
            opt = {"main": self.opt.init(p)}
        else:
            actor, critic = n.init_markov(key) if self.arch == "markov" else n.init_separate(key)
            params = {"actor": actor, "critic": critic, "actor_t": actor, "critic_t": critic}
            opt = {"actor": self.opt.init(actor), "critic": self.opt.init(critic)}
        log_alpha = jnp.zeros((), F32)  # official: log_alpha init 0 (alpha = 1)
        return {"params": params, "opt": opt, "log_alpha": log_alpha, "alpha_opt": self.alpha_opt.init(log_alpha)}

    def alpha(self, learner):
        if self.auto_alpha:
            return jnp.exp(learner["log_alpha"])
        return jnp.asarray(self.fixed_alpha, F32)

    # ------------------------------------------------------------------ noise
    def draw_noise(self, key, T, B):
        """standard-normal noise for the target action (TD3 smoothing / SAC target sample) and the actor sample."""
        k1, k2 = jax.random.split(key)
        A = self.nets.act_dim
        shape = (T, B, A) if T is not None else (B, A)
        return {"tgt": jax.random.normal(k1, shape, F32), "act": jax.random.normal(k2, shape, F32)}

    # ------------------------------------------------------------------ policy heads
    def _policy(self, out, eps):
        """head output -> (action or probs, log-prob or None)."""
        if self.algo == "td3":
            return jnp.tanh(out), None
        if self.algo == "sac":
            return tanh_gauss(out[0], out[1], eps)
        return categorical(out)

    def _td3_smooth(self, a, eps):
        noise = jnp.clip(eps * self.tgt_noise, -self.tgt_clip, self.tgt_clip)
        return jnp.clip(a + noise, -1.0, 1.0)

    def _next_v(self, q1t, q2t, a2, lp2, alpha):
        nq = jnp.minimum(q1t, q2t)
        if self.algo == "sac":
            nq = nq - alpha * lp2
        elif self.algo == "sacd":
            nq = jnp.sum(a2 * (nq - alpha * lp2), -1, keepdims=True)
        return nq

    def _gather(self, q, act):
        if self.algo == "sacd":
            return jnp.take_along_axis(q, jnp.argmax(act, -1)[..., None], -1)
        return q

    def _pi_loss_terms(self, q1, q2, an, lp, alpha):
        # TD3 actor on min(Q1, Q2) every step (no policy delay): official td3.py.  DEVIATION from the Fujimoto et al.
        # (2018) defaults Table 5 cites (actor on Q1, delay 2) -- sources.yaml td3_actor_q / td3_policy_delay.
        mq = jnp.minimum(q1, q2)
        if self.algo == "td3":
            return -mq, None
        if self.algo == "sac":
            return -mq + alpha * lp, lp
        pl = jnp.sum(an * (-mq + alpha * lp), -1, keepdims=True)
        return pl, jnp.sum(an * lp, -1, keepdims=True)

    # ------------------------------------------------------------------ separate
    def _critic_qs(self, cp, h, o, a):
        n = self.nets
        z = jnp.concatenate([h, n.short_oa_out(cp["short"], o, a)], -1)
        return n.q_out(cp["q1"], z), n.q_out(cp["q2"], z)

    def sep_critic_loss(self, cp, actor_x, critic_t, batch, noise, alpha):
        n = self.nets
        o, a, r, d, m = batch["obs"], batch["act"], batch["rew"], batch["term"], batch["mask"]
        actor_x, critic_t = sg(actor_x), sg(critic_t)
        # frozen target encoders: one no-grad scan for both; the online critic gets its own differentiated scan
        hqt, hpx = n.encode_stack([critic_t["enc"], actor_x["enc"]], a, r, o)
        hq, _ = n.encode(cp["enc"], a, r, o)
        zpi = jnp.concatenate([hpx, n.short_o_out(actor_x["short"], o)], -1)
        a2, lp2 = self._policy(n.pi_out(actor_x["pi"], zpi), noise["tgt"])
        if self.algo == "td3":
            a2 = self._td3_smooth(a2, noise["tgt"])
        q1t, q2t = self._critic_qs(critic_t, hqt, o, a2)
        y = sg(r + (1.0 - d) * self.gamma * self._next_v(q1t, q2t, a2, lp2, alpha))[1:]
        q1, q2 = self._critic_qs(cp, hq[:-1], o[:-1], a[1:])
        q1, q2 = self._gather(q1, a[1:]), self._gather(q2, a[1:])
        nv = jnp.maximum(1.0, jnp.sum(m))
        l1 = jnp.sum((q1 * m - y * m) ** 2) / nv
        l2 = jnp.sum((q2 * m - y * m) ** 2) / nv
        return l1 + l2, {"qf1_loss": l1, "qf2_loss": l2, "q1_mean": masked_mean(q1, m, nv)}

    def sep_actor_loss(self, ap, cp, batch, noise, alpha, critic_grad=False):
        n = self.nets
        o, a, r, m = batch["obs"], batch["act"], batch["rew"], batch["mask"]
        cpx = cp if critic_grad else sg(cp)
        ha, _ = n.encode(ap["enc"], a, r, o)
        hq, _ = n.encode(cpx["enc"], a, r, o)
        zpi = jnp.concatenate([ha, n.short_o_out(ap["short"], o)], -1)
        an, lp = self._policy(n.pi_out(ap["pi"], zpi), noise["act"])
        q1, q2 = self._critic_qs(cpx, hq, o, an)
        pl, lp_ent = self._pi_loss_terms(q1, q2, an, lp, alpha)
        nv = jnp.maximum(1.0, jnp.sum(m))
        loss = masked_mean(pl[:-1], m, nv)
        mean_lp = masked_mean(lp_ent[:-1], m, nv) if lp_ent is not None else jnp.zeros((), F32)
        return loss, {"policy_loss": loss, "mean_logp": sg(mean_lp)}

    def _alpha_step(self, learner, mean_lp):
        if not self.auto_alpha:
            return learner["log_alpha"], learner["alpha_opt"]
        la = learner["log_alpha"]
        g = -jnp.exp(la) * (mean_lp + self.target_entropy)  # d/dlog_alpha of -exp(log_alpha)(E log pi + H)
        upd, st = self.alpha_opt.update(g, learner["alpha_opt"], la)
        return optax.apply_updates(la, upd), st

    def update_separate(self, learner, batch, noise, coef, diag: bool):
        P, O = learner["params"], learner["opt"]
        alpha = self.alpha(learner)
        actor_x = P["actor_t"] if self.algo == "td3" else P["actor"]
        (lq, aq), gq = jax.value_and_grad(self.sep_critic_loss, has_aux=True)(
            P["critic"], actor_x, P["critic_t"], batch, noise, alpha)
        upd, oc = self.opt.update(gq, O["critic"], P["critic"])
        critic = optax.apply_updates(P["critic"], upd)
        if diag:
            (lpi, api), (ga, gc2) = jax.value_and_grad(self.sep_actor_loss, argnums=(0, 1), has_aux=True)(
                P["actor"], critic, batch, noise, alpha, True)
        else:
            (lpi, api), ga = jax.value_and_grad(self.sep_actor_loss, has_aux=True)(
                P["actor"], critic, batch, noise, alpha, False)
        upd, oa = self.opt.update(ga, O["actor"], P["actor"])
        actor = optax.apply_updates(P["actor"], upd)
        critic_t = polyak(P["critic_t"], critic, self.tau)
        actor_t = polyak(P["actor_t"], actor, self.tau) if self.algo == "td3" else P["actor_t"]
        la, ast = self._alpha_step(learner, api["mean_logp"])
        new = {"params": {"actor": actor, "critic": critic, "actor_t": actor_t, "critic_t": critic_t},
               "opt": {"actor": oa, "critic": oc}, "log_alpha": la, "alpha_opt": ast}
        info = {**aq, **api, "alpha": alpha, "entropy": -api["mean_logp"]}
        if diag:
            gtot_c = jax.tree_util.tree_map(jnp.add, gq, gc2)  # torch .grad of the critic after the actor backward
            info.update({
                "off_q_rnn": mean_tensor_norm(gtot_c["enc"]["rnn"]), "off_q": mean_tensor_norm(gtot_c),
                "off_pi_rnn": mean_tensor_norm(ga["enc"]["rnn"]), "off_pi": mean_tensor_norm(ga),
                "clean_q_rnn": global_norm(gq["enc"]["rnn"]), "clean_pi_rnn": global_norm(ga["enc"]["rnn"]),
                "clean_q_enc": global_norm(gq["enc"]), "clean_pi_enc": global_norm(ga["enc"]),
                "clean_pi_into_q_rnn": global_norm(gc2["enc"]["rnn"]),
                # the paper's caption definition (Fig. 6/21): squared l2 norm of the whole RNN gradient
                "off_sq_q_rnn": global_norm(gtot_c["enc"]["rnn"]) ** 2,
                "off_sq_pi_rnn": global_norm(ga["enc"]["rnn"]) ** 2})
        return new, info

    # ------------------------------------------------------------------ shared
    def shared_losses(self, p, tgt, batch, noise, alpha, coef):
        n = self.nets
        o, a, r, d, m = batch["obs"], batch["act"], batch["rew"], batch["term"], batch["mask"]
        h, _ = n.encode(p["enc"], a, r, o)
        oe = n.short_o_out(p["short_o"], o)
        jp = jnp.concatenate([h, oe], -1)
        # --- target (official: torch.no_grad, online hidden states, target heads, ONLINE obs-action shortcut)
        jps, hs = sg(jp), sg(h)
        if self.algo == "td3":
            a2 = self._td3_smooth(jnp.tanh(n.pi_out(tgt["pi"], jps)), noise["tgt"])
            lp2 = None
        else:
            a2, lp2 = self._policy(n.pi_out(sg(p["pi"]), jps), noise["tgt"])
        z2 = jnp.concatenate([hs, n.short_oa_out(sg(p["short_oa"]), o, a2)], -1)
        nq = self._next_v(n.q_out(tgt["q1"], z2), n.q_out(tgt["q2"], z2), a2, lp2, alpha)
        y = sg(r + (1.0 - d) * self.gamma * nq)[1:]
        z = jnp.concatenate([h[:-1], n.short_oa_out(p["short_oa"], o[:-1], a[1:])], -1)
        q1, q2 = n.q_out(p["q1"], z), n.q_out(p["q2"], z)
        nv = jnp.maximum(1.0, jnp.sum(m))
        l1 = jnp.sum((q1 * m - y * m) ** 2) / nv
        l2 = jnp.sum((q2 * m - y * m) ** 2) / nv
        # --- actor
        h_a = mix(h, coef["enc"])
        an, lp = self._policy(n.pi_out(p["pi"], jnp.concatenate([h_a, oe], -1)), noise["act"])
        pq = mix({"short_oa": p["short_oa"], "q1": p["q1"], "q2": p["q2"]}, coef["leak"])
        zn = jnp.concatenate([h_a, n.short_oa_out(pq["short_oa"], o, an)], -1)
        pl, lp_ent = self._pi_loss_terms(n.q_out(pq["q1"], zn), n.q_out(pq["q2"], zn), an, lp, alpha)
        lpi = masked_mean(pl[:-1], m, nv)
        mean_lp = masked_mean(lp_ent[:-1], m, nv) if lp_ent is not None else jnp.zeros((), F32)
        aux = {"qf1_loss": l1, "qf2_loss": l2, "policy_loss": lpi, "q1_mean": masked_mean(q1, m, nv),
               "mean_logp": sg(mean_lp)}
        return 0.5 * (l1 + l2), lpi, aux

    def update_shared(self, learner, batch, noise, coef, diag: bool):
        P, O = learner["params"], learner["opt"]
        alpha = self.alpha(learner)
        tgt = sg(P["tgt"])
        if diag:
            def f(p):
                lq, lpi, aux = self.shared_losses(p, tgt, batch, noise, alpha, coef)
                return (lq, lpi), aux
            (lq, lpi), vjp, aux = jax.vjp(f, P["main"], has_aux=True)
            one, zero = jnp.ones((), F32), jnp.zeros((), F32)
            (g_q,) = vjp((one, zero))
            (g_pi,) = vjp((zero, one))
            g = jax.tree_util.tree_map(jnp.add, g_q, g_pi)
        else:
            def f(p):
                lq, lpi, aux = self.shared_losses(p, tgt, batch, noise, alpha, coef)
                return lq + lpi, aux
            (_, aux), g = jax.value_and_grad(f, has_aux=True)(P["main"])
        upd, om = self.opt.update(g, O["main"], P["main"])
        main = optax.apply_updates(P["main"], upd)
        new_tgt = {"q1": polyak(P["tgt"]["q1"], main["q1"], self.tau),
                   "q2": polyak(P["tgt"]["q2"], main["q2"], self.tau),
                   "pi": polyak(P["tgt"]["pi"], main["pi"], self.tau) if self.algo == "td3" else P["tgt"]["pi"]}
        la, ast = self._alpha_step(learner, aux["mean_logp"])
        new = {"params": {"main": main, "tgt": new_tgt}, "opt": {"main": om}, "log_alpha": la, "alpha_opt": ast}
        info = {**aux, "alpha": alpha, "entropy": -aux["mean_logp"]}
        if diag:
            info.update({
                "off_rnn": mean_tensor_norm(g["enc"]["rnn"]), "off_q": mean_tensor_norm(g["q1"]),
                "off_pi": mean_tensor_norm(g["pi"]),
                "clean_q_rnn": global_norm(g_q["enc"]["rnn"]), "clean_pi_rnn": global_norm(g_pi["enc"]["rnn"]),
                "clean_q_enc": global_norm(g_q["enc"]), "clean_pi_enc": global_norm(g_pi["enc"]),
                "clean_pi_into_q_heads": global_norm({"q1": g_pi["q1"], "q2": g_pi["q2"],
                                                      "s": g_pi["short_oa"]}),
                # the paper's caption definition (Fig. 6/21): squared l2 norm of the whole RNN gradient
                "off_sq_rnn": global_norm(g["enc"]["rnn"]) ** 2})
        return new, info

    # ------------------------------------------------------------------ markov
    def _mq(self, qp, o, a):
        x = o if self.algo == "sacd" else jnp.concatenate([o, a], -1)
        return self.nets.q_out(qp, x)

    def mk_critic_loss(self, cp, actor_x, critic_t, b, noise, alpha):
        n = self.nets
        actor_x, critic_t = sg(actor_x), sg(critic_t)
        a2, lp2 = self._policy(n.pi_out(actor_x["pi"], b["obs2"]), noise["tgt"])
        if self.algo == "td3":
            a2 = self._td3_smooth(a2, noise["tgt"])
        nq = self._next_v(self._mq(critic_t["q1"], b["obs2"], a2), self._mq(critic_t["q2"], b["obs2"], a2),
                          a2, lp2, alpha)
        y = sg(b["rew"] + (1.0 - b["term"]) * self.gamma * nq)
        q1 = self._gather(self._mq(cp["q1"], b["obs"], b["act"]), b["act"])
        q2 = self._gather(self._mq(cp["q2"], b["obs"], b["act"]), b["act"])
        l1, l2 = jnp.mean((q1 - y) ** 2), jnp.mean((q2 - y) ** 2)
        return l1 + l2, {"qf1_loss": l1, "qf2_loss": l2, "q1_mean": jnp.mean(q1)}

    def mk_actor_loss(self, ap, cp, b, noise, alpha):
        n = self.nets
        cp = sg(cp)
        an, lp = self._policy(n.pi_out(ap["pi"], b["obs"]), noise["act"])
        pl, lp_ent = self._pi_loss_terms(self._mq(cp["q1"], b["obs"], an), self._mq(cp["q2"], b["obs"], an),
                                         an, lp, alpha)
        loss = jnp.mean(pl)
        mean_lp = jnp.mean(lp_ent) if lp_ent is not None else jnp.zeros((), F32)
        return loss, {"policy_loss": loss, "mean_logp": sg(mean_lp)}

    def update_markov(self, learner, b, noise, coef, diag: bool):
        P, O = learner["params"], learner["opt"]
        alpha = self.alpha(learner)
        actor_x = P["actor_t"] if self.algo == "td3" else P["actor"]
        (_, aq), gq = jax.value_and_grad(self.mk_critic_loss, has_aux=True)(
            P["critic"], actor_x, P["critic_t"], b, noise, alpha)
        upd, oc = self.opt.update(gq, O["critic"], P["critic"])
        critic = optax.apply_updates(P["critic"], upd)
        critic_t = polyak(P["critic_t"], critic, self.tau)            # official MLP: soft update BEFORE actor step
        actor_t = polyak(P["actor_t"], P["actor"], self.tau) if self.algo == "td3" else P["actor_t"]
        (_, api), ga = jax.value_and_grad(self.mk_actor_loss, has_aux=True)(P["actor"], critic, b, noise, alpha)
        upd, oa = self.opt.update(ga, O["actor"], P["actor"])
        actor = optax.apply_updates(P["actor"], upd)
        la, ast = self._alpha_step(learner, api["mean_logp"])
        new = {"params": {"actor": actor, "critic": critic, "actor_t": actor_t, "critic_t": critic_t},
               "opt": {"actor": oa, "critic": oc}, "log_alpha": la, "alpha_opt": ast}
        return new, {**aq, **api, "alpha": alpha, "entropy": -api["mean_logp"]}

    def update(self, learner, batch, noise, coef, diag=False):
        if self.arch == "separate":
            return self.update_separate(learner, batch, noise, coef, diag)
        if self.arch == "shared":
            return self.update_shared(learner, batch, noise, coef, diag)
        return self.update_markov(learner, batch, noise, coef, diag)

    # ------------------------------------------------------------------ acting
    def actor_parts(self, learner):
        P = learner["params"]
        if self.arch == "shared":
            return P["main"]["enc"], P["main"]["short_o"], P["main"]["pi"]
        if self.arch == "markov":
            return None, None, P["actor"]["pi"]
        return P["actor"]["enc"], P["actor"]["short"], P["actor"]["pi"]

    def init_carry(self, B):
        from .model import zero_carry
        return zero_carry(self.nets.cell, (B,), self.nets.H)

    def act(self, learner, carry, prev_a, r, o, key, deterministic: bool):
        """one environment step for a batch of B episodes (prev_a (B,A), r (B,1), o (B,O))."""
        n = self.nets
        enc_p, short_p, pi_p = self.actor_parts(learner)
        if self.arch == "markov":
            z = o
        else:
            h, carry = n.encode(enc_p, prev_a[None], r[None], o[None], carry)
            z = jnp.concatenate([h[0], n.short_o_out(short_p, o)], -1)
        out = n.pi_out(pi_p, z)
        B, A = o.shape[0], n.act_dim
        if self.algo == "td3":
            mu = jnp.tanh(out)
            if deterministic:
                return mu, carry
            return jnp.clip(mu + jax.random.normal(key, (B, A)) * self.expl_noise, -1.0, 1.0), carry
        if self.algo == "sac":
            if deterministic:
                return jnp.tanh(out[0]), carry
            a, _ = tanh_gauss(out[0], out[1], jax.random.normal(key, (B, A)))
            return a, carry
        if deterministic:
            return jax.nn.one_hot(jnp.argmax(out, -1), A), carry
        return jax.nn.one_hot(jax.random.categorical(key, out, -1), A), carry
