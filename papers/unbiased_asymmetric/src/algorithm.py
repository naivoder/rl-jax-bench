"""A2C family of Baisero & Amato (2022) as pure JAX functions (Alg. 1 + App. E + asym-rlpo@1a065b0 semantics).

One update (per run; vmapped over runs by train.py):
  1. sample E=2 episodes with the stochastic policy (cap 100 steps)         [App. E]
  2. push their undiscounted returns into the avg-100 window; log if the log dispenser fires on the PRE-update
     env-step count                                                          [official main_a2c.py]
  3. t_env += lengths; lambda = linear(lambda0 -> lambda0/10 over 2M)        [App. E]
  4. if t_env >= next_target: target critic <- critic; next_target = t_env + 10k    [App. E + official Dispenser]
  5. critic step: L_c = mean_ep sum_t (V(x_t) - [r_t + g V_target(x_{t+1})])^2      [eqs 6/7/22, semi-gradient]
  6. actor step with the UPDATED online critic:
     L_a = mean_ep sum_t g^t (r_t + g V(x_{t+1}) - V(x_t)) (-log pi(a_t|h_t)) - lambda mean_ep sum_t H(pi(.|h_t))
     [eqs 5/21; official actor_losses]
Time-limit truncation bootstraps 0 (official td0_q_estimator); flag ``bootstrap_truncation`` is the Rule-1
diagnostic. ``td: tdlambda`` is the official td-lambda estimator (diagnostic).
The ONLY difference between the arms is the critic input x_t (model.METHODS) and, for react-k, the history window.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from model import Actor, Critic, build


class Episode(NamedTuple):
    states: jnp.ndarray   # [C+1]   s_0..s_C  (position L = state after the last step)
    obs: jnp.ndarray      # [C+1]   o_0 (reset obs)..o_C
    actions: jnp.ndarray  # [C+1]   a_0..a_{C-1}, dummy 0 at C
    rewards: jnp.ndarray  # [C]
    mask: jnp.ndarray     # [C]     t < L
    trunc: jnp.ndarray    # []      episode ended by the time limit (not by the env)
    length: jnp.ndarray   # []      L


def entropy_weight(t, lam0, final_factor, nsteps):
    # official linear_schedule: u = clip(t / (nsteps - 1), 0, 1)
    u = jnp.clip(t / (nsteps - 1.0), 0.0, 1.0)
    return lam0 * (1.0 - u) + lam0 * final_factor * u


def make_rollout(envf, spec, method, cap):
    actor, _ = build(spec, method)

    def episode(pa, key):
        kr, ks = jax.random.split(key)
        s0, o0 = envf.reset(kr)
        carry0 = actor.apply({"params": pa}, method=Actor.init_carry)

        def body(c, t):
            carry, s, o, prev_a, alive = c
            carry, logp = actor.apply({"params": pa}, carry, prev_a, o, method=Actor.step)
            ka, ke = jax.random.split(jax.random.fold_in(ks, t))
            a = jax.random.categorical(ka, logp).astype(jnp.int32)
            s2, o2, r, d = envf.step(ke, s, a)
            return (carry, s2, o2, a, alive & ~d), (s, o, a, r, alive)

        (_, sC, oC, _, aliveC), (S, O, A, R, M) = jax.lax.scan(
            body, (carry0, s0, o0, jnp.int32(-1), jnp.bool_(True)), jnp.arange(cap))
        cat = lambda x, y: jnp.concatenate([x, jnp.asarray(y, x.dtype)[None]], 0)  # noqa: E731
        return Episode(states=cat(S, sC), obs=cat(O, oC), actions=cat(A, 0), rewards=R, mask=M,
                       trunc=aliveC, length=M.sum().astype(jnp.int32))

    return episode


def td_targets(ep: Episode, v, gamma, td="td0", lam=0.7, bootstrap_truncation=False):
    """v: [C+1] critic values on the episode positions. Returns targets y_t for t < C (masked later)."""
    C = ep.rewards.shape[0]
    nxt_valid = jnp.concatenate([ep.mask[1:], jnp.zeros((1,), bool)])
    if bootstrap_truncation:  # Rule-1 diagnostic: bootstrap at the time limit instead of treating it as terminal
        nxt_valid = nxt_valid | (ep.trunc & (jnp.arange(C) == ep.length - 1))
    vn = jnp.where(nxt_valid, v[1:], 0.0)
    if td == "td0":
        return ep.rewards + gamma * vn
    # official tdlambda_q_estimator: Q_t = sum_{k>=t} (g lam)^{k-t} (r_k + g (1 - lam) V_{k+1}), zero past the end
    inc = jnp.where(ep.mask, ep.rewards + gamma * (1.0 - lam) * vn, 0.0)

    def back(g, x):
        g = x + gamma * lam * g
        return g, g

    _, q = jax.lax.scan(back, jnp.zeros((), inc.dtype), inc, reverse=True)
    return q


def pg_terms(logp, ep: Episode, v, gamma, td="td0", td_lambda=0.7, bootstrap_truncation=False):
    """Per-episode actor terms (official actor_losses): policy term sum_t g^t sg(y_t - V(x_t)) (-log pi(a_t|h_t)),
    negentropy -sum_t H(pi(.|h_t)), and the mean per-step entropy. logp: [C, A]; v: [C+1] critic values (no grad).
    Used by the training loss AND by the toy check Monte-Carlo check (toys.py), so toy check exercises this code."""
    adv = td_targets(ep, v, gamma, td, td_lambda, bootstrap_truncation) - v[:-1]
    nll = -jnp.take_along_axis(logp, ep.actions[:-1, None], -1)[:, 0]
    disc = gamma ** jnp.arange(ep.rewards.shape[0], dtype=ep.rewards.dtype)
    pol = jnp.sum(jnp.where(ep.mask, disc * adv * nll, 0.0))
    ent = -jnp.sum(jnp.where(logp > -jnp.inf, jnp.exp(logp) * logp, 0.0), -1)
    negent = -jnp.sum(jnp.where(ep.mask, ent, 0.0))
    return pol, negent, jnp.sum(jnp.where(ep.mask, ent, 0.0)) / jnp.maximum(ep.length, 1)


def make_losses(spec, method, gamma, td="td0", td_lambda=0.7, bootstrap_truncation=False):
    actor, critic = build(spec, method)

    def values(pc, ep):
        return critic.apply({"params": pc}, ep.actions, ep.obs, ep.states)

    def targets(ep, v):
        return td_targets(ep, v, gamma, td, td_lambda, bootstrap_truncation)

    def critic_loss(pc, pt, eps: Episode):
        def one(ep):
            v = values(pc, ep)
            y = jax.lax.stop_gradient(targets(ep, values(pt, ep)))
            return jnp.sum(jnp.where(ep.mask, (v[:-1] - y) ** 2, 0.0))
        return jnp.mean(jax.vmap(one)(eps))

    def actor_loss(pa, pc, eps: Episode, lam_ent):
        def one(ep):
            v = jax.lax.stop_gradient(values(pc, ep))
            logp = actor.apply({"params": pa}, ep.actions, ep.obs)[:-1]       # [C, A]
            return pg_terms(logp, ep, v, gamma, td, td_lambda, bootstrap_truncation)
        pol, negent, ment = jax.vmap(one)(eps)
        loss = jnp.mean(pol) + lam_ent * jnp.mean(negent)
        return loss, (jnp.mean(pol), jnp.mean(negent), jnp.mean(ment))

    return values, critic_loss, actor_loss


class RunState(NamedTuple):
    pa: dict
    pc: dict
    pt: dict
    oa: optax.OptState
    oc: optax.OptState
    key: jnp.ndarray        # loop key (per run)
    upd: jnp.ndarray        # update counter
    t_env: jnp.ndarray      # env steps so far (official simulation_timesteps)
    next_target: jnp.ndarray
    next_log: jnp.ndarray
    log_idx: jnp.ndarray
    win: jnp.ndarray        # [100] returns window
    win_n: jnp.ndarray
    bad: jnp.ndarray        # any non-finite loss seen
    logs: dict              # per log point arrays [NL, ...]


def make_optimizers(alpha_pi, alpha_v, eps):
    # torch.optim.Adam(lr, eps=1e-4), default betas; separate actor / critic optimizers; no clipping (official)
    return optax.adam(alpha_pi, eps=eps), optax.adam(alpha_v, eps=eps)


def init_run_state(pa, pc, key, cfg, n_probes=0):
    opt_a, opt_c = make_optimizers(cfg["alpha_pi"], cfg["alpha_v"], cfg["adam_eps"])
    NL = cfg["n_log"] + 2
    logs = dict(t=jnp.full((NL,), -1, jnp.int32), avg100=jnp.zeros((NL,)), mean_len=jnp.zeros((NL,)),
                critic_loss=jnp.zeros((NL,)), actor_pol=jnp.zeros((NL,)), entropy=jnp.zeros((NL,)),
                lam=jnp.zeros((NL,)), probe=jnp.zeros((NL, max(n_probes, 1))))
    z = jnp.int32(0)
    return RunState(pa=pa, pc=pc, pt=pc, oa=opt_a.init(pa), oc=opt_c.init(pc), key=key, upd=z, t_env=z,
                    next_target=z, next_log=z, log_idx=z, win=jnp.zeros((100,)), win_n=z, bad=jnp.bool_(False),
                    logs=logs)


def make_rollout_batched(envf, spec, method, cap):
    """Rollout of E episodes for each of R runs, as ONE lax.while_loop that stops as soon as every episode of the
    batch has ended (instead of always stepping `cap` times). Same key tree as `make_rollout` (per-episode key
    -> (reset key, step key); step t uses fold_in(step key, t)), so both produce identical episodes
    (tests/test_smoke.py checks this)."""
    actor, _ = build(spec, method)
    step_pol = jax.vmap(jax.vmap(lambda pa, c, pr, o: actor.apply({"params": pa}, c, pr, o, method=Actor.step),
                                 (None, 0, 0, 0)), (0, 0, 0, 0))

    def rollout(pa, keys, active):
        R, E = keys.shape[:2]
        kr, ks = jnp.moveaxis(jax.vmap(jax.vmap(jax.random.split))(keys), 2, 0)
        s0, o0 = jax.vmap(jax.vmap(envf.reset))(kr)
        carry0 = actor.apply({"params": jax.tree.map(lambda x: x[0], pa)}, method=Actor.init_carry)
        carry0 = jnp.broadcast_to(carry0, (R, E) + carry0.shape)
        S = jnp.zeros((R, E, cap + 1) + s0.shape[2:], s0.dtype).at[:, :, 0].set(s0)
        O = jnp.zeros((R, E, cap + 1) + o0.shape[2:], o0.dtype).at[:, :, 0].set(o0)
        A = jnp.zeros((R, E, cap + 1), jnp.int32)
        Rw = jnp.zeros((R, E, cap), jnp.float32)
        M = jnp.zeros((R, E, cap), bool)
        alive0 = jnp.broadcast_to(active[:, None], (R, E))
        prev0 = jnp.full((R, E), -1, jnp.int32)

        def cond(c):
            return (c[0] < cap) & jnp.any(c[5])

        def body(c):
            t, carry, s, o, prev, alive, S, O, A, Rw, M = c
            carry, logp = step_pol(pa, carry, prev, o)
            kk = jax.vmap(jax.vmap(lambda k: jax.random.split(jax.random.fold_in(k, t))))(ks)
            a = jax.vmap(jax.vmap(jax.random.categorical))(kk[:, :, 0], logp).astype(jnp.int32)
            s2, o2, r, d = jax.vmap(jax.vmap(envf.step))(kk[:, :, 1], s, a)
            S = jax.lax.dynamic_update_index_in_dim(S, s2, t + 1, 2)
            O = jax.lax.dynamic_update_index_in_dim(O, o2, t + 1, 2)
            A = jax.lax.dynamic_update_index_in_dim(A, a, t, 2)
            Rw = jax.lax.dynamic_update_index_in_dim(Rw, r.astype(jnp.float32), t, 2)
            M = jax.lax.dynamic_update_index_in_dim(M, alive, t, 2)
            return (t + 1, carry, s2, o2, a, alive & ~d, S, O, A, Rw, M)

        c = jax.lax.while_loop(cond, body, (jnp.int32(0), carry0, s0, o0, prev0, alive0, S, O, A, Rw, M))
        _, _, _, _, _, alive, S, O, A, Rw, M = c
        length = M.sum(-1).astype(jnp.int32)
        return Episode(states=S, obs=O, actions=A, rewards=Rw, mask=M, trunc=alive & (length == cap),
                       length=length)

    return rollout


def slice_episodes(eps: Episode, L: int) -> Episode:
    """Keep positions 0..L (L steps + the position after the last step); valid when every episode has length <= L.
    The time axis is the LAST axis of `actions`; vector-feature envs (Car-Flag, Cleaner) carry a trailing feature
    axis on states/obs, so slice by the time axis index, never by `...` (the full-scale pre-launch smoke caught that)."""
    ta = eps.actions.ndim - 1

    def cut(x, n):
        return jax.lax.slice_in_dim(x, 0, n, axis=ta)
    return Episode(states=cut(eps.states, L + 1), obs=cut(eps.obs, L + 1), actions=cut(eps.actions, L + 1),
                   rewards=cut(eps.rewards, L), mask=cut(eps.mask, L), trunc=eps.trunc, length=eps.length)


def make_update(envf, spec, method, cfg, probes=None):
    """Returns update(rs) -> rs over a BATCH of runs (leading axis R). cfg: scalars (configs/*.yaml).

    The rollout stops when every episode in the batch has ended; the losses are then evaluated on sequences
    padded only to the smallest length bucket >= the longest episode (lax.switch over static buckets). Padding
    is masked, so results are identical to padding to `cap` (tests/test_smoke.py)."""
    gamma, cap, E = cfg["gamma"], cfg["cap"], cfg["episodes_per_update"]
    T_max, period, log_period = cfg["max_env_steps"], cfg["target_period"], cfg["max_env_steps"] // cfg["n_log"]
    buckets = sorted({b for b in cfg.get("length_buckets", [16, 32, 64]) if b < cap} | {cap})
    opt_a, opt_c = make_optimizers(cfg["alpha_pi"], cfg["alpha_v"], cfg["adam_eps"])
    rollout = make_rollout_batched(envf, spec, method, cap)
    values, critic_loss, actor_loss = make_losses(spec, method, gamma, cfg.get("td", "td0"),
                                                  cfg.get("td_lambda", 0.7), cfg.get("bootstrap_truncation", False))
    actor_uses_target = cfg.get("actor_uses_target", False)  # Rule-1 diagnostic (official: online critic)
    critic_mod = Critic(spec, *_mode_k(method))

    def probe_values(pc):
        if probes is None:
            return jnp.zeros((1,))
        def one(a, o, s, i):
            return critic_mod.apply({"params": pc}, a, o, s)[i]
        return jax.vmap(one)(probes["actions"], probes["obs"], probes["states"], probes["idx"])

    def learn(rs: RunState, eps: Episode) -> RunState:  # ONE run; eps: [E, ...] sliced to a bucket
        active = rs.t_env < T_max
        rets = jnp.sum(jnp.where(eps.mask, eps.rewards, 0.0), -1)
        win, win_n = rs.win, rs.win_n
        for e in range(E):
            win = win.at[win_n % 100].set(rets[e])
            win_n = win_n + 1
        nwin = jnp.minimum(win_n, 100)
        avg100 = jnp.sum(jnp.where(jnp.arange(100) < nwin, win, 0.0)) / jnp.maximum(nwin, 1)
        do_log = active & (rs.t_env >= rs.next_log) & (rs.log_idx < rs.logs["t"].shape[0])
        next_log = jnp.where(do_log, rs.t_env + log_period, rs.next_log)

        t_env = rs.t_env + jnp.sum(eps.length)
        lam_ent = entropy_weight(t_env, cfg["lambda0"], cfg["lambda_final_factor"], cfg["negentropy_nsteps"])
        refresh = t_env >= rs.next_target
        pt = jax.tree.map(lambda c, t: jnp.where(refresh, c, t), rs.pc, rs.pt)
        next_target = jnp.where(refresh, t_env + period, rs.next_target)

        lc, gc = jax.value_and_grad(critic_loss)(rs.pc, pt, eps)
        uc, oc = opt_c.update(gc, rs.oc, rs.pc)
        pc = optax.apply_updates(rs.pc, uc)

        (la, (lpol, lneg, ment)), ga = jax.value_and_grad(actor_loss, has_aux=True)(
            rs.pa, pt if actor_uses_target else pc, eps, lam_ent)
        ua, oa = opt_a.update(ga, rs.oa, rs.pa)
        pa = optax.apply_updates(rs.pa, ua)
        bad = rs.bad | ~jnp.isfinite(lc) | ~jnp.isfinite(la)

        i = rs.log_idx
        rec = dict(t=rs.t_env, avg100=avg100, mean_len=jnp.mean(eps.length.astype(jnp.float32)), critic_loss=lc,
                   actor_pol=lpol, entropy=ment, lam=lam_ent)
        logs = {k: rs.logs[k].at[i].set(jnp.where(do_log, rec[k].astype(rs.logs[k].dtype), rs.logs[k][i]))
                for k in rec}
        pv = probe_values(pc)
        logs["probe"] = rs.logs["probe"].at[i].set(jnp.where(do_log, pv, rs.logs["probe"][i]))
        new = RunState(pa=pa, pc=pc, pt=pt, oa=oa, oc=oc, key=rs.key, upd=rs.upd + 1, t_env=t_env,
                       next_target=next_target, next_log=next_log, log_idx=i + do_log.astype(jnp.int32), win=win,
                       win_n=win_n, bad=bad, logs=logs)
        return jax.tree.map(lambda n, o: jnp.where(active, n, o), new, rs)

    def update(rs: RunState) -> RunState:
        active = rs.t_env < T_max
        keys = jax.vmap(lambda k, u: jax.random.split(jax.random.fold_in(k, u), E))(rs.key, rs.upd)
        eps = rollout(rs.pa, keys, active)
        maxlen = jnp.max(eps.length)
        idx = jnp.sum(jnp.asarray(buckets) < maxlen)
        branches = [lambda r, e, L=L: jax.vmap(learn)(r, slice_episodes(e, L)) for L in buckets]
        return jax.lax.switch(idx, branches, rs, eps)

    update.learn = learn          # exposed for the official differential test (tests/test_official_losses.py)
    update.rollout = rollout
    return update


def _mode_k(method):
    from model import METHODS
    return METHODS[method]
