"""Demonstrator training: a compact PPO (purejaxrl-style, one jitted process, vmapped envs) on MinAtar.

The expert is a DIFFERENT learner family from the Q-learner (like the paper's human).  PPO is
NOT the object of study, so truncation is treated as done (standard purejaxrl) and no claim is made about it.
All PPO constants are in src/configs/default.yaml (section ppo, tagged in sources.yaml).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .envs import MinAtar
from .model import ActorCritic


def train_ppo(env: MinAtar, key, cfg: dict, total_steps: int, log=print, chunk_updates: int = 20, snap_fracs=()):
    c = cfg["ppo"]
    n_envs, n_steps = c["n_envs"], c["n_steps"]
    n_upd = max(1, total_steps // (n_envs * n_steps))
    mb_size = n_envs * n_steps // c["n_minibatch"]
    net = ActorCritic(env.n_actions, channels=cfg["network"]["conv_channels"], hidden=c["hidden"])
    k_init, k_env, k_run = jax.random.split(key, 3)
    params = net.init(k_init, jnp.zeros((1,) + env.obs_shape, jnp.float32))
    sched = optax.linear_schedule(c["lr"], 0.0, n_upd * c["epochs"] * c["n_minibatch"]) if c["anneal_lr"] else c["lr"]
    tx = optax.chain(optax.clip_by_global_norm(c["max_grad_norm"]), optax.adam(sched, eps=1e-5))
    opt = tx.init(params)
    obs, est = jax.vmap(env.reset)(jax.random.split(k_env, n_envs))
    gamma, lam = cfg["env"]["gamma"], c["gae_lambda"]

    def rollout_step(carry, k):
        params, obs, est, ep_ret, fin_sum, fin_n = carry
        ka, ks = jax.random.split(k)
        logits, v = net.apply(params, obs.astype(jnp.float32))
        a = jax.random.categorical(ka, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(n_envs), a]
        obs2, est2, r, term, trunc = jax.vmap(env.step)(jax.random.split(ks, n_envs), est, a.astype(jnp.int32))
        done = term | trunc
        obs_r, est_r = jax.vmap(env.reset)(jax.random.split(jax.random.fold_in(ks, 1), n_envs))
        obs_n = jnp.where(done[:, None, None, None], obs_r, obs2)
        est_n = jax.tree_util.tree_map(lambda x, y: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), est_r, est2)
        ret = ep_ret + r
        fin_sum = fin_sum + jnp.sum(jnp.where(done, ret, 0.0))
        fin_n = fin_n + jnp.sum(done)
        tr = (obs, a, logp, v, r, done)
        return (params, obs_n, est_n, jnp.where(done, 0.0, ret), fin_sum, fin_n), tr

    def update(carry, k):
        params, opt, obs, est, ep_ret = carry
        kr, ku = jax.random.split(k)
        (_, obs, est, ep_ret, fs, fn), tr = jax.lax.scan(
            rollout_step, (params, obs, est, ep_ret, jnp.zeros(()), jnp.zeros(())), jax.random.split(kr, n_steps))
        o, a, lp, v, r, d = tr
        _, last_v = net.apply(params, obs.astype(jnp.float32))

        def gae(c2, x):
            adv_next, v_next = c2
            r_t, d_t, v_t = x
            delta = r_t + gamma * v_next * (1 - d_t) - v_t
            adv = delta + gamma * lam * (1 - d_t) * adv_next
            return (adv, v_t), adv

        _, adv = jax.lax.scan(gae, (jnp.zeros_like(last_v), last_v), (r, d.astype(jnp.float32), v), reverse=True)
        tgt = adv + v
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        data = tuple(map(flat, (o, a, lp, adv, tgt)))

        def epoch(c3, k3):
            params, opt = c3
            perm = jax.random.permutation(k3, n_envs * n_steps).reshape(c["n_minibatch"], mb_size)

            def mb(c4, ix):
                params, opt = c4
                bo, ba, blp, badv, btg = (x[ix] for x in data)

                def lf(p):
                    logits, val = net.apply(p, bo.astype(jnp.float32))
                    lpn = jax.nn.log_softmax(logits)
                    lp_a = jnp.take_along_axis(lpn, ba[:, None], 1)[:, 0]
                    ratio = jnp.exp(lp_a - blp)
                    nadv = (badv - badv.mean()) / (badv.std() + 1e-8)
                    pg = -jnp.minimum(ratio * nadv, jnp.clip(ratio, 1 - c["clip_eps"], 1 + c["clip_eps"]) * nadv).mean()
                    vl = 0.5 * jnp.square(val - btg).mean()
                    ent = -(jnp.exp(lpn) * lpn).sum(-1).mean()
                    return pg + c["vf_coef"] * vl - c["ent_coef"] * ent

                g = jax.grad(lf)(params)
                upd, opt = tx.update(g, opt, params)
                return (optax.apply_updates(params, upd), opt), None

            return jax.lax.scan(mb, (params, opt), perm)[0], None

        (params, opt), _ = jax.lax.scan(epoch, (params, opt), jax.random.split(ku, c["epochs"]))
        return (params, opt, obs, est, ep_ret), (fs, fn)

    @jax.jit
    def run_chunk(carry, key):
        carry, (fs, fn) = jax.lax.scan(update, carry, jax.random.split(key, chunk_updates))
        return carry, fs.sum() / jnp.maximum(fn.sum(), 1.0)

    carry = (params, opt, obs, est, jnp.zeros(n_envs))
    curve, snaps, t0 = [], {}, time.time()
    for i in range(-(-n_upd // chunk_updates)):
        carry, mret = run_chunk(carry, jax.random.fold_in(k_run, i))
        steps = (i + 1) * chunk_updates * n_envs * n_steps
        curve.append((steps, float(mret)))
        for fr in snap_fracs:
            if fr not in snaps and steps >= fr * total_steps:
                snaps[fr] = (steps, jax.tree_util.tree_map(jnp.copy, carry[0]))
        if i % 5 == 0 or i == -(-n_upd // chunk_updates) - 1:
            log(f"[METRIC] ppo steps={steps} train_return={float(mret):.3f} elapsed_s={time.time() - t0:.0f}")
    return carry[0], curve, net, snaps


def make_act_fn(net, params, eps: float, env: MinAtar):
    """Demonstrator policy: argmax of the PPO logits, replaced by a uniform random action with probability eps."""
    def act(obs, key):  # obs: (N, H, W, C) bool
        logits, _ = net.apply(params, obs.astype(jnp.float32))
        k1, k2 = jax.random.split(key)
        n = obs.shape[0]
        rand = jax.random.randint(k1, (n,), 0, env.n_actions)
        return jnp.where(jax.random.uniform(k2, (n,)) < eps, rand, jnp.argmax(logits, -1)).astype(jnp.int32)
    return act
