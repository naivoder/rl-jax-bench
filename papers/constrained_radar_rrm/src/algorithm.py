"""Learners: CDRL-DDPG (Alg. 2) and CDRL-DQL (Alg. 1). The dual update (eq. 33) lives in env.dual_update and runs
inside every env step for every policy (it is part of the environment the learner sees).

DDPG update, generic over apply functions so the CleanRL oracle networks can be plugged in:
  y = r + gamma * disc * Q'(s', mu'(s'));  critic: mean (Q(s,a) - y)^2;  actor: -mean Q_new(s, mu(s))
  (the actor step uses the JUST-UPDATED critic, as CleanRL ddpg_continuous_action_jax), then soft targets
  theta' <- tau theta + (1 - tau) theta'  (Table V tau = 0.01). One critic + one actor step per call.
DQL: rl_jax_bench.drqn QConfig(lstm=0, trace_len=1, double=False, hidden=(64, 64)) -- imported, not copied.
"""
from __future__ import annotations

import sys
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from rl_jax_bench import drqn  # noqa: E402

from .model import Actor, Critic  # noqa: E402


class DDPGState(NamedTuple):
    actor: Any
    critic: Any
    actor_t: Any
    critic_t: Any
    opt_a: Any
    opt_c: Any
    n: jax.Array


def ddpg_init(key, obs_dim, act_dim, actor_net, critic_net, opt_a, opt_c) -> DDPGState:
    ka, kc = jax.random.split(key)
    s0 = jnp.zeros((1, obs_dim), jnp.float32)
    a0 = jnp.zeros((1, act_dim), jnp.float32)
    pa = actor_net.init(ka, s0)
    pc = critic_net.init(kc, s0, a0)
    return DDPGState(pa, pc, jax.tree.map(jnp.copy, pa), jax.tree.map(jnp.copy, pc), opt_a.init(pa), opt_c.init(pc),
                     jnp.zeros((), jnp.int32))


def critic_loss(pc, st: DDPGState, batch, actor_apply, critic_apply, gamma, a_clip=None):
    a_next = actor_apply(st.actor_t, batch["s2"])
    if a_clip is not None:
        a_next = jnp.clip(a_next, *a_clip)
    q_next = critic_apply(st.critic_t, batch["s2"], a_next).reshape(-1)
    y = jax.lax.stop_gradient(batch["r"] + gamma * batch["disc"] * q_next)
    q = critic_apply(pc, batch["s"], batch["a"]).reshape(-1)
    return ((q - y) ** 2).mean(), q.mean()


def actor_loss(pa, pc, batch, actor_apply, critic_apply):
    return -critic_apply(pc, batch["s"], actor_apply(pa, batch["s"])).reshape(-1).mean()


def ddpg_update(st: DDPGState, batch, actor_apply: Callable, critic_apply: Callable, opt_a, opt_c, gamma: float,
                tau_soft: float, a_clip=None):
    (lc, qm), gc = jax.value_and_grad(critic_loss, has_aux=True)(st.critic, st, batch, actor_apply, critic_apply,
                                                                   gamma, a_clip)
    uc, oc = opt_c.update(gc, st.opt_c, st.critic)
    pc = optax.apply_updates(st.critic, uc)
    la, ga = jax.value_and_grad(actor_loss)(st.actor, pc, batch, actor_apply, critic_apply)
    ua, oa = opt_a.update(ga, st.opt_a, st.actor)
    pa = optax.apply_updates(st.actor, ua)
    st2 = DDPGState(pa, pc, optax.incremental_update(pa, st.actor_t, tau_soft),
                    optax.incremental_update(pc, st.critic_t, tau_soft), oa, oc, st.n + 1)
    return st2, dict(critic_loss=lc, actor_loss=la, q_mean=qm, grad_c=optax.tree.norm(gc), grad_a=optax.tree.norm(ga))


def make_ddpg(y: dict, K: int, T0: float):
    d = y["ddpg"]
    actor = Actor(act_dim=K, hidden=tuple(d["actor_hidden"]), T0=T0, out_bias=float(d.get("actor_out_bias", 0.0)))
    critic = Critic(hidden=tuple(d["critic_hidden"]), T0=T0)
    opt_a = optax.adam(d["actor_lr"])        # Adam default eps 1e-8 (A-25)
    opt_c = optax.adam(d["critic_lr"])
    return actor, critic, opt_a, opt_c


# ----------------------------------------------------------------------------- replay for DDPG (flat transitions)
class Replay(NamedTuple):
    s: jax.Array
    a: jax.Array
    r: jax.Array
    s2: jax.Array
    ptr: jax.Array
    count: jax.Array


def replay_init(n, obs_dim, act_dim) -> Replay:
    f = jnp.float32
    return Replay(jnp.zeros((n, obs_dim), f), jnp.zeros((n, act_dim), f), jnp.zeros(n, f), jnp.zeros((n, obs_dim), f),
                  jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32))


def replay_add(b: Replay, s, a, r, s2) -> Replay:
    i = b.ptr
    n = b.s.shape[0]
    return Replay(b.s.at[i].set(s), b.a.at[i].set(a), b.r.at[i].set(r), b.s2.at[i].set(s2), (i + 1) % n,
                  jnp.minimum(b.count + 1, n))


def replay_sample(b: Replay, key, batch: int, gamma_disc: float = 1.0) -> dict:
    idx = jax.random.randint(key, (batch,), 0, jnp.maximum(b.count, 1))
    return dict(s=b.s[idx], a=b.a[idx], r=b.r[idx], s2=b.s2[idx], disc=jnp.full((batch,), gamma_disc, jnp.float32))


# ----------------------------------------------------------------------------- DQL via drqn.py
def make_dql_cfg(y: dict, K: int, buffer_traces: int) -> drqn.QConfig:
    q = y["dql"]
    return drqn.QConfig(n_actions=q["n_levels"], obs_dim=2 * K + 2, hidden=tuple(q["hidden"]), lstm=0, double=False,
                        gamma=q["gamma"], lr=q["lr"], trace_len=1, burn_in=0, batch_size=q["batch"],
                        buffer_traces=buffer_traces, target_period=q["target_period"])


def dql_q(cfg: drqn.QConfig, params, obs):
    """Q values [..., A] for per-agent observations [..., 2K+2] (feed-forward: empty carry)."""
    q, _ = drqn.act_q(drqn.make_network(cfg), cfg, params, obs.astype(jnp.float32), ())
    return q
