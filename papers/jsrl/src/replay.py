# PROVENANCE: copied from our earlier DQfD reproduction (src/replay.py; our own prior code, not the authors'); JSRL changes are marked 'JSRL:'.
"""ONE prioritized replay buffer with a permanent demo region and an agent FIFO ring (paper §DQfD, Algorithm 1).

Single-run pure functions (vmapped over runs by the caller).  Layout of the flat arrays (length npad):
    [0, n_demo)                 demonstration transitions, never overwritten
    [n_demo, n_demo + cap)      agent FIFO ring
    [n_demo + cap, npad)        zero-priority padding so that npad = nb * bs
Storage is bit-packed (obs and next_obs are binary 10x10xC grids) so a 100k ring costs ~25 MB per run.

Sampling is EXACT proportional sampling, P(i) = pa_i / sum_j pa_j with pa = p^alpha, done in two levels (block
sums, then an in-block cumulative sum) so float32 cumulative error stays ~1e-6 relative.
n-step returns are computed at SAMPLE time from the stored 1-step rewards by scanning up to n successors; a new
agent transition becomes sampleable (priority activated) only once its n-step window is complete (n-1 steps
later).  Termination (game over) stops the window with no bootstrap; truncation (time cap)
stops the window and bootstraps from the stored next observation with gamma^m.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Layout(NamedTuple):
    n_demo: int
    cap: int
    bs: int
    nb: int
    npad: int
    obs_bytes: int
    n_step: int
    gamma: float
    alpha: float


def make_layout(n_demo: int, cap: int, obs_bytes: int, n_step: int = 10, gamma: float = 0.99,
                alpha: float = 0.4, bs: int = 256) -> Layout:
    n = n_demo + cap
    nb = -(-n // bs)
    return Layout(n_demo, cap, bs, nb, nb * bs, obs_bytes, n_step, gamma, alpha)


def init_buffer(L: Layout, demo: dict | None, hp: dict) -> dict:
    """Empty buffer; demo = dict(obs, nobs, act, rew, term, end) of length n_demo (rew already transformed).
    hp: demo_on (0/1: demo transitions are sampleable), eps_d, eps_a, rbs (0/1: pre-fill the agent ring with demos)."""
    z = lambda shape, dt: jnp.zeros(shape, dt)
    buf = dict(obs=z((L.npad, L.obs_bytes), jnp.uint8), nobs=z((L.npad, L.obs_bytes), jnp.uint8),
               act=z((L.npad,), jnp.int32), rew=z((L.npad,), jnp.float32), term=z((L.npad,), bool),
               end=z((L.npad,), bool), pa=z((L.npad,), jnp.float32), is_demo=z((L.npad,), bool))
    nd = L.n_demo
    p0 = 1.0 + hp["eps_d"]  # demo initial priority |delta|:=1, p = |delta| + eps_d
    if demo is not None and nd > 0:
        for k in ("obs", "nobs", "act", "rew", "term", "end"):
            buf[k] = buf[k].at[:nd].set(demo[k])
        buf["is_demo"] = buf["is_demo"].at[:nd].set(True)
        buf["pa"] = buf["pa"].at[:nd].set(hp["demo_on"] * p0 ** L.alpha)
    maxp = jnp.where(hp["demo_on"] > 0, p0, 1.0).astype(jnp.float32)
    ptr = jnp.zeros((), jnp.int32)
    nw = jnp.zeros((), jnp.int32)
    if demo is not None and nd > 0 and nd <= L.cap:
        r = hp["rbs"] > 0  # Replay Buffer Spiking: agent ring starts full of demo data (overwritable)
        for k in ("obs", "nobs", "act", "rew", "term", "end"):
            buf[k] = buf[k].at[nd:nd + nd].set(jnp.where(r, demo[k], buf[k][nd:nd + nd]))
        buf["pa"] = buf["pa"].at[nd:nd + nd].set(jnp.where(r, (1.0 + hp["eps_a"]) ** L.alpha, 0.0))
        ptr = jnp.where(r, nd % L.cap, 0).astype(jnp.int32)
        nw = jnp.where(r, nd, 0).astype(jnp.int32)
        maxp = jnp.where(r, 1.0 + hp["eps_a"], maxp).astype(jnp.float32)
    buf.update(ptr=ptr, nw=nw, maxp=maxp)
    return buf


def add_transition(L: Layout, buf: dict, obs_p, nobs_p, act, rew, term, end) -> dict:
    """Append one agent transition at the ring head; activate the transition whose n-step window just completed."""
    w = L.n_demo + buf["ptr"]
    b = dict(buf)
    b["obs"] = buf["obs"].at[w].set(obs_p)
    b["nobs"] = buf["nobs"].at[w].set(nobs_p)
    b["act"] = buf["act"].at[w].set(act)
    b["rew"] = buf["rew"].at[w].set(rew)
    b["term"] = buf["term"].at[w].set(term)
    b["end"] = buf["end"].at[w].set(end)
    b["is_demo"] = buf["is_demo"].at[w].set(False)
    pa = buf["pa"].at[w].set(0.0)  # pending until its n-step window is complete
    nw2 = buf["nw"] + 1
    w_act = L.n_demo + (buf["ptr"] - (L.n_step - 1)) % L.cap
    pa = pa.at[w_act].set(jnp.where(nw2 >= L.n_step, buf["maxp"] ** L.alpha, pa[w_act]))
    b["pa"] = pa
    b["ptr"] = (buf["ptr"] + 1) % L.cap
    b["nw"] = nw2
    return b


def sample(L: Layout, pa, key, batch: int):
    """Exact proportional sampling. Returns idx (B,), P (B,) = pa[idx]/sum(pa), total, valid (pa[idx] > 0)."""
    pa2 = pa.reshape(L.nb, L.bs)
    bsum = pa2.sum(axis=1)
    cb = jnp.cumsum(bsum)
    total = cb[-1]
    u = jax.random.uniform(key, (batch,)) * total
    blk = jnp.clip(jnp.sum(cb[None, :] <= u[:, None], axis=1), 0, L.nb - 1)
    ur = u - (cb[blk] - bsum[blk])
    cr = jnp.cumsum(pa2[blk], axis=1)
    inb = jnp.clip(jnp.sum(cr <= ur[:, None], axis=1), 0, L.bs - 1)
    idx = blk * L.bs + inb
    P = pa[idx] / jnp.maximum(total, 1e-30)
    return idx, P, total, pa[idx] > 0


def is_weights(P, n_active, beta, valid):
    """w_i = (N P(i))^-beta, normalised by the batch max; invalid samples get weight 0."""
    raw = jnp.where(valid, (jnp.maximum(n_active, 1.0) * jnp.maximum(P, 1e-30)) ** (-beta), 0.0)
    return raw / jnp.maximum(raw.max(), 1e-30)


def _window(L: Layout, idx):
    """(B, n) indices of the n-step window starting at idx: ring successor for agent entries, +1 for demo entries."""
    k = jnp.arange(L.n_step)[None, :]
    ring = L.n_demo + (idx[:, None] - L.n_demo + k) % L.cap
    lin = jnp.minimum(idx[:, None] + k, L.npad - 1)
    return jnp.where(idx[:, None] >= L.n_demo, ring, lin)


def gather_batch(L: Layout, buf: dict, idx):
    """Everything the loss needs for transitions idx (B,): 1-step and n-step quantities (one gather per array)."""
    g = L.gamma
    out = dict(obs=buf["obs"][idx], act=buf["act"][idx], is_demo=buf["is_demo"][idx],
               r1=buf["rew"][idx], term1=buf["term"][idx], nobs1=buf["nobs"][idx])
    W = _window(L, idx)                                   # (B, n)
    rk, tk, ek = buf["rew"][W], buf["term"][W], buf["end"][W]
    # alive_k: no episode end strictly before position k
    alive = jnp.concatenate([jnp.ones_like(ek[:, :1]), jnp.cumsum(ek[:, :-1].astype(jnp.int32), axis=1) == 0], axis=1)
    disc = g ** jnp.arange(L.n_step, dtype=jnp.float32)[None, :]
    G = jnp.sum(jnp.where(alive, disc * rk, 0.0), axis=1)
    m = alive.sum(axis=1).astype(jnp.int32)
    last = jnp.take_along_axis(W, (m - 1)[:, None], axis=1)[:, 0]
    term_hit = jnp.any(alive & tk, axis=1)
    out["Gn"] = G
    out["nobsn"] = buf["nobs"][last]
    out["disc_n"] = jnp.where(term_hit, 0.0, g ** m.astype(jnp.float32))
    out["m"] = m
    return out


def update_priorities(L: Layout, buf: dict, idx, td_abs, valid, hp: dict) -> dict:
    """p_i <- |delta_i| + eps_d (demo) / eps_a (agent);  pa = p^alpha;  running max priority tracked."""
    eps = jnp.where(buf["is_demo"][idx], hp["eps_d"], hp["eps_a"])
    p = td_abs + eps
    new = jnp.where(valid, p ** L.alpha, buf["pa"][idx])
    b = dict(buf)
    b["pa"] = buf["pa"].at[idx].set(new)
    b["maxp"] = jnp.maximum(buf["maxp"], jnp.max(jnp.where(valid, p, 0.0)))
    return b
