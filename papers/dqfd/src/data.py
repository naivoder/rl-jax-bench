"""Demonstration data: roll out a demonstrator for WHOLE episodes, store bit-packed transitions, hash the file.

One demo set per (game, demonstrator) shared by all learner seeds.  Labels are the executed actions
(including the demonstrator's epsilon-noise actions).  Truncation (time cap) is stored as `end & ~term`.
"""
from __future__ import annotations

import hashlib
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from .envs import MinAtar, pack_obs, reward_transform


def rollout_episodes(env: MinAtar, act_fn, key, n_envs: int, keep_transitions: bool = True):
    """Run n_envs independent episodes to their end (cap = env.max_steps). Returns per-env (T, ...) arrays + masks."""
    k0, k1 = jax.random.split(key)
    obs, st = jax.vmap(env.reset)(jax.random.split(k0, n_envs))

    def body(c, k):
        obs, st, alive = c
        ka, ks = jax.random.split(k)
        a = act_fn(obs, ka)
        obs2, st2, r, term, trunc = jax.vmap(env.step)(jax.random.split(ks, n_envs), st, a)
        out = dict(valid=alive, r=jnp.where(alive, r, 0.0), term=term & alive, end=(term | trunc) & alive)
        if keep_transitions:
            out.update(obs=pack_obs(obs), nobs=pack_obs(obs2), act=a)
        return (obs2, st2, alive & ~(term | trunc)), out

    _, tr = jax.lax.scan(body, (obs, st, jnp.ones(n_envs, bool)), jax.random.split(k1, env.max_steps))
    return tr  # each (T, n_envs, ...)


def episode_returns(tr):
    return np.asarray(tr["r"].sum(0))


def collect_demos(env: MinAtar, act_fn, key, target: int, n_envs: int = 64, max_rounds: int = 20):
    """Whole episodes (in env-index order within a round, never selected by return) until >= target transitions."""
    obs, nobs, act, rew, term, end, ep_ret, ep_len = [], [], [], [], [], [], [], []
    total, rnd = 0, 0
    run = jax.jit(lambda k: rollout_episodes(env, act_fn, k, n_envs))
    while total < target and rnd < max_rounds:
        tr = jax.tree_util.tree_map(np.asarray, run(jax.random.fold_in(key, rnd)))
        rnd += 1
        for e in range(n_envs):
            v = tr["valid"][:, e]
            n = int(v.sum())
            if n == 0:
                continue
            sl = slice(0, n)
            obs.append(tr["obs"][sl, e]); nobs.append(tr["nobs"][sl, e]); act.append(tr["act"][sl, e])
            rew.append(tr["r"][sl, e]); term.append(tr["term"][sl, e]); end.append(tr["end"][sl, e])
            ep_ret.append(float(tr["r"][sl, e].sum())); ep_len.append(n)
            total += n
            if total >= target:
                break
    cap_frac = float(np.mean([l >= env.max_steps for l in ep_len]))
    d = dict(obs=np.concatenate(obs), nobs=np.concatenate(nobs), act=np.concatenate(act).astype(np.int32),
             rew_raw=np.concatenate(rew).astype(np.float32), term=np.concatenate(term), end=np.concatenate(end),
             ep_ret=np.asarray(ep_ret, np.float32), ep_len=np.asarray(ep_len, np.int32))
    d["rew"] = np.asarray(reward_transform(jnp.asarray(d["rew_raw"])))
    d["meta"] = dict(n_transitions=int(total), n_episodes=len(ep_ret), cap_fraction=cap_frac,
                     best=float(np.max(ep_ret)), worst=float(np.min(ep_ret)), mean=float(np.mean(ep_ret)))
    return d


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_demos(path: str, d: dict) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrs = {k: v for k, v in d.items() if k != "meta"}
    np.savez_compressed(path, **arrs, meta=json.dumps(d["meta"]))
    h = sha256_file(path)
    with open(path + ".sha256", "w") as f:
        f.write(h + "\n")
    return h


def load_demos(path: str, expect_sha: str | None = None) -> dict:
    h = sha256_file(path)
    if expect_sha is not None and h != expect_sha:
        raise RuntimeError(f"demo file hash mismatch for {path}: {h} != {expect_sha} (refusing to run)")
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    d["sha256"] = h
    return d


def demo_to_buffer_arrays(d: dict) -> dict:
    """Arrays in the layout expected by replay.init_buffer."""
    return dict(obs=jnp.asarray(d["obs"]), nobs=jnp.asarray(d["nobs"]), act=jnp.asarray(d["act"], jnp.int32),
                rew=jnp.asarray(d["rew"], jnp.float32), term=jnp.asarray(d["term"]), end=jnp.asarray(d["end"]))


def policy_returns(env: MinAtar, act_fn, key, n_eps: int, chunk: int = 250):
    """Raw-return of n_eps fresh rollouts (chunked to bound memory)."""
    run = jax.jit(lambda k: rollout_episodes(env, act_fn, k, chunk, keep_transitions=False)["r"].sum(0))
    out = [np.asarray(run(jax.random.fold_in(key, i))) for i in range(-(-n_eps // chunk))]
    return np.concatenate(out)[:n_eps]
