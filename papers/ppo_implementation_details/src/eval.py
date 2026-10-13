"""Evaluation: the paper's metric (raw undiscounted episodic return of TRAINING episodes vs global step) summarized
per seed, a greedy diagnostic eval, and the seed-level aggregation helpers.

Per seed, from the episode records [global_step, return_sum, count, length_sum] (one row per vector step with >= 1
finished episode; sums over the N envs):
* final_window  = sum(returns) / count over episodes finishing in the last 10 % of training;
* auc           = mean over 50 equal step bins of the bin-mean return (empty bins carry the previous value;
                  leading empty bins are ignored);
* steps_to_threshold = first bin end where the 10-bin running mean of the 50-bin series reaches the threshold
                  (CartPole-v1 475, Acrobot-v1 -100; gym reward_threshold), else None.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

THRESHOLDS = {"CartPole-v1": 475.0, "Acrobot-v1": -100.0}  # sources.yaml eval.thresholds (ASSUMED)
FINAL_FRAC = 0.10  # sources.yaml eval.final_window_frac (ASSUMED)
N_BINS = 50        # sources.yaml eval.auc_bins (ASSUMED)


def binned(episodes: np.ndarray, total: int, n_bins: int = N_BINS) -> np.ndarray:
    e = np.asarray(episodes, dtype=float).reshape(-1, 4)
    edges = np.linspace(0, total, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, e[:, 0], side="left") - 1, 0, n_bins - 1)
    s = np.bincount(idx, weights=e[:, 1], minlength=n_bins)
    c = np.bincount(idx, weights=e[:, 2], minlength=n_bins)
    out = np.full(n_bins, np.nan)
    last = np.nan
    for i in range(n_bins):
        if c[i] > 0:
            last = s[i] / c[i]
        out[i] = last
    return out


def summarize(episodes: np.ndarray, total: int, env_id: str) -> dict[str, Any]:
    e = np.asarray(episodes, dtype=float).reshape(-1, 4)
    w = e[:, 0] > (1.0 - FINAL_FRAC) * total
    fw = float(e[w, 1].sum() / e[w, 2].sum()) if e[w, 2].sum() > 0 else float("nan")
    b = binned(e, total)
    auc = float(np.nanmean(b)) if np.isfinite(b).any() else float("nan")
    thr = THRESHOLDS.get(env_id)
    sts = None
    if thr is not None:
        rm = np.array([np.nanmean(b[max(0, i - 9): i + 1]) if np.isfinite(b[max(0, i - 9): i + 1]).any() else np.nan
                       for i in range(len(b))])
        hit = np.where(rm >= thr)[0]
        sts = int((hit[0] + 1) * total / len(b)) if len(hit) else None
    return dict(final_window=fw, final_window_episodes=int(e[w, 2].sum()), auc=auc, steps_to_threshold=sts,
                n_episodes=int(e[:, 2].sum()), curve_50bins=b.tolist())


# ------------------------------------------------------------------ greedy diagnostic (not a headline)
def greedy_eval(ppo, runner, n_episodes: int = 10, master_key: int = 12345):
    """Deterministic-action episodes with FROZEN normalization (train-only stats); returns (S, n_episodes)."""
    import jax
    import jax.numpy as jnp
    from . import model as mdl

    env, spec, net = ppo.env, ppo.spec, ppo.net
    T = spec.max_steps

    def one_episode(params, vs, key):
        o, s = env.reset_one(key)

        def body(c, t):
            s, o, ret, alive, carry = c
            on = env.normalize_frozen(vs, o[None])[0]
            if ppo.recurrent:
                logits, _, carry = net.apply(params, on[None, None], jnp.zeros((1, 1)), carry)
                a = mdl.mode(logits[0, 0], False)
            else:
                dist, _ = net.apply(params, on)
                a = mdl.mode(dist, ppo.cont)
            a_env = env._prep_action(a)
            o2, s2, r, term, trunc, _ = env.step_one(jax.random.fold_in(key, t), s, a_env)
            ret = ret + alive * r
            alive = alive * (1.0 - jnp.logical_or(term, trunc).astype(jnp.float32))
            return (s2, o2, ret, alive, carry), None

        carry0 = net.initial_carry(1) if ppo.recurrent else (jnp.zeros((1, 1)), jnp.zeros((1, 1)))
        (_, _, ret, _, _), _ = jax.lax.scan(body, (s, o, jnp.float32(0.0), jnp.float32(1.0), carry0), jnp.arange(T))
        return ret

    def per_seed(params, vs):
        keys = jax.random.split(jax.random.PRNGKey(master_key), n_episodes)
        return jax.vmap(lambda k: one_episode(params, vs, k))(keys)

    out = jax.jit(jax.vmap(per_seed))(runner.params, runner.vs)
    return np.asarray(out)


# ------------------------------------------------------------------ aggregation
def load_runs(root: str, env_id: str, arm: str) -> dict[int, dict]:
    d = os.path.join(root, env_id, arm)
    out = {}
    if not os.path.isdir(d):
        return out
    for s in sorted(os.listdir(d)):
        p = os.path.join(d, s, "metrics.json")
        if s.startswith("seed_") and os.path.exists(p):
            with open(p) as f:
                out[int(s[5:])] = json.load(f)
    return out


def seed_table(root: str, env_id: str, arm: str, key: str = "final_window") -> dict[int, float]:
    return {s: r["summary"][key] for s, r in load_runs(root, env_id, arm).items()}
