"""small-scale: obs-normalization / reward-scaling wrapper differential test (K6-K9), our side.

A recorded 1000-step stream for N=2 envs (obs_dim 3, random episode lengths, both terminations and truncations) is
replayed through our VecEnv in both modes. The oracle replays the same stream through baselines@ea25b9e VecNormalize
(baselines mode) and gymnasium 0.29.1 NormalizeObservation/NormalizeReward inside SyncVectorEnv, CleanRL's wrapper
order (gymnasium029 mode), and gym 0.21.0 wrappers as in the post's own ppo_continuous_action.py (gym021 mode,
oracle_wrappers_gym021.py under .venv_post): scripts/official/oracle_wrappers.py.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.train import CONFIG_DIR, load_config  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import data as dt  # noqa: E402

OUT = os.path.join(REPO, "runs", "small", "difftest")
N, T, D = 2, 1000, 3


def make_stream(seed=0):
    rng = np.random.default_rng(seed)
    nxt = rng.normal(2.0, 3.0, (N, T, D)).astype(np.float32)        # raw obs after step t (terminal obs if done)
    rst = rng.normal(-1.0, 0.5, (N, T, D)).astype(np.float32)       # reset obs used if step t ends the episode
    rew = np.round(rng.normal(0.5, 2.0, (N, T)), 3).astype(np.float32)
    term = np.zeros((N, T), bool)
    trunc = np.zeros((N, T), bool)
    for i in range(N):
        t = 0
        while t < T:
            L = int(rng.integers(5, 40))
            t += L
            if t - 1 < T:
                (term if rng.random() < 0.5 else trunc)[i, t - 1] = True
    init = rng.normal(0.0, 1.0, (N, D)).astype(np.float32)
    return dict(nxt=nxt, rst=rst, rew=rew, term=term, trunc=trunc, init=init)


def run(mode: str, s: dict):
    cfg = load_config(os.path.join(CONFIG_DIR, "pendulum.yaml"))
    cfg.update(num_envs=N, obs_norm=True, reward_scaling=True, reward_scaling_mode=mode)
    env = dt.VecEnv(cfg)
    S = {k: jnp.asarray(v) for k, v in s.items()}

    def reset_one(key):  # key carries the env index in its first word (replay only)
        i = key[0]
        return S["init"][i], (i, jnp.int32(0))

    def step_one(key, state, action):
        i, t = state
        done = jnp.logical_or(S["term"][i, t], S["trunc"][i, t])
        o_next = S["nxt"][i, t]
        return (jnp.where(done, S["rst"][i, t], o_next), (i, t + 1), S["rew"][i, t], S["term"][i, t],
                S["trunc"][i, t], o_next)

    env.reset_one, env.step_one = reset_one, step_one
    keys = jnp.stack([jnp.array([i, 0], jnp.uint32) for i in range(N)])
    obs0, es = jax.vmap(reset_one)(keys)
    vs, obs0 = env.wrap_reset(obs0, es)
    step = jax.jit(lambda vs: env.step(jnp.zeros(2, jnp.uint32), vs, jnp.zeros((N, 1))))
    obs, rews, fins = [np.asarray(obs0)], [], []
    for t in range(T):
        vs, so = step(vs)
        obs.append(np.asarray(so.obs)); rews.append(np.asarray(so.reward)); fins.append(np.asarray(so.final_obs))
    return np.stack(obs, 1), np.stack(rews, 1), np.stack(fins, 1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    s = make_stream()
    out = {f"stream/{k}": v for k, v in s.items()}
    for mode in ["baselines", "gymnasium029", "gym021"]:
        o, r, f = run(mode, s)
        out.update({f"{mode}/obs": o, f"{mode}/rew": r, f"{mode}/final": f})
    np.savez(os.path.join(OUT, "ours_wrappers.npz"), **out)
    print("[INFO] wrote ours_wrappers.npz", {k: v.shape for k, v in out.items()})
