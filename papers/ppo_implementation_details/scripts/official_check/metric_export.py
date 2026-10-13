"""authors'-code check metric differential test, our side: the recorded wrapper stream (scripts/small/wrapper_export.make_stream:
N=2 envs, 1000 steps, terminations AND truncations, random rewards) through our VecEnv in all three wrapper modes; we
save the per-step raw episodic return / length of every episode that ends (StepOut.ep_ret_done / ep_len_done, the
quantities train.py turns into the paper's metric). The oracle (scripts/official/oracle_metric.py, gymnasium 0.29.1 and gym
0.21.0) replays the same stream through RecordEpisodeStatistics innermost + the normalization wrappers outside it.
Run: uv run python -m scripts.official_check.metric_export
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.train import CONFIG_DIR, load_config  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.small.wrapper_export import N, T, make_stream  # noqa: E402
from src import data as dt  # noqa: E402

OUT = os.path.join(REPO, "runs", "small", "difftest")


def episodes(mode, s):
    cfg = load_config(os.path.join(CONFIG_DIR, "pendulum.yaml"))
    cfg.update(num_envs=N, obs_norm=True, reward_scaling=True, reward_scaling_mode=mode)
    env = dt.VecEnv(cfg)
    S = {k: jnp.asarray(v) for k, v in s.items()}

    def reset_one(key):
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
    vs, _ = env.wrap_reset(obs0, es)
    step = jax.jit(lambda vs: env.step(jnp.zeros(2, jnp.uint32), vs, jnp.zeros((N, 1))))
    r, ln = [], []
    for _ in range(T):
        vs, so = step(vs)
        r.append(np.asarray(so.ep_ret_done)); ln.append(np.asarray(so.ep_len_done))
    return np.stack(r, 1), np.stack(ln, 1)   # (N, T)


if __name__ == "__main__":
    s = make_stream()
    out = {f"stream/{k}": v for k, v in s.items()}
    for mode in ("baselines", "gymnasium029", "gym021"):
        r, ln = episodes(mode, s)
        out[f"{mode}/ep_ret"], out[f"{mode}/ep_len"] = r, ln
    np.savez(os.path.join(OUT, "ours_metric.npz"), **out)
    print("[INFO] wrote ours_metric.npz; episodes:", int((out["baselines/ep_len"] > 0).sum()))
