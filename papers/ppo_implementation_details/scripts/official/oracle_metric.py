"""ORACLE (authors'-code check) for the metric differential test: replay our recorded stream through the authors' env stack with
RecordEpisodeStatistics innermost (CleanRL ppo_continuous_action.py make_env order; post ppo.py/ppo_continuous_action.py
likewise) and collect the episodic return/length each env reports, per vector step.
* gymnasium 0.29.1 (.venv): SyncVectorEnv, infos["final_info"][i]["episode"] (CleanRL @fe8d8a0 reads it there).
* gym 0.21.0 (.venv_post): SyncVectorEnv, infos[i]["episode"] (the post's ppo.py @fbef824 reads it there, and logs only
  the FIRST such env per step: `break`, both the full and the first-only arrays are saved).
Usage: <venv python> -I scripts/official/oracle_metric.py <dir with ours_metric.npz>"""
import os
import sys

import numpy as np

try:
    import gymnasium as gym
    NEW = True
except ImportError:
    import gym
    NEW = False
DIR = sys.argv[1]
z = np.load(os.path.join(DIR, "ours_metric.npz"))
S = {k.split("/")[1]: z[k] for k in z.files if k.startswith("stream/")}
N, T, D = S["nxt"].shape


class ReplayEnv(gym.Env):
    def __init__(self, i):
        self.i, self.t, self.first = i, 0, True
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (D,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def reset(self, seed=None, options=None):
        o = S["init"][self.i].copy() if self.first else S["rst"][self.i, self.t - 1].copy()
        self.first = False
        return (o, {}) if NEW else o

    def step(self, action):
        t = self.t
        self.t += 1
        o, r = S["nxt"][self.i, t].copy(), float(S["rew"][self.i, t])
        te, tr = bool(S["term"][self.i, t]), bool(S["trunc"][self.i, t])
        return (o, r, te, tr, {}) if NEW else (o, r, te or tr, {})


def make(i):
    def thunk():
        env = gym.wrappers.RecordEpisodeStatistics(ReplayEnv(i))
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


envs = gym.vector.SyncVectorEnv([make(i) for i in range(N)])
envs.reset()
R, L, R1 = np.zeros((N, T)), np.zeros((N, T)), np.zeros((N, T))
for t in range(T):
    if NEW:
        _, _, _, _, infos = envs.step(np.zeros((N, 1), np.float32))
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                if info and "episode" in info:
                    R[i, t], L[i, t] = float(np.ravel(info["episode"]["r"])[0]), float(np.ravel(info["episode"]["l"])[0])
    else:
        _, _, _, infos = envs.step(np.zeros((N, 1), np.float32))
        first = True
        for i, item in enumerate(infos):
            if "episode" in item.keys():
                R[i, t], L[i, t] = item["episode"]["r"], item["episode"]["l"]
                if first:
                    R1[i, t], first = item["episode"]["r"], False
tag = "gymnasium029" if NEW else "gym021"
out = {f"{tag}/ep_ret": R, f"{tag}/ep_len": L}
if not NEW:
    out["gym021/ep_ret_first_only"] = R1
np.savez(os.path.join(DIR, f"oracle_metric_{tag}.npz"), **out)
print("[INFO] oracle wrote", f"oracle_metric_{tag}.npz", gym.__version__, "episodes", int((L > 0).sum()))
