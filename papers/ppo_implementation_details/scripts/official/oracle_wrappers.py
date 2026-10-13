"""ORACLE for the wrapper differential test: replays our recorded stream through
(a) openai/baselines@ea25b9e VecNormalize (source exec'd; only its two relative/TF imports are bound by hand), and
(b) gymnasium 0.29.1 NormalizeObservation -> TransformObservation(clip) -> NormalizeReward -> TransformReward(clip)
    per env inside SyncVectorEnv (CleanRL ppo_continuous_action.py make_env order).
Usage: .venv/bin/python -I scripts/official/oracle_wrappers.py <dir with ours_wrappers.npz>
"""
import os
import sys
import types

import gymnasium as gym
import numpy as np

HERE = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
BASE = os.path.join(HERE, "baselines", "baselines", "common")
DIR = sys.argv[1]
z = np.load(os.path.join(DIR, "ours_wrappers.npz"))
S = {k.split("/")[1]: z[k] for k in z.files if k.startswith("stream/")}
N, T, D = S["nxt"].shape


# ---- (a) baselines VecNormalize
def load_baselines_vecnormalize():
    rms_src = open(os.path.join(BASE, "running_mean_std.py")).read()
    rms_src = rms_src.replace("import tensorflow as tf", "tf = None").replace(
        "from baselines.common.tf_util import get_session", "get_session = None")
    rms_ns = {}
    exec(compile(rms_src, "running_mean_std.py", "exec"), rms_ns)
    mod = types.ModuleType("baselines.common.running_mean_std")
    mod.RunningMeanStd = rms_ns["RunningMeanStd"]
    for name in ["baselines", "baselines.common"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["baselines.common.running_mean_std"] = mod

    class VecEnvWrapper:  # minimal stand-in for baselines.common.vec_env.VecEnvWrapper (attribute plumbing only)
        def __init__(self, venv):
            self.venv = venv
            self.num_envs = venv.num_envs
            self.observation_space = venv.observation_space

    src = open(os.path.join(BASE, "vec_env", "vec_normalize.py")).read().replace("from . import VecEnvWrapper", "")
    ns = {"VecEnvWrapper": VecEnvWrapper}
    exec(compile(src, "vec_normalize.py", "exec"), ns)
    return ns["VecNormalize"]


class ReplayVec:
    num_envs = N
    observation_space = types.SimpleNamespace(shape=(D,))

    def __init__(self):
        self.t = 0

    def reset(self):
        return S["init"].astype(np.float64)

    def step_wait(self):
        t = self.t
        news = S["term"][:, t] | S["trunc"][:, t]
        obs = np.where(news[:, None], S["rst"][:, t], S["nxt"][:, t]).astype(np.float64)
        self.t += 1
        return obs, S["rew"][:, t].astype(np.float64), news, [{} for _ in range(N)]


VN = load_baselines_vecnormalize()
venv = VN(ReplayVec(), ob=True, ret=True, clipob=10.0, cliprew=10.0, gamma=0.99, epsilon=1e-8)
obs_b, rew_b = [venv.reset()], []
for t in range(T):
    o, r, _, _ = venv.step_wait()
    obs_b.append(o); rew_b.append(r)


# ---- (b) gymnasium 0.29.1 per-env wrappers in SyncVectorEnv
class ReplayEnv(gym.Env):
    def __init__(self, i):
        self.i, self.t = i, 0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (D,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.first = True

    def reset(self, seed=None, options=None):
        if self.first:
            self.first = False
            return S["init"][self.i].copy(), {}
        return S["rst"][self.i, self.t - 1].copy(), {}   # the reset obs recorded for the step that just ended

    def step(self, action):
        t = self.t
        self.t += 1
        return S["nxt"][self.i, t].copy(), float(S["rew"][self.i, t]), bool(S["term"][self.i, t]), \
            bool(S["trunc"][self.i, t]), {}


def make(i):
    def thunk():
        env = ReplayEnv(i)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


envs = gym.vector.SyncVectorEnv([make(i) for i in range(N)])
o, _ = envs.reset()
obs_g, rew_g, fin_g = [o], [], []
for t in range(T):
    o, r, term, trunc, info = envs.step(np.zeros((N, 1), np.float32))
    obs_g.append(o); rew_g.append(r)
    f = o.copy()
    if "final_observation" in info:
        for i in range(N):
            if info["_final_observation"][i]:
                f[i] = info["final_observation"][i]
    fin_g.append(f)

np.savez(os.path.join(DIR, "oracle_wrappers.npz"),
         **{"baselines/obs": np.stack(obs_b, 1), "baselines/rew": np.stack(rew_b, 1),
            "gymnasium029/obs": np.stack(obs_g, 1), "gymnasium029/rew": np.stack(rew_g, 1),
            "gymnasium029/final": np.stack(fin_g, 1)})
print("[INFO] oracle wrote oracle_wrappers.npz", "gymnasium", gym.__version__)
