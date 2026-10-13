"""ORACLE (authors'-code check) for the wrapper differential test against the POST'S OWN continuous-action code: the make_env of
ppo-implementation-details@fbef824 ppo_continuous_action.py (ClipAction -> NormalizeObservation ->
TransformObservation(clip 10) -> NormalizeReward -> TransformReward(clip 10)) with gym 0.21.0's own wrapper classes,
per env inside gym 0.21.0 SyncVectorEnv (auto-reset inside the vector env). ClipAction is irrelevant for a replay.
Usage: .venv_post/bin/python -I scripts/official/oracle_wrappers_gym021.py <dir with ours_wrappers.npz>"""
import os
import sys

import gym
import numpy as np

assert gym.__version__ == "0.21.0", gym.__version__
DIR = sys.argv[1]
z = np.load(os.path.join(DIR, "ours_wrappers.npz"))
S = {k.split("/")[1]: z[k] for k in z.files if k.startswith("stream/")}
N, T, D = S["nxt"].shape


class ReplayEnv(gym.Env):
    def __init__(self, i):
        self.i, self.t, self.first = i, 0, True
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (D,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def reset(self):
        if self.first:
            self.first = False
            return S["init"][self.i].copy()
        return S["rst"][self.i, self.t - 1].copy()   # the reset obs recorded for the step that just ended

    def step(self, action):
        t = self.t
        self.t += 1
        done = bool(S["term"][self.i, t] or S["trunc"][self.i, t])   # gym 0.21: TimeLimit sets done=True too
        return S["nxt"][self.i, t].copy(), float(S["rew"][self.i, t]), done, {}


def make(i):
    def thunk():
        env = ReplayEnv(i)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)                        # gamma=0.99 default, as in the post
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


envs = gym.vector.SyncVectorEnv([make(i) for i in range(N)])
obs, rew = [envs.reset()], []
for t in range(T):
    o, r, d, info = envs.step(np.zeros((N, 1), np.float32))
    obs.append(o); rew.append(r)
np.savez(os.path.join(DIR, "oracle_wrappers_gym021.npz"),
         **{"gym021/obs": np.stack(obs, 1), "gym021/rew": np.stack(rew, 1)})
print("[INFO] oracle wrote oracle_wrappers_gym021.npz", "gym", gym.__version__)
