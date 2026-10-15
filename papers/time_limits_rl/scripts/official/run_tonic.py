"""Oracle harness: runs Tonic @0e20c89's OWN code on inputs from an .npz and writes its outputs.

Run with the isolated oracle venv:  .venv/bin/python scripts/official/run_tonic.py <in.npz> <out.npz>
Loads tonic/replays/utils.py, tonic/environments/wrappers.py and tonic/environments/distributed.py BY FILE PATH (the
package __init__ imports torch/TF, which this venv deliberately does not have). Cases:
  lambda_returns : Tonic replays.utils.lambda_returns on (values, next_values, rewards, resets, terminations)
  runner         : Tonic Sequential(max_episode_steps=T) over TimeFeature(ScriptedEnv, T) with a scripted action list
"""
import importlib.util
import os
import sys

import gym
import numpy as np

ROOT = os.path.join(os.environ.get("OFFICIAL_ROOT", "external/official"), "tonic", "tonic")


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


utils = load("replays/utils.py", "tonic_replays_utils")
wrappers = load("environments/wrappers.py", "tonic_env_wrappers")
distributed = load("environments/distributed.py", "tonic_env_distributed")


class ScriptedEnv(gym.Env):
    """obs = [steps since reset, sum of actions since reset]; reward = 1 + action; terminates when action == 1
    (environmental termination) -- identical to tests/test_oracle_tonic.py::scripted_env."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
    action_space = gym.spaces.Discrete(2)
    name = "scripted"

    def reset(self):
        self.k, self.s = 0, 0.0
        return np.array([self.k, self.s], np.float32)

    def step(self, a):
        self.k += 1
        self.s += float(a)
        return np.array([self.k, self.s], np.float32), 1.0 + float(a), bool(a == 1), {}


def main(inp, out):
    z = dict(np.load(inp))
    res = {}
    if "lr_values" in z:
        res["lambda_returns"] = utils.lambda_returns(z["lr_values"], z["lr_next_values"], z["lr_rewards"],
                                                     z["lr_resets"], z["lr_terminations"], float(z["lr_gamma"]),
                                                     float(z["lr_lam"]))
    if "rn_actions" in z:
        T = int(z["rn_T"])
        seq = distributed.Sequential(lambda: wrappers.TimeFeature(ScriptedEnv(), T), max_episode_steps=T, workers=1)
        obs0 = seq.start()
        obs, fin, rst, trm, rew = [], [], [], [], []
        for a in z["rn_actions"]:
            o, info = seq.step(np.array([a]))
            obs.append(o[0]); fin.append(info["observations"][0]); rst.append(info["resets"][0])
            trm.append(info["terminations"][0]); rew.append(info["rewards"][0])
        res.update(rn_obs0=obs0[0], rn_obs=np.array(obs), rn_final=np.array(fin), rn_resets=np.array(rst),
                   rn_terms=np.array(trm), rn_rewards=np.array(rew))
    res["meta_numpy"] = np.array(np.__version__)
    res["meta_gym"] = np.array(gym.__version__)
    np.savez(out, **res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
