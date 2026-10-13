"""ORACLE for the gymnax-vs-gymnasium dynamics test: gym 0.21.0 classic-control envs (the version the post's ppo.py ran on; authors'-code check), state injected through
env.unwrapped.state. Usage: .venv/bin/python -I scripts/official/oracle_envs_gym021.py (.venv_post) <dir with ours_envs.npz>"""
import os
import sys
import warnings

import gym
import numpy as np

warnings.filterwarnings("ignore")
DIR = sys.argv[1]
z = np.load(os.path.join(DIR, "ours_envs.npz"))
out = {}


def act(env_id, a):
    return np.asarray(a, np.float32).reshape(1) if env_id == "Pendulum-v1" else int(a)


for env_id in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]:
    env = gym.make(env_id)
    env.seed(0); env.reset()
    u = env.unwrapped
    S, A = z[f"{env_id}/tf/state"], z[f"{env_id}/tf/action"]
    obs_l, rew_l, term_l = [], [], []
    for s, a in zip(S, A):
        u.state = np.array(s, dtype=np.float64)
        if hasattr(u, "steps_beyond_done"):          # gym 0.21's name for the post-termination counter
            u.steps_beyond_done = None
        o, r, term, _ = u.step(act(env_id, a))   # unwrapped: no TimeLimit; gym 0.21 done == terminated here
        obs_l.append(o); rew_l.append(r); term_l.append(term)
    out[f"{env_id}/tf/obs"] = np.stack(obs_l)
    out[f"{env_id}/tf/reward"] = np.array(rew_l)
    out[f"{env_id}/tf/terminated"] = np.array(term_l)
    # open loop
    trajs = []
    for init in z[f"{env_id}/ol/init"]:
        u.state = np.array(init, dtype=np.float64)
        if hasattr(u, "steps_beyond_done"):          # gym 0.21's name for the post-termination counter
            u.steps_beyond_done = None
        ol = []
        for a in z[f"{env_id}/ol/actions"]:
            o, *_ = u.step(act(env_id, a))
            ol.append(o)
        trajs.append(np.stack(ol))
    out[f"{env_id}/ol/obs"] = np.stack(trajs)
np.savez(os.path.join(DIR, "oracle_envs_gym021.npz"), **out)
print("[INFO] oracle wrote oracle_envs_gym021.npz", gym.__version__)
