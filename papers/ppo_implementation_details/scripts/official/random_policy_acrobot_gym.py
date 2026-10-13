"""DIVERGENT check (full-scale, Acrobot): uniform-random policy on gymnasium 0.29.1 Acrobot-v1 (float64, the CleanRL
oracle's env). Usage: .venv/bin/python -I scripts/official/random_policy_acrobot_gym.py <n_episodes> <out.json>"""
import json
import sys

import gymnasium as gym
import numpy as np

n, out = int(sys.argv[1]), sys.argv[2]
env = gym.make("Acrobot-v1")
rng = np.random.default_rng(2026)
rets, succ = [], []
for i in range(n):
    env.reset(seed=int(rng.integers(2**31)))
    ret, done = 0.0, False
    while True:
        _, r, term, trunc, _ = env.step(int(rng.integers(3)))
        ret += r
        if term or trunc:
            succ.append(float(term))
            break
    rets.append(ret)
rets, succ = np.array(rets), np.array(succ)
o = dict(env="gymnasium 0.29.1 Acrobot-v1 (float64)", n=n, success_rate=float(succ.mean()),
         success_se=float(np.sqrt(succ.mean() * (1 - succ.mean()) / n)), mean_return=float(rets.mean()),
         return_se=float(rets.std(ddof=1) / np.sqrt(n)))
print("[METRIC]", json.dumps(o))
json.dump(o, open(out, "w"), indent=1)
