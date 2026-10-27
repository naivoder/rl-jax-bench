"""Does the third-party JSRLAlgorithm(DQN) keep DQN's epsilon-greedy?  (Its predict() overrides DQN.predict and calls policy.predict directly.)
Run in the reference venv.  Sets exploration_rate = 1.0 (pure random under stock DQN) and counts how often the executed action differs from greedy."""
import json, sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
torch.set_num_threads(1)
import gymnasium as gym
from stable_baselines3 import DQN
from jsrl import get_jsrl_algorithm

env = gym.make("CartPole-v1")
guide = DQN("MlpPolicy", env, seed=3).policy
J = get_jsrl_algorithm(DQN)("MlpPolicy", env, seed=0, verbose=0, policy_kwargs=dict(guide_policy=guide, max_horizon=0, horizons=[0], strategy="curriculum"))
S = DQN("MlpPolicy", env, seed=0)
res = {}
for name, m in (("stock_DQN", S), ("jsrl_DQN", J)):
    m.exploration_rate = 1.0
    rng = np.random.default_rng(0)
    diff = 0; N = 400
    for _ in range(N):
        o = rng.normal(size=(1, 4)).astype(np.float32)
        a, _ = m.predict(o, deterministic=False)
        g, _ = (m.policy.predict(o, deterministic=True) if name == "stock_DQN" else m.policy.predict(o, np.zeros(1, np.int32) + 10**6, None, None, True))
        diff += int(a[0] != g[0])
        if name == "jsrl_DQN":
            m._timesteps[:] = 10**6          # keep the counter beyond the horizon: the explorer acts
    res[name] = dict(n=N, frac_action_differs_from_greedy_at_eps1=diff / N)
print(json.dumps(res)); json.dump(res, open(sys.argv[1], "w"))
