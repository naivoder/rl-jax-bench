"""MinAtar 1.0.15 reference rollouts (official venv): uniform-random and always-action-0 policies, N episodes per game,
sticky p = 0.1 and minimal action sets as registered (MinAtar/<Game>-v1). Writes an npz of returns and lengths.
Episodes are capped at CAP steps (Seaquest under always-no-op never terminates: the sub idles at the surface).
Usage: <official venv python> minatar_rollouts.py <out.npz> <n_episodes> <cap> <Game,Game,...>"""
import sys

import gymnasium as gym
import numpy as np

out, N, CAP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
GAMES = ["Breakout", "SpaceInvaders", "Asterix", "Seaquest"]
res = {}
for game in sys.argv[4].split(","):
    gi = GAMES.index(game)
    env = gym.make(f"MinAtar/{game}-v1")
    rng = np.random.default_rng(100 + gi)
    for pol in ("random", "fixed0"):
        env.reset(seed=1000 + gi)
        R, L = [], []
        for ep in range(N):
            env.reset()
            ret, n, done = 0.0, 0, False
            while not done and n < CAP:
                a = int(rng.integers(env.action_space.n)) if pol == "random" else 0
                _, r, done, _, _ = env.step(a)
                ret += r
                n += 1
            R.append(ret)
            L.append(n)
        res[f"{game}/{pol}/ret"] = np.array(R)
        res[f"{game}/{pol}/len"] = np.array(L)
        print(game, pol, np.mean(R), np.mean(L), flush=True)
np.savez(out, **res)
