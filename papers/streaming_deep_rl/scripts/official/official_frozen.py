"""Frozen-policy check (oracle side): the AUTHORS' StreamAC frozen at init (no update ever called) in their full wrapper stack
(ClipAction, ScaleReward, NormalizeObservation, AddTimeInfo -- the normalizer keeps updating, as in a real run),
Hopper-v4 (gymnasium 0.29.1 + mujoco 2.3.7), their seeding (torch/np manual_seed, env.reset(seed=seed) once).
Per seed: mean return of episodes ending in (0, 25k]. Usage: official_frozen.py <steps> <seed> ..."""
import os
import json, sys
import numpy as np, torch, gymnasium as gym
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' clone + venv (scripts/official/setup.sh)
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, "repo"))
from stream_ac_continuous import StreamAC
from normalization_wrappers import NormalizeObservation, ScaleReward
from time_wrapper import AddTimeInfo
T = int(sys.argv[1]); out = {}
for seed in map(int, sys.argv[2:]):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make('Hopper-v4'); env = gym.wrappers.FlattenObservation(env); env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env); env = ScaleReward(env, gamma=0.99); env = NormalizeObservation(env); env = AddTimeInfo(env)
    ag = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
    s, _ = env.reset(seed=seed); ep = []
    with torch.no_grad():
        for t in range(1, T + 1):
            a = ag.sample_action(s); s, r, te, tr, info = env.step(a)
            if te or tr:
                ep.append((t, float(info['episode']['r'][0]), int(info['episode']['l'][0]))); s, _ = env.reset()
    R = np.array([e[1] for e in ep]); out[seed] = dict(mean=float(R.mean()), n=len(R), first30=float(R[:30].mean()))
    print(seed, out[seed], flush=True)
json.dump(out, open('official_frozen.json', 'w'), indent=1)
m = [v['mean'] for v in out.values()]
print('authors frozen-init (0,%d] mean over seeds %.2f +- %.2f' % (T, np.mean(m), np.std(m, ddof=1) / np.sqrt(len(m))))
