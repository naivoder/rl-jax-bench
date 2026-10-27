"""Differential test 2, SB3 side (stock SB3 2.7.0 NStepReplayBuffer, the n-step learner used in the reference run): same stream,
compare per transition the n-step return, the effective bootstrap coefficient (discount x (1 - final_done)) and the bootstrap obs."""
import sys, json, os
sys.path.insert(0, os.path.join(os.environ.get('OFFICIAL_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'external', 'official')), 'sb3_nstep'))   # stable-baselines3 2.7.0 (setup.sh)
import numpy as np, gymnasium as gym
from stable_baselines3.common.buffers import NStepReplayBuffer
import stable_baselines3; assert stable_baselines3.__version__ == '2.7.0'
d = np.load('nstep_stream.npz'); T = len(d['rew']); N, G = 10, 0.99
b = NStepReplayBuffer(T + 10, gym.spaces.Box(-1e9, 1e9, (1,), np.float32), gym.spaces.Discrete(3), device='cpu', n_envs=1,
                      handle_timeout_termination=True, n_steps=N, gamma=G)
for i in range(T):
    done = bool(d['term'][i] or d['trunc'][i])
    b.add(np.array([[i]], np.float32), np.array([[100000 + i]], np.float32), np.array([[0]]), np.array([d['rew'][i]], np.float32),
          np.array([done]), [{'TimeLimit.truncated': bool(d['trunc'][i] and not d['term'][i])}])
s = b._get_samples(np.arange(T))
G_t = s.rewards.numpy()[:, 0]; coef_t = (s.discounts.numpy() * (1 - s.dones.numpy()))[:, 0]
last_t = s.next_observations.numpy()[:, 0].astype(np.int64) - 100000
coef_o = d['disc']
act = d['active']          # ours: only transitions whose n-step window is complete are sampleable
res = dict(n_compared=int(act.sum()), max_abs_G=float(np.abs(G_t[act] - d['G'][act]).max()), max_abs_coef=float(np.abs(coef_t[act] - coef_o[act]).max()),
           last_obs_mismatch=int((last_t[act] != d['last'][act]).sum()),
           n_term=int(d['term'].sum()), n_trunc=int(d['trunc'].sum()))
# the inactive tail (last n-1 transitions): SB3 samples them with a truncated window, ours never samples them
tail = ~act
res['n_tail_not_sampleable_in_ours'] = int(tail.sum())
print(json.dumps(res, indent=1)); json.dump(res, open('nstep_parity.json', 'w'), indent=1)
