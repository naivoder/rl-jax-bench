"""Differential run: the AUTHORS' stream_td.py classes on ETTm2 from GIVEN initial weights
(torch layout npz written by td_same_init.py), saving per-step predictions/cumulants/deltas.
Usage: <official python> td_same_init_official.py <weights.npz> <out.npz> <T>   (cwd must hold ETTm2.csv)"""
import os
import sys

OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' clone + venv (scripts/official/setup.sh)
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, "repo"))
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import stream_td as st  # noqa: E402

wpath, out, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.set_num_threads(1)
W = np.load(wpath)
env = st.ETTEnvironment()
env = gym.wrappers.RecordEpisodeStatistics(env)
env = st.ObservationTraces(env, beta=0.999)
env = st.NormalizeObservation(env)
env = st.ScaleReward(env, gamma=0.99)
agent = st.StreamTD(n_obs=7, lr=1.0, gamma=0.99, lamda=0.8, kappa_value=2.0)
with torch.no_grad():
    for n, p in agent.named_parameters():
        p.copy_(torch.from_numpy(W[n]))
s, _ = env.reset()
P, C, S = [], [], []
for _ in range(T):
    s_prime, c, terminated, _, _ = env.step(None)
    agent.update_params(s, c, s_prime, terminated)
    s = s_prime
    P.append(agent.predict(s) * np.sqrt(env.reward_stats.var + 1e-8).squeeze())
    C.append(c * np.sqrt(env.reward_stats.var + 1e-8).squeeze())
    S.append(np.asarray(s, float))
np.savez(out, pred=np.array(P, float), cum=np.array(C, float), obs=np.array(S))
print("[INFO] saved", out)
