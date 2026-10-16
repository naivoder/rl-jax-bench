"""Reference: the AUTHORS' stream_td.py classes (ETTEnvironment, ObservationTraces,
NormalizeObservation, ScaleReward, StreamTD) driven by a copy of its main() loop that SAVES the per-step predictions and
cumulants instead of only plotting them (official main() writes PDFs only). Run in the official venv from a scratch dir
that holds ETTm2.csv (the official env downloads it to the cwd and deletes it on close; we never call close).
Usage: <official python> run_official_td.py <out.npz> <seed> [total_steps]"""
import os
import sys

OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' clone + venv (scripts/official/setup.sh)
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, "repo"))
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import stream_td as st  # noqa: E402

out, seed = sys.argv[1], int(sys.argv[2])
T = int(sys.argv[3]) if len(sys.argv) > 3 else 68_000
gamma, lamda = 0.99, 0.8
torch.set_num_threads(1)
torch.manual_seed(seed)
np.random.seed(seed)
env = st.ETTEnvironment()
env = gym.wrappers.RecordEpisodeStatistics(env)
env = st.ObservationTraces(env, beta=0.999)
env = st.NormalizeObservation(env)
env = st.ScaleReward(env, gamma=gamma)
agent = st.StreamTD(n_obs=env.observation_space.shape[0], lr=1.0, gamma=gamma, lamda=lamda, kappa_value=2.0)
s, _ = env.reset()
cumulants, predictions = [], []
for _ in range(T):
    s_prime, c, terminated, _, _ = env.step(None)
    agent.update_params(s, c, s_prime, terminated)
    s = s_prime
    predictions.append(agent.predict(s) * np.sqrt(env.reward_stats.var + 1e-8).squeeze())
    cumulants.append(c * np.sqrt(env.reward_stats.var + 1e-8).squeeze())
    if terminated:
        break
np.savez(out, pred=np.array(predictions, float), cum=np.array(cumulants, float))
print("[INFO] saved", out, len(predictions))
