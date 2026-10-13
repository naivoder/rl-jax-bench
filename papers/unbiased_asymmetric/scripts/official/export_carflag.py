"""ORACLE (official venv): Car-Flag (asym-rlpo@1a065b0 extra-car-flag-v0 = CarEnvWrapper(CarEnv(), num_actions=7),
TimeLimit(100)) streams on random action sequences -> tests/test_official_carflag.py (float64 replay)."""
import os
import sys
import types
import warnings

warnings.filterwarnings("ignore")
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official"))
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, "asym-rlpo-aamas2022"))
sys.modules["gym.envs.classic_control.rendering"] = types.ModuleType("rendering")   # headless shim
import numpy as np

from asym_rlpo.envs import make_env

N, CAP = 1000, 100
env = make_env("extra-car-flag-v0", max_episode_timesteps=CAP)
env.seed(0)
rng = np.random.default_rng(7)
S0 = np.zeros((N, 3))
O0 = np.zeros((N, 3))
ACT = np.full((N, CAP), -1, np.int64)
ST = np.zeros((N, CAP, 3))
OB = np.zeros((N, CAP, 3))
RW = np.zeros((N, CAP))
DN = np.zeros((N, CAP), bool)
LEN = np.zeros(N, np.int64)
for i in range(N):
    O0[i] = env.reset()
    S0[i] = env.state
    bias = rng.integers(0, 7)          # mix of drifting and random policies so episodes reach the flags
    for t in range(CAP):
        a = int(bias if rng.random() < 0.7 else rng.integers(0, 7))
        o, r, d, _ = env.step(a)
        ACT[i, t], OB[i, t], RW[i, t], DN[i, t], ST[i, t] = a, o, r, d, env.state
        if d:
            LEN[i] = t + 1
            break
np.savez_compressed(os.path.join(OFFICIAL_ROOT, "out", "carflag_traces.npz"),
                    s0=S0, o0=O0, act=ACT, st=ST, ob=OB, rw=RW, dn=DN, len=LEN)
print("mean len", LEN.mean(), "terminal", int((np.abs(RW).sum(1) > 0).sum()), "pos rew", int((RW.sum(1) > 0).sum()))
