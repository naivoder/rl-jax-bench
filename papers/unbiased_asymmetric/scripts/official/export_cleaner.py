"""ORACLE (official venv): Cleaner (asym-rlpo@1a065b0 extra-cleaner-v0 = SingleAgentWrapper(EnvCleaner_Fix(EnvCleaner()))
+ TimeLimit(100)). Exports (1) the fixed seed-5 maze as a data fixture (tests/fixtures/cleaner_maze_seed5.npy) and (2)
state/observation/reward streams on random joint actions -> tests/test_official_cleaner.py."""
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

HERE = os.path.dirname(os.path.abspath(__file__))
env = make_env("extra-cleaner-v0", max_episode_timesteps=100)
o0 = env.reset()
inner = env.unwrapped.env if hasattr(env.unwrapped, "env") else None
core = env
while not hasattr(core, "occupancy"):
    core = core.env
maze = core.generate_maze(core.seed).astype(np.int8)
np.save(os.path.join(PAPER, "tests", "fixtures", "cleaner_maze_seed5.npy"), maze)
N, CAP = 200, 100
rng = np.random.default_rng(11)
S0 = np.zeros((N, 845), np.float32)
O0 = np.zeros((N, 54), np.float32)
ACT = np.zeros((N, CAP), np.int64)
ST = np.zeros((N, CAP, 845), np.float32)
OB = np.zeros((N, CAP, 54), np.float32)
RW = np.zeros((N, CAP))
DN = np.zeros((N, CAP), bool)
for i in range(N):
    O0[i] = env.reset()
    S0[i] = env.state
    for t in range(CAP):
        a = int(rng.integers(0, 16))
        o, r, d, _ = env.step(a)
        ACT[i, t], OB[i, t], RW[i, t], DN[i, t], ST[i, t] = a, o, r, d, env.state
        if d:
            break
np.savez_compressed(os.path.join(OFFICIAL_ROOT, "out", "cleaner_traces.npz"), s0=S0, o0=O0, act=ACT, st=ST, ob=OB, rw=RW, dn=DN)
print("maze free cells", int((maze == 2).sum()), "mean return", RW.sum(1).mean(), "done at", DN.argmax(1)[:3])
