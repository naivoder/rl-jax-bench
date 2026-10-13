"""ORACLE (official venv): does the official Cleaner terminate when the floor is clean?

Paper App. C.2.2 says "The task is complete when every cell in the grid has been visited by at least one agent";
the official EnvCleaner.step returns done = (i_step >= 200) only. This drives the official env
(extra-cleaner-v0 + TimeLimit(100)) with a greedy covering policy (each agent BFS-walks to its nearest dirty cell)
until every dirty cell is clean, then keeps stepping to the cap, and records the per-step done flag, reward and
number of dirty cells left -> tests/fixtures/official/cleaner_fullclean.npz (committed; tests/test_official_parity.py)."""
import os
import sys
import types
import warnings
from collections import deque

warnings.filterwarnings("ignore")
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official"))
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, "asym-rlpo-aamas2022"))
sys.modules["gym.envs.classic_control.rendering"] = types.ModuleType("rendering")   # headless shim
import numpy as np

from asym_rlpo.envs import make_env

HERE = os.path.dirname(os.path.abspath(__file__))
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # official action ids: up, down, left, right


def first_move(occ, start, avoid=None):
    """BFS over free cells (occ != 1) to the nearest dirty cell (occ == 2), excluding `avoid`; first action id."""
    q, seen = deque([(tuple(start), None)]), {tuple(start)}
    while q:
        p, a0 = q.popleft()
        if a0 is not None and occ[p] == 2 and p != avoid:
            return a0, p
        for a, (di, dj) in enumerate(MOVES):
            n = (p[0] + di, p[1] + dj)
            if occ[n] != 1 and n not in seen:
                seen.add(n)
                q.append((n, a if a0 is None else a0))
    return 0, None


env = make_env("extra-cleaner-v0", max_episode_timesteps=100)
env.reset()
core = env
while not hasattr(core, "occupancy"):
    core = core.env
ACT, RW, DN, LEFT = [], [], [], []
for t in range(100):
    occ = np.asarray(core.occupancy)
    a0, tgt0 = first_move(occ, core.agt_pos_list[0])
    a1, _ = first_move(occ, core.agt_pos_list[1], avoid=tgt0)
    a = 4 * a0 + a1
    o, r, d, _ = env.step(a)
    ACT.append(a); RW.append(r); DN.append(bool(d)); LEFT.append(int((np.asarray(core.occupancy) == 2).sum()))
    if d:
        break
ACT, RW, DN, LEFT = map(np.asarray, (ACT, RW, DN, LEFT))
np.savez_compressed(os.path.join(PAPER, "tests", "fixtures", "official", "cleaner_fullclean.npz"),
                    act=ACT, rw=RW, dn=DN, left=LEFT)
k = int(np.argmax(LEFT == 0)) if (LEFT == 0).any() else None
print("steps", len(ACT), "return", RW.sum(), "all clean after step", None if k is None else k + 1,
      "done flags True at", np.nonzero(DN)[0].tolist())
