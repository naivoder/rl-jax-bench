"""ORACLE (run in the official venv): roll the official gym-pomdps@b52e520 envs (+ asym-rlpo@1a065b0 FlatPaddingWrapper,
gym TimeLimit(100)) on random action sequences from forced start states; save the (state, obs, reward, done) streams.

    $OFFICIAL_ROOT/.venv/bin/python scripts/official/export_env_traces.py

tests/test_official_env.py replays the same actions in our JAX env and requires identical streams.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official"))
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, "asym-rlpo-aamas2022"))
import numpy as np
from gym.wrappers import TimeLimit
from gym_pomdps.envs.pomdp import POMDP

from asym_rlpo.wrapper import FlatPaddingWrapper

FIX = os.path.join(PAPER, "tests", "fixtures")
FILES = {"heavenhell_3": "heavenhell_3.pomdp", "heavenhell_4": "heavenhell_4.pomdp",
         "shopping_5": "shopping_5.v1.pomdp", "shopping_6": "shopping_6.v1.pomdp"}
N_SEQ, CAP = 1000, 100


def main():
    out = {}
    for name, f in FILES.items():
        env = TimeLimit(FlatPaddingWrapper(POMDP(open(os.path.join(FIX, f)).read(), episodic=True, seed=0)), CAP)
        un = env.unwrapped
        rng = np.random.default_rng(12345)
        starts = np.nonzero(un.start)[0]
        S0 = np.zeros(N_SEQ, np.int64)
        ACT = np.full((N_SEQ, CAP), -1, np.int64)
        ST = np.full((N_SEQ, CAP + 1), -2, np.int64)
        OB = np.full((N_SEQ, CAP + 1), -2, np.int64)
        RW = np.zeros((N_SEQ, CAP))
        DN = np.zeros((N_SEQ, CAP), bool)
        LEN = np.zeros(N_SEQ, np.int64)
        for i in range(N_SEQ):
            obs = env.reset()
            un.state = int(rng.choice(starts))
            S0[i] = un.state
            ST[i, 0], OB[i, 0] = un.state, obs
            # heaven-hell: bias towards exploring (random actions); shopping: uniform
            for t in range(CAP):
                a = int(rng.integers(un.action_space.n))
                obs, r, done, _ = env.step(a)
                ACT[i, t], RW[i, t], DN[i, t] = a, r, done
                ST[i, t + 1], OB[i, t + 1] = un.state, obs
                if done:
                    LEN[i] = t + 1
                    break
        out.update({f"{name}/{k}": v for k, v in dict(s0=S0, act=ACT, st=ST, ob=OB, rw=RW, dn=DN, len=LEN).items()})
        # parsed arrays (rl_parsers@b2197e3 via gym-pomdps): T [S,A,S'], O [S,A,S',O] -> [A,S',O], R const over (s',o)
        m = un.model
        R = m.R  # [A, S, S', O]
        assert np.all(R == R[:, :, :1, :1]), "reward depends on (s', o)"
        out.update({f"{name}/T": un.T, f"{name}/O": m.O, f"{name}/R": R[:, :, 0, 0].T, f"{name}/D": un.D,
                    f"{name}/start": un.start})
        print(name, "mean len", LEN.mean(), "terminal-by-env", int((LEN < CAP).sum()), "reset obs", OB[0, 0])
    np.savez_compressed(os.path.join(OFFICIAL_ROOT, "out", "env_traces.npz"), **out)


if __name__ == "__main__":
    main()
