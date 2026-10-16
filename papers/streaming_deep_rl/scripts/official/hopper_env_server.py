"""Env server for the version-matched rule-1 test (): real gymnasium Hopper-v4 stepped INSIDE the authors' venv
(gymnasium 0.29.1, mujoco 2.3.7 -- the versions their reference runs used), driven over a binary pipe by OUR JAX agent
(scripts/ours_on_mujoco_vm.py, main venv). No pickling (numpy 1.26 vs 2.x): raw float64 buffers.
Protocol: argv = seeds. On start: write L*11 float64 (reset(seed=s) observations).
Each step: read L*3 float64 actions -> write L*11 obs, L reward, L terminated, L*11 reset-obs (0 if no reset).
Episode end (as scripts/ours_on_mujoco.py): terminated or our own length counter >= 1000 -> env.reset()."""
import sys

import gymnasium as gym
import mujoco
import numpy as np

seeds = [int(s) for s in sys.argv[1:]]
L = len(seeds)
envs = [gym.make("Hopper-v4") for _ in seeds]
sys.stderr.write(f"[INFO] server gymnasium {gym.__version__} mujoco {mujoco.__version__} L={L}\n")
inp, out = sys.stdin.buffer, sys.stdout.buffer
out.write(np.stack([e.reset(seed=s)[0] for e, s in zip(envs, seeds)]).astype(np.float64).tobytes())
out.flush()
ep_len = np.zeros(L, int)
nb = L * 3 * 8
while True:
    buf = inp.read(nb)
    if len(buf) < nb:
        break
    a = np.frombuffer(buf, np.float64).reshape(L, 3)
    o2 = np.zeros((L, 11)); r = np.zeros(L); te_ = np.zeros(L); rr = np.zeros((L, 11))
    for i, e in enumerate(envs):
        ob, rew, te, tru, _ = e.step(np.clip(a[i], -1.0, 1.0))
        o2[i], r[i], te_[i] = ob, rew, float(te)
        ep_len[i] += 1
        if te or ep_len[i] >= 1000:
            rr[i] = e.reset()[0]
            ep_len[i] = 0
    out.write(o2.tobytes() + r.tobytes() + te_.tobytes() + rr.tobytes())
    out.flush()
