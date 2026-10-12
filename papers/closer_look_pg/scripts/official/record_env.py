"""Record gymnasium MuJoCo v4 transitions (the closest runnable stand-in for the official gym *-v2 / mujoco-py env)
for an env-level differential against our brax/MJX proxy. Random uniform actions in [-1,1], episodes until 2000
transitions. Saves qpos/qvel before and after, action, gym obs before/after, reward, terminated.
Usage: .venv/bin/python -I record_env.py <Hopper-v4|Walker2d-v4> <seed> <out.npz>"""
import sys
import gymnasium
import numpy as np


def main(name, seed, out, n=2000):
    env = gymnasium.make(name); u = env.unwrapped
    rng = np.random.default_rng(seed)
    o, _ = env.reset(seed=seed)
    rec = {k: [] for k in ["qpos", "qvel", "obs", "act", "qpos1", "qvel1", "obs1", "rew", "term"]}
    while len(rec["act"]) < n:
        q, qd = u.data.qpos.copy(), u.data.qvel.copy()
        a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        o1, r, term, trunc, _ = env.step(a)
        for k, v in zip(rec, [q, qd, o, a, u.data.qpos.copy(), u.data.qvel.copy(), o1, r, term]):
            rec[k].append(v)
        o = o1
        if term or trunc:
            o, _ = env.reset()
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in rec.items()}, dt=u.dt,
                        mujoco_version=__import__("mujoco").__version__, gymnasium_version=gymnasium.__version__)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
