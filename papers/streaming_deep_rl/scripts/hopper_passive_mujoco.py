import json, numpy as np, gymnasium as gym, mujoco
N, CAP = 32, 1000
res = {"versions": f"gymnasium {gym.__version__} mujoco {mujoco.__version__}"}
env = gym.make("Hopper-v4")
res["dt"] = float(env.unwrapped.dt)
rng = np.random.default_rng(0)
for pol in ("zero", "random", "plus", "minus"):
    R, L = [], []
    for e in range(N):
        env.reset(seed=e); ret = 0; n = 0; done = False
        while not done and n < CAP:
            a = {"zero": np.zeros(3), "plus": 0.5 * np.ones(3), "minus": -0.5 * np.ones(3),
                 "random": rng.uniform(-1, 1, 3)}[pol]
            _, r, te, tr, _ = env.step(a); ret += r; n += 1; done = te
        R.append(ret); L.append(n)
    R, L = np.array(R), np.array(L)
    res[pol] = dict(ret=float(R.mean()), ret_sd=float(R.std()), len=float(L.mean()), len_min=int(L.min()),
                    len_max=int(L.max()), rps=float((R / L).mean()))
    print(pol, res[pol], flush=True)
json.dump(res, open("hopper_passive_mujoco.json", "w"), indent=1)
