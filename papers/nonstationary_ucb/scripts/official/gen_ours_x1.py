"""Our X1 grid-a trajectories at long horizon (D-UCB Remark-3 gamma, SW-UCB Remark-9 tau, xi = 0.6, printed 2B),
dumped with reward tables for the SMPyBandits long-horizon replay. Throwaway probe.
Usage: JAX_PLATFORMS=cpu python gen_ours_x1.py T UPS NREPS OUT.npz"""
import sys, math
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # the paper folder
import src  # GPU bootstrap before jax
import numpy as np, jax, jax.numpy as jnp
from src import algorithm as alg, data
T, U, R, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
env = data.alternating(T, U, [0.5, 0.3, 0.4], [0.5, 0.9, 0.4])
K = env.K
means, resets = alg.as_device_env(env.means, env.resets)
ts = jnp.arange(1, T + 1, dtype=jnp.int32)
cases = {"ducb": dict(xi=0.6, gamma=1.0 - math.sqrt(U / T) / 4.0),
         "swucb": dict(xi=0.6, tau=int(round(2.0 * math.sqrt(T * math.log(T) / U))))}
res = {"means": env.means.astype(np.float32)}
for c, kw in cases.items():
    k_env, k_pol = alg.rep_keys(0, 11, T * 1000 + U, alg.POLICY_ID[c], R)
    buf = max(2048, kw.get("tau", 1))
    A, REG, X = [], [], []
    for r in range(R):
        acts, aux, acc, _ = alg.trajectory(c, K, "lowest", buf, alg.make_params(**kw), means, resets, k_env[r], k_pol[r])
        Uu = np.asarray(jax.vmap(lambda t: alg.env_uniforms(k_env[r], t, K))(ts))
        X.append((Uu < env.means).astype(np.uint8)); A.append(np.asarray(acts).astype(np.int8)); REG.append(float(acc.regret))
    res[f"{c}_X"] = np.stack(X); res[f"{c}_acts"] = np.stack(A); res[f"{c}_regret"] = np.array(REG); res[f"{c}_kw"] = np.array(repr(kw))
    print(c, kw, REG, flush=True)
np.savez_compressed(out, **res)
