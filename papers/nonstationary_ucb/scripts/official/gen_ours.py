"""Dump OUR trajectories (actions, indices/probs, regret) plus the exact reward tables and EXP3.S draw
uniforms, so the SMPyBandits oracle can replay the same inputs. Throwaway recheck probe.
Usage: JAX_PLATFORMS=cpu uv run python gen_ours.py OUT.npz NREPS
"""
import sys, os
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # the paper folder
import src  # GPU bootstrap before jax
import numpy as np
import jax, jax.numpy as jnp
from src import algorithm as alg, data
F64 = jnp.float64

out, R = sys.argv[1], int(sys.argv[2])
T = 10_000
envs = {"e1": (data.abrupt(T, [[.5, .3, .4], [.5, .9, .4], [.5, .3, .4]], [3000, 5000]), 1),
        "e2": (data.periodic(T), 2)}
ts = jnp.arange(1, T + 1, dtype=jnp.int32)
res = {}
for ename, (env, exp_id) in envs.items():
    K = env.K
    means, resets = alg.as_device_env(env.means, env.resets)
    res[f"{ename}_means"] = env.means
    cases = {"ucb1": ("ucb1", dict(xi=0.5)),
             "ducb": ("ducb", dict(xi=0.5, gamma=alg.paper_gamma(T))),
             "ducb_padB": ("ducb", dict(xi=0.125, gamma=alg.paper_gamma(T))),
             "swucb": ("swucb", dict(xi=0.5, tau=alg.paper_tau(T))),
             "exp3s": ("exp3s", dict(g=alg.exp3s_g(K, 2, T), alpha=1.0 / T))}
    for case, (name, kw) in cases.items():
        k_env, k_pol = alg.rep_keys(0, exp_id, 0, alg.POLICY_ID[name], R)
        traj = jax.vmap(lambda ke, kp: alg.trajectory(name, K, "lowest", 2048, alg.make_params(**kw), means, resets, ke, kp))
        acts, aux, acc, _ = traj(k_env, k_pol)
        U = np.asarray(jax.vmap(lambda ke: jax.vmap(lambda t: alg.env_uniforms(ke, t, K))(ts))(k_env))
        X = (U < env.means[None]).astype(np.uint8)
        Up = np.asarray(jax.vmap(lambda kp: jax.vmap(lambda t: jax.random.uniform(jax.random.fold_in(kp, t), (), F64))(ts))(k_pol))
        res[f"{ename}_{case}_X"] = X
        res[f"{ename}_{case}_Up"] = Up
        res[f"{ename}_{case}_acts"] = np.asarray(acts)
        res[f"{ename}_{case}_aux"] = np.asarray(aux)
        res[f"{ename}_{case}_regret"] = np.asarray(acc.regret)
        res[f"{ename}_{case}_kw"] = np.array(repr(kw))
        print(ename, case, kw, "mean regret", float(np.mean(acc.regret)), flush=True)
np.savez_compressed(out, **res)
