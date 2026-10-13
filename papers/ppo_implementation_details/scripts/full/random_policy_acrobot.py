"""DIVERGENT check (full-scale, Acrobot): success rate and return of the uniform-random policy on OUR env (gymnax
Acrobot-v1, float32). Compared with gymnasium 0.29.1 Acrobot-v1 (float64) by
scripts/official/random_policy_acrobot_gym.py. If our env is harder for a random policy, early lucky successes
are rarer and a seed is more likely to never learn. Usage: python -m scripts.full.random_policy_acrobot [n_episodes]"""
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
import json  # noqa: E402
import os  # noqa: E402

import gymnax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
env, params = gymnax.make("Acrobot-v1")
T = 500


def episode(key):
    k0, k1 = jax.random.split(key)
    obs, s = env.reset_env(k0, params)

    def body(c, t):
        s, ret, alive, length = c
        ka, ks = jax.random.split(jax.random.fold_in(k1, t))
        a = jax.random.randint(ka, (), 0, 3)
        o2, s2, r, d, _ = env.step_env(ks, s, a, params)
        ret = ret + alive * r
        length = length + alive
        alive = alive * (1.0 - d.astype(jnp.float32))
        return (s2, ret, alive, length), None

    (_, ret, alive, length), _ = jax.lax.scan(body, (s, 0.0, 1.0, 0.0), jnp.arange(T))
    return ret, alive, length


ret, alive, length = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(2026), n))
ret, alive = np.asarray(ret), np.asarray(alive)
succ = 1.0 - alive  # terminated (reached the goal) before the 500-step limit
out = dict(env="gymnax Acrobot-v1 (float32)", n=n, success_rate=float(succ.mean()),
           success_se=float(np.sqrt(succ.mean() * (1 - succ.mean()) / n)), mean_return=float(ret.mean()),
           return_se=float(ret.std(ddof=1) / np.sqrt(n)), device=str(jax.devices()[0]))
print("[METRIC]", json.dumps(out))
p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs", "full",
                 "divergent", "random_policy_gymnax.json")
os.makedirs(os.path.dirname(p), exist_ok=True)
json.dump(out, open(p, "w"), indent=1)
