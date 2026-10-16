"""Hopper fidelity probe: episode length / return / reward-per-step under fixed simple policies (zero torque, uniform
random in [-1,1], constant +0.5 / -0.5) in Brax positional (our proxy), Brax generalized, and (separate script) real
MuJoCo Hopper-v4. 32 episodes each, 1000-step cap."""
import sys, json
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import jax, jax.numpy as jnp, numpy as np
from brax import envs as benvs
N, CAP = 32, 1000
res = {}
for backend in sys.argv[1].split(","):
    env = benvs.get_environment("hopper", backend=backend)
    res[backend + "_dt"] = float(env.dt)
    for pol in ("zero", "random", "plus", "minus"):
        def ep(key):
            s0 = env.reset(key)
            def f(c, k):
                s, alive, ret, n = c
                a = {"zero": jnp.zeros(3), "plus": 0.5 * jnp.ones(3), "minus": -0.5 * jnp.ones(3),
                     "random": jax.random.uniform(k, (3,), minval=-1, maxval=1)}[pol]
                s2 = env.step(s, a)
                ret = ret + alive * s2.reward
                n = n + alive
                alive = alive * (1.0 - s2.done)
                return (s2, alive, ret, n), None
            (s, alive, ret, n), _ = jax.lax.scan(f, (s0, 1.0, 0.0, 0.0), jax.random.split(key, CAP))
            return ret, n
        R, L = jax.jit(jax.vmap(ep))(jax.random.split(jax.random.PRNGKey(0), N))
        R, L = np.asarray(R), np.asarray(L)
        res[f"{backend}/{pol}"] = dict(ret=float(R.mean()), ret_sd=float(R.std()), len=float(L.mean()),
                                       len_min=float(L.min()), len_max=float(L.max()), rps=float((R / L).mean()))
        print(backend, pol, res[f"{backend}/{pol}"], flush=True)
json.dump(res, open(f"hopper_passive_{sys.argv[1].replace(',', '_')}.json", "w"), indent=1)
