"""Random-policy floor for the normalised-scale judgement : uniform actions in the agent's [-1, 1]
space, Pendulum-V (1000 episodes) and Semi-Circle (20 eval goals x 50 meta-episodes, 2 inner episodes each).
CPU only.  Writes runs/full/random_policy.json."""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import load_config  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.envs import make_env  # noqa: E402


def rollout(env, key, task):
    k0, kr = jax.random.split(key)
    state, _ = env.reset(k0, task)

    def step(c, t):
        state, ret = c
        ka, ks = jax.random.split(jax.random.fold_in(kr, t))
        a = jax.random.uniform(ka, (env.act_dim,), minval=-1.0, maxval=1.0)
        state, _, r, _, _ = env.step(ks, state, task, a)
        return (state, ret + r), None
    (_, ret), _ = jax.lax.scan(step, (state, jnp.zeros(())), jnp.arange(env.H))
    return ret


def main():
    out = {}
    env = make_env(load_config("pendulum_v.yaml"))
    keys = jax.random.split(jax.random.PRNGKey(12345), 1000)
    r = jax.jit(jax.vmap(lambda k: rollout(env, k, jnp.zeros((0,), jnp.float32))))(keys)
    out["Pendulum-V"] = {"mean": float(r.mean()), "se": float(r.std() / jnp.sqrt(r.size)), "n": int(r.size)}
    env = make_env(load_config("semicircle.yaml"))
    tasks = env.make_tasks(jax.random.PRNGKey(0))["eval"]
    keys = jax.random.split(jax.random.PRNGKey(12346), 50 * tasks.shape[0])
    tt = jnp.tile(tasks, (50, 1))
    r = jax.jit(jax.vmap(lambda k, t: rollout(env, k, t)))(keys, tt)
    out["Semi-Circle"] = {"mean": float(r.mean()), "se": float(r.std() / jnp.sqrt(r.size)), "n": int(r.size)}
    with open(os.path.join(common.REPO, "runs", "full", "random_policy.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(out)


if __name__ == "__main__":
    main()
