"""Measure brax-mjx env-step throughput at several batch widths (budget planning, README budget lines)."""
import sys, time
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import warnings
import jax, jax.numpy as jnp
from rl_jax_bench.jaxlog import info, metric, check_device
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from brax import envs

check_device()
name = sys.argv[1]; widths = [int(w) for w in sys.argv[2].split(',')]
env = envs.get_environment(name, backend='mjx')
info("env", name=name, obs=env.observation_size, act=env.action_size)
for W in widths:
    keys = jax.random.split(jax.random.PRNGKey(0), W)
    st = jax.jit(jax.vmap(env.reset))(keys)
    @jax.jit
    def run(st, k):
        def body(c, kk):
            a = jax.random.uniform(kk, (W, env.action_size), minval=-1, maxval=1)
            c = jax.vmap(env.step)(c, a)
            return c, c.reward.sum()
        return jax.lax.scan(body, st, jax.random.split(k, 50))
    t = time.time(); out = run(st, jax.random.PRNGKey(1)); jax.block_until_ready(out); tc = time.time() - t
    t = time.time(); out = run(out[0], jax.random.PRNGKey(2)); jax.block_until_ready(out); dt = time.time() - t
    metric("bench", env=name, width=W, compile_s=tc, ms_per_step=1000 * dt / 50, steps_per_s=50 * W / dt)
