"""Micro-benchmark: the vmapped update alone (80 lanes), per arm and precision. Usage: python scripts/bench_update.py"""
import sys, os, time, dataclasses
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import jax, jax.numpy as jnp
import algorithm as alg, train as TR
from model import arm_config
cfg = TR.load_cfg(os.path.join(os.path.dirname(TR.__file__), "configs", "default.yaml"))
NL = int(os.environ.get("NL", 80))
for arm in os.environ.get("ARMS", "dqn,ddrqn,ddqn_tr10").split(","):
    q = arm_config(arm, cfg)
    L = q.trace_len
    keys = jax.random.split(jax.random.PRNGKey(0), NL)
    st = jax.vmap(lambda k: alg.init(k, q))(keys)
    batch = dict(inputs=jax.random.uniform(keys[0], (NL, 32, L + 1, 7)), actions=jnp.zeros((NL, 32, L), jnp.int32),
                 rewards=jnp.ones((NL, 32, L)), discounts=jnp.full((NL, 32, L), 0.9), resets=jnp.zeros((NL, 32, L + 1), bool))
    for prec in ["default"]:
        with jax.default_matmul_precision(prec):
            @jax.jit
            def many(st, batch):
                def body(s, _):
                    s, inf = jax.vmap(lambda a, b: alg.update(a, b, q))(s, batch)
                    return s, inf["loss"]
                return jax.lax.scan(body, st, None, length=200)
            s2, l = many(st, batch); jax.block_until_ready(l)
            t = time.perf_counter(); s2, l = many(st, batch); jax.block_until_ready(l); dt = (time.perf_counter() - t) / 200
            print(f"[METRIC] update-only arm={arm} prec={prec} ms={1000*dt:.3f}", flush=True)
