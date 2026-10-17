"""Throughput A/B of learner-loop variants (same lanes, same process). Usage: python scripts/bench.py [arm ...]"""
import sys, os, time, dataclasses
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import jax, jax.numpy as jnp
import train as TR, data as D
from model import arm_config
cfg = TR.load_cfg(os.path.join(os.path.dirname(TR.__file__), "configs", "default.yaml"))
cfg["eval_cpis"] = 70
lanes, ids = TR.build_lanes(cfg, list(D.EXPERIMENTS), ["eq3", "lin"], list(range(10)))
for arm in (sys.argv[1:] or ["dqn", "ddrqn"]):
    for fuse in (True, False):
        q = dataclasses.replace(arm_config(arm, cfg), fuse_target_unroll=fuse)
        fn = TR.make_chunk_fn(q, 500, 1, True)
        c = jax.jit(jax.vmap(lambda ln: TR.init_carry(ln, q)))(lanes)
        c, o, _ = fn(c, lanes, jnp.int32(0)); jax.block_until_ready(o)
        t = time.perf_counter()
        for k in range(1, 3):
            c, o, _ = fn(c, lanes, jnp.int32(k)); jax.block_until_ready(o)
        dt = (time.perf_counter() - t) / 2
        print(f"[METRIC] bench arm={arm} fuse={fuse} lanes={len(ids)} ms_per_step={1000*dt/1000:.3f} est_570k_min={570*dt/60:.1f}", flush=True)
