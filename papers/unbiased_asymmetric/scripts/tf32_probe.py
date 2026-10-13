"""Does the GPU use TF32 for our float32 matmuls at JAX's default precision?
Compares a float32 matmul of GRU-like shapes at default and 'highest' precision against float64 numpy.
TF32 keeps a 10-bit mantissa (rel. error ~1e-3); full fp32 gives ~1e-7.   python scripts/tf32_probe.py"""
import sys
import time

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

t0 = time.time()
print("[INFO] devices", jax.devices(), "jax", jax.__version__, "default_matmul_precision", jax.config.jax_default_matmul_precision)
rng = np.random.default_rng(0)
for shape in ((200, 128, 384), (200, 64, 64), (2048, 256, 256)):
    a = rng.standard_normal(shape[:2]).astype(np.float32)
    b = rng.standard_normal(shape[1:]).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    for prec in (None, "highest"):
        f = jax.jit(lambda x, y, p=prec: jnp.dot(x, y, precision=p))
        out = np.asarray(f(jnp.asarray(a), jnp.asarray(b)), np.float64)
        err = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
        print(f"[METRIC] shape={shape} precision={prec or 'default'} max_rel_err={err:.2e} "
              f"-> {'TF32-like' if err > 1e-5 else 'fp32'}")
print(f"[INFO] probe wall {time.time() - t0:.1f}s")
