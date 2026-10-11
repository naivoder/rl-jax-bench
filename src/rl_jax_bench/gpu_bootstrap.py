"""Make the host NVIDIA driver libs visible to JAX inside the VS Code flatpak sandbox.

Preloads the full
same-version ``libnvidia-*`` + ``libcuda`` companion set with RTLD_GLOBAL so cuInit
does not fail with CUDA error 303. No-op on a normal host shell. Also sets the
shared persistent compile cache and the determinism flags. Call BEFORE importing jax.
"""
from __future__ import annotations

import ctypes
import glob
import os

_HOST_LIB = "/run/host/usr/lib/x86_64-linux-gnu"
_SKIP = ("eglcore", "glcore", "glsi", "glvkspirv", "encode", "opticalflow", "present",
         "fbc", "ngx", "tls", "container", "vulkan", "wayland", "egl-", "ifr", "vksc")
_CACHE = os.path.expanduser(os.environ.get("RL_JAX_BENCH_CACHE", "~/.cache/rl-jax-bench/jax"))


def setup(deterministic: bool = True, preallocate: bool = False) -> None:
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE)
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.05")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if not preallocate:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if deterministic and "--xla_gpu_deterministic_ops" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()
    if os.environ.get("CODE_GPU_BOOTSTRAPPED") == "1" or not os.path.isdir(_HOST_LIB):
        return
    os.environ["CODE_GPU_BOOTSTRAPPED"] = "1"
    pending = [p for p in sorted(glob.glob(os.path.join(_HOST_LIB, "libnvidia-*.so.*")))
               + sorted(glob.glob(os.path.join(_HOST_LIB, "libcuda.so.*")))
               if os.path.basename(p).count(".") >= 3 and not any(s in os.path.basename(p) for s in _SKIP)]
    for _ in range(2):
        failed = []
        for p in pending:
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(p)
        pending = failed
        if not pending:
            break
