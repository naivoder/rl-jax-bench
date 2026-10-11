"""Structured logging, device check, in-jit NaN/Inf guard and metrics-file helpers for the JAX reproductions in this repository.

    from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()          # BEFORE jax is imported
    from rl_jax_bench.jaxlog import info, metric, warn, error, check_device, nan_guard, write_json, Timer

* ``info/metric/warn/error`` print one flushed line ``[LEVEL] msg k=v ...``.
* ``check_device()`` logs ``jax.devices()`` and warns loudly on a CPU fallback (unless JAX_PLATFORMS=cpu).
* ``nan_guard(tree, payload, tag, dump_dir)`` runs inside jit: if any floating leaf of ``tree`` is non-finite,
  a ``jax.debug.callback`` dumps ``payload`` to ``dump_dir/nan_<time>.npz`` and aborts the process (exit 3).
* ``write_json`` converts jax/numpy leaves and writes atomically (tmp + rename).
* ``Timer().lap()`` for separating first-compile time from steady-state throughput.

"""
from __future__ import annotations

import functools
import json
import os
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["log", "info", "metric", "warn", "error", "check_device", "nan_guard", "to_py", "write_json", "Timer"]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def log(level: str, msg: str, **kv: Any) -> None:
    """Structured stdout line: ``[LEVEL] msg k=v ...`` (flushed)."""
    extra = " ".join(f"{k}={_fmt(v)}" for k, v in kv.items())
    print(f"[{level}] {msg}" + (f" {extra}" if extra else ""), flush=True)


def info(msg: str, **kv: Any) -> None:
    log("INFO", msg, **kv)


def metric(msg: str, **kv: Any) -> None:
    log("METRIC", msg, **kv)


def warn(msg: str, **kv: Any) -> None:
    log("WARN", msg, **kv)


def error(msg: str, **kv: Any) -> None:
    log("ERROR", msg, **kv)


def check_device() -> str:
    """Log the visible devices; warn loudly if JAX fell back to CPU without being asked to."""
    devs = jax.devices()
    kind = type(devs[0]).__name__ if devs else "none"
    plat = devs[0].platform if devs else "none"
    if plat != "gpu" and os.environ.get("JAX_PLATFORMS", "") != "cpu":
        warn("!!!!! JAX FELL BACK TO CPU -- no CUDA device visible !!!!!", devices=str(devs))
    else:
        info("devices", devices=str(devs), kind=kind)
    return plat


# ---------------------------------------------------------------- NaN / Inf guard
def _nan_host_callback(ok, tag, payload, *, dump_dir: str):
    if bool(ok):
        return
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"nan_{int(time.time())}.npz")
    flat = {f"leaf_{i}": np.asarray(x) for i, x in enumerate(jax.tree_util.tree_leaves(payload))}
    np.savez(path, **flat)
    error("NaN/Inf detected -- batch dumped, aborting", tag=int(tag), dump=path)
    sys.stdout.flush()
    os._exit(3)


def nan_guard(tree_to_check, payload, tag: int = 0, dump_dir: str = "nan_dumps") -> None:
    """Inside jit: if any floating leaf of ``tree_to_check`` is non-finite, dump ``payload`` on host and abort."""
    leaves = [jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree_to_check)
              if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)]
    ok = jnp.all(jnp.stack(leaves)) if leaves else jnp.array(True)
    jax.debug.callback(functools.partial(_nan_host_callback, dump_dir=dump_dir), ok, tag, payload, ordered=False)


# ---------------------------------------------------------------- metrics files
def to_py(x):
    """Recursively convert jax/numpy arrays and scalars to plain Python for json."""
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    if isinstance(x, (np.ndarray, jax.Array)):
        a = np.asarray(x)
        return a.item() if a.ndim == 0 else a.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def write_json(path: str, obj: dict) -> None:
    """Atomic json write (tmp file + os.replace), creating parent directories."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_py(obj), f, indent=1)
    os.replace(tmp, path)


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        t = time.perf_counter()
        dt, self.t0 = t - self.t0, t
        return dt
