"""Shared helpers: GPU bootstrap (must run before jax is imported), logging, config loading, git info.

Every entry point does ``from src import common`` (or imports a module that does) BEFORE importing jax.
The bootstrap preloads the host NVIDIA libraries (flatpak sandbox), sets the persistent compile cache and
``--xla_gpu_deterministic_ops``.  float32 everywhere; matmuls at HIGHEST precision (no TF32) so that the
differential tests against the PyTorch oracle and the GPU runs use the same arithmetic (README "dtype policy").
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import jax  # noqa: E402

# Tests / differential checks: HIGHEST (true fp32).  Training entry points may switch to TF32 via the config key
# ``matmul_precision`` (README dtype policy); set_precision() is called by train.main before tracing anything.
jax.config.update("jax_default_matmul_precision", "highest")


def set_precision(p: str) -> None:
    jax.config.update("jax_default_matmul_precision", p)

from rl_jax_bench.jaxlog import Timer, check_device, error, info, metric, nan_guard, to_py, warn, write_json  # noqa: E402,F401

import yaml  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO, "src", "configs")


def load_config(path: str, overrides=()) -> dict:
    """YAML config with an optional ``base:`` key (another file in the same directory), then k=v overrides."""
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(CONFIG_DIR, path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        base.update(cfg)
        cfg = base
    for ov in overrides:
        k, v = ov.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "src"], text=True,
                                      stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return True


def src_tree_sha256() -> str:
    """sha256 over sorted src/*.py + src/configs/*.yaml (path + bytes)."""
    h = hashlib.sha256()
    for sub in ("src", "src/configs"):
        d = os.path.join(REPO, sub)
        for fn in sorted(os.listdir(d)):
            if fn.endswith((".py", ".yaml")):
                h.update(f"{sub}/{fn}".encode())
                with open(os.path.join(d, fn), "rb") as f:
                    h.update(f.read())
    return h.hexdigest()
