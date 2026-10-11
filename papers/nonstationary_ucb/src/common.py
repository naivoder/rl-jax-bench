"""Logging, timing, NaN guard, provenance and metrics-file helpers shared by every entry point.

Generic pieces come from ``rl_jax_bench.jaxlog``; repo-specific bits (git provenance, the NaN dump
directory, float64) are defined here. Importing this module turns on ``jax_enable_x64``.
"""
from __future__ import annotations

import os
import subprocess
import sys

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # float64 everywhere (exact ties, bit-comparable references)

from rl_jax_bench import jaxlog as _jl  # noqa: E402
from rl_jax_bench.jaxlog import Timer, check_device, error, info, log, metric, to_py, warn, write_json  # noqa: E402,F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAN_DUMP_DIR = os.path.join(REPO, "runs", "nan_dumps")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "src"], stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return True


def nan_guard(tree_to_check, payload, tag: int = 0) -> None:
    """Inside jit: if any floating leaf of ``tree_to_check`` is non-finite, dump ``payload`` and abort (exit 3)."""
    _jl.nan_guard(tree_to_check, payload, tag, dump_dir=_NAN_DUMP_DIR)
