"""Logging, timing, NaN guard and metrics-file helpers shared by every entry point.

The generic pieces live in ``rl_jax_bench.jaxlog`` (shared across the papers in this repository) and are re-exported here;
only the repo-specific bits (git provenance, the NaN dump directory) are defined locally.
"""
from __future__ import annotations

import os
import subprocess
import sys

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
    """Inside jit: if any leaf of ``tree_to_check`` is non-finite, dump ``payload`` to runs/nan_dumps and abort."""
    _jl.nan_guard(tree_to_check, payload, tag, dump_dir=_NAN_DUMP_DIR)
