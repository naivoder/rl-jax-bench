"""Logging, timing, NaN guard and metrics-file helpers (re-exported from rl_jax_bench.jaxlog) plus git provenance.

Entry points call ``bootstrap()`` BEFORE importing jax (GPU bootstrap, persistent compile cache)."""
from __future__ import annotations

import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The authors' code (allenai/advisor + AllenAct), its venv and reference-run outputs; see scripts/official/setup.sh.
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official"))
_NAN_DUMP_DIR = os.path.join(REPO, "runs", "nan_dumps")


def bootstrap() -> None:
    from rl_jax_bench import gpu_bootstrap
    gpu_bootstrap.setup()


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


def git_provenance() -> dict:
    """git sha + dirty flag for a metrics file. When src/ (or scripts/) has uncommitted changes, also the
    `git diff --stat` and the sha256 of the full `git diff HEAD` of those paths, so a dirty run is still traceable."""
    out = dict(git=git_sha(), dirty=git_dirty())
    if out["dirty"]:
        import hashlib
        try:
            diff = subprocess.check_output(["git", "-C", REPO, "diff", "HEAD", "--", "src", "scripts"],
                                           stderr=subprocess.DEVNULL)
            stat = subprocess.check_output(["git", "-C", REPO, "diff", "HEAD", "--stat", "--", "src", "scripts"],
                                           stderr=subprocess.DEVNULL).decode()
            untracked = subprocess.check_output(["git", "-C", REPO, "ls-files", "--others", "--exclude-standard",
                                                 "src", "scripts"], stderr=subprocess.DEVNULL).decode().split()
            out.update(diff_sha256=hashlib.sha256(diff).hexdigest(), diff_stat=stat.strip().splitlines(),
                       untracked=untracked)
        except Exception as e:  # pragma: no cover
            out.update(diff_error=repr(e))
    return out


def _jl():
    from rl_jax_bench import jaxlog
    return jaxlog


def info(msg, **kv):
    _jl().info(msg, **kv)


def metric(msg, **kv):
    _jl().metric(msg, **kv)


def warn(msg, **kv):
    _jl().warn(msg, **kv)


def error(msg, **kv):
    _jl().error(msg, **kv)


def write_json(path, obj):
    _jl().write_json(path, obj)


def check_device():
    return _jl().check_device()


def Timer():
    return _jl().Timer()


def nan_guard(tree_to_check, payload, tag: int = 0) -> None:
    """Inside jit: abort (exit 3) after dumping ``payload`` if any leaf of ``tree_to_check`` is non-finite."""
    _jl().nan_guard(tree_to_check, payload, tag, dump_dir=_NAN_DUMP_DIR)
