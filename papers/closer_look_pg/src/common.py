"""Shared runtime helpers: GPU bootstrap (must run before jax import), logging re-exports, PRNG key tree, git info.

PRNG key tree (one root seed per run):
    root = PRNGKey(seed)
    k_init  = fold_in(root, 0)                         -> nets.init split {pi, v}
    k_train = fold_in(root, 1) -> fold_in(., it)       -> split {rollout, update}; update -> split {value perms, policy perms|fisher subset}
    k_instr = fold_in(root, 2) -> fold_in(., it)       -> per checkpoint / logged iteration
        fold_in(., 0) k_ref      reference-gradient rollouts
        fold_in(., 1) k_est      -> fold_in(n_index) -> fold_in(rep i)   repeated estimates
        fold_in(., 2) k_vstar    V* data + init
        fold_in(., 3) k_heldout  value-MRE held-out rollout
        fold_in(., 4) k_land     -> fold_in 0 random dir u, 1 true-reward rollouts, 2 many-sample step data, 3 few-sample step data
        fold_in(., 5) k_eval     10-episode stochastic evaluation
    k_boot  = fold_in(root, 3)                         numpy bootstrap seeds
"""
from __future__ import annotations

import os
import subprocess

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup(deterministic=False)  # deterministic scatter makes MJX ~15x slower (README dtype/perf policy)

import jax  # noqa: E402

# float32 policy: XLA uses TF32 for f32 matmuls on Ampere GPUs by default (measured max error 3.3e-2 on a 512^2
# product vs 8.9e-5 at 'highest'); the instruments (and MJX physics on the GPU) must run in true fp32.
jax.config.update("jax_default_matmul_precision", "highest")

from rl_jax_bench.jaxlog import Timer, check_device, error, info, metric, nan_guard, warn, write_json  # noqa: E402,F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

K_INIT, K_TRAIN, K_INSTR, K_BOOT = 0, 1, 2, 3
I_REF, I_EST, I_VSTAR, I_HELDOUT, I_LAND, I_EVAL = 0, 1, 2, 3, 4, 5


def k_instr(root, it, branch):
    return jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(root, K_INSTR), it), branch)


def git_info() -> dict:
    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "src", "tests"], text=True).strip())
    except Exception:  # noqa: BLE001
        sha, dirty = "unknown", True
    return {"git_sha": sha, "git_dirty": dirty}
