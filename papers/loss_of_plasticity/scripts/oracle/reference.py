"""Oracle reference runs: the OFFICIAL learner classes (Backprop / ContinualBackprop + FFNN, torch's own init and
torch's own CBP randomness) trained on OUR Slowly-Changing-Regression streams (identical X, Y for a given seed).

    JAX_PLATFORMS=cpu uv run python -m scripts.oracle.reference [--steps 1000000] [--procs 5] [--seeds 0 1 2 3 4]

Outputs runs/oracle_ref/<case>/seed_<s>.npz (mse_per_10k).  Distributional comparison with our runs in eval.py.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np

from src import data as D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "runs", "oracle_ref")
OFF = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
ORACLE_PY = os.path.join(OFF, ".venv", "bin", "python")
RUNNER = os.path.join(REPO, "scripts", "oracle", "run_oracle.py")

# name: (scr mode, act, opt, lr, rho, mt, util_type, wd, sigma) -- each matches an arm of one of our configs
CASES = {
    "relu_sgd_0.01_bp_flip": ("flip_one", "relu", "sgd", 0.01, 0.0, 10, "adaptable_contribution", 0.0, 0.0),
    "tanh_sgd_0.01_bp_flip": ("flip_one", "tanh", "sgd", 0.01, 0.0, 10, "adaptable_contribution", 0.0, 0.0),
    "tanh_adam_0.01_bp_flip": ("flip_one", "tanh", "adam", 0.01, 0.0, 10, "adaptable_contribution", 0.0, 0.0),
    "tanh_adam_0.01_cbp1e-4_flip": ("flip_one", "tanh", "adam", 0.01, 1e-4, 10, "adaptable_contribution", 0.0, 0.0),
    "relu_sgd_0.01_bp_resample": ("resample", "relu", "sgd", 0.01, 0.0, 10, "adaptable_contribution", 0.0, 0.0),
    "relu_sgd_0.01_cbp1e-4_resample": ("resample", "relu", "sgd", 0.01, 1e-4, 10, "adaptable_contribution", 0.0, 0.0),
    "tanh_adam_0.01_cbp1e-4_resample": ("resample", "tanh", "adam", 0.01, 1e-4, 10, "adaptable_contribution", 0.0, 0.0),
    "tanh_adam_0.01_adapt1e-4_resample": ("resample", "tanh", "adam", 0.01, 1e-4, 10, "adaptation", 0.0, 0.0),
}


def export(mode, seed, steps):
    path = os.path.join(OUT, "streams", f"{mode}_seed{seed}_{steps}.npz")
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    spec = D.scr_spec(mode)
    root = jax.random.PRNGKey(seed)
    X, Y, _ = D.scr_stream(jax.random.fold_in(root, 0), jax.random.fold_in(root, 1), spec, steps // spec.T)
    np.savez(path, X=np.asarray(X).astype(np.uint8), Y=np.asarray(Y))
    return path


def run_one(case, seed, steps):
    mode, act, opt, lr, rho, mt, ut, wd, sigma = CASES[case]
    out = os.path.join(OUT, case, f"seed_{seed}.npz")
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stream = dict(np.load(export(mode, seed, steps)))
    inp = dict(stream, act=act, opt=opt, lr=lr, rho=rho, mt=mt, util_type=ut, wd=wd, sigma=sigma,
               torch_seed=10_000 + seed)
    ip = out.replace(".npz", "_in.npz")
    np.savez(ip, **inp)
    t = time.time()
    subprocess.run([ORACLE_PY, RUNNER, "cbp_run", ip, out], check=True, capture_output=True)
    os.remove(ip)
    print(f"[INFO] oracle_ref {case} seed={seed} {time.time() - t:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    a = ap.parse_args()
    for case in a.cases:
        for s in a.seeds:
            export(CASES[case][0], s, a.steps)
    jobs = [(c, s) for c in a.cases for s in a.seeds]
    with ThreadPoolExecutor(a.procs) as ex:
        list(ex.map(lambda j: run_one(j[0], j[1], a.steps), jobs))
    print("[INFO] oracle_ref done", flush=True)


if __name__ == "__main__":
    main()
