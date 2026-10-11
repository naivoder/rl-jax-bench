"""Init-matched cross-over: the OFFICIAL Backprop + FFNN learner (run_oracle.py bp_traj) started from OUR initial
weights (the init of run s in scripts/oracle/ours_on_official_streams.py: init_mlp(fold_in(PRNGKey(90000+s), 2)))
on the OFFICIAL generator's streams.  Compared with ours_on_official_streams_10.json (same init, same stream), any
difference is numerics only; compared with the official pipeline (torch's own init), the difference is the init draw.

    JAX_PLATFORMS=cpu uv run python -m scripts.oracle.init_matched [--procs 5]
-> runs/oracle_pipeline/init_matched/<case>_seed<s>.npz (official per-10k MSE)
"""
import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np

from scripts.oracle.diff import ORACLE_PY, RUNNER
from scripts.oracle.ours_on_official_streams import CASES, stream
from src.model import INIT_GAIN, init_mlp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "runs", "oracle_pipeline", "init_matched")


def job(case, s):
    out = os.path.join(OUT, f"{case}_seed{s}.npz")
    if os.path.exists(out):
        return
    sdir, act, kw = CASES[case]
    n = int(sdir.split("_")[-1])
    X, Y = stream(sdir, s)
    p = init_mlp(jax.random.fold_in(jax.random.PRNGKey(90_000 + s), 2), 20, (5,), 1, INIT_GAIN[act])
    inp = {"kind": "ffnn", "n_in": 20, "width": 5, "act": act, "opt": "sgd", "lr": kw["lr"], "wd": kw.get("wd", 0.0),
           "X": X[:n], "Y": Y[:n]}
    for i in range(2):
        inp[f"W{i}"] = np.asarray(p["params"][f"layer_{i}"]["kernel"]).T.copy()
        inp[f"b{i}"] = np.asarray(p["params"][f"layer_{i}"]["bias"])
    os.makedirs(OUT, exist_ok=True)
    ip, op = out.replace(".npz", "_in.npz"), out.replace(".npz", "_raw.npz")
    np.savez(ip, **inp)
    t = time.time()
    subprocess.run([ORACLE_PY, RUNNER, "bp_traj", ip, op], check=True, capture_output=True)
    errs = np.load(op)["errs"]
    np.savez(out, per10k=errs.reshape(-1, 10_000).mean(1), first_100k=errs[:100_000])
    os.remove(ip)
    os.remove(op)
    print(f"[INFO] init_matched {case} seed={s} {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cases", nargs="*", default=["flip_tanh_bp_0.01", "flip_relu_bp_0.01", "rs_relu_bp_0.01"])
    a = ap.parse_args()
    jobs = [(c, s) for c in a.cases for s in a.seeds]
    for c, s in jobs:   # materialise the npz streams serially
        stream(CASES[c][0], s)
    with ThreadPoolExecutor(a.procs) as ex:
        list(ex.map(lambda j: job(*j), jobs))
    print("[INFO] init_matched done", flush=True)


if __name__ == "__main__":
    main()
