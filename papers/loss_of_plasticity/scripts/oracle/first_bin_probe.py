"""full-scale probe for the Fig. B.10 FIRST-40k-bin gaps (ours above the paper's plot read).

Two checks, both with the OFFICIAL FFNN + Backprop (run_oracle.py, official code @ a6b7958, CPU) on OUR flip-one streams:
  (a) torch's own init (torch.manual_seed(20000 + s)), streams 0..29, first 40k examples -> first-bin distribution of
      the authors' learner on the identical data, compared with our full_scr_b10 first bins for the same streams;
  (b) init-matched: the official learner started from OUR initial weights for streams 0..4 -> per-step agreement
      (if identical, the first-bin level is fully determined by the shared definition of problem + learner).

    JAX_PLATFORMS=cpu uv run python -m scripts.oracle.first_bin_probe [--procs 5]
-> runs/oracle_ref/first_bin_probe.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np

from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402

from scripts.oracle.diff import ORACLE_PY, RUNNER  # noqa: E402
from scripts.oracle.reference import export  # noqa: E402
from src.eval import load  # noqa: E402
from src.model import INIT_GAIN, init_mlp  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "runs", "oracle_ref", "first_bin")
STEPS = 40_000
ARMS = ["sigmoid_0.01", "elu_0.01", "elu_0.003", "swish_0.001", "sigmoid_0.001", "relu_0.01", "tanh_0.01"]


def split(arm):
    a, lr = arm.rsplit("_", 1)
    return a, float(lr)


def torch_init(arm, s):
    out = os.path.join(OUT, f"torchinit_{arm}_seed{s}.npz")
    if os.path.exists(out):
        return
    act, lr = split(arm)
    stream = dict(np.load(export("flip_one", s, STEPS)))
    ip = out.replace(".npz", "_in.npz")
    np.savez(ip, **stream, act=act, opt="sgd", lr=lr, rho=0.0, mt=10, util_type="adaptable_contribution", wd=0.0,
             sigma=0.0, torch_seed=20_000 + s)
    subprocess.run([ORACLE_PY, RUNNER, "cbp_run", ip, out], check=True, capture_output=True)
    os.remove(ip)


def init_matched(arm, s):
    out = os.path.join(OUT, f"ourinit_{arm}_seed{s}.npz")
    if os.path.exists(out):
        return
    act, lr = split(arm)
    stream = dict(np.load(export("flip_one", s, STEPS)))
    p = init_mlp(jax.random.fold_in(jax.random.PRNGKey(s), 2), 20, (5,), 1, INIT_GAIN[act])   # = our run's init
    inp = {"kind": "ffnn", "n_in": 20, "width": 5, "act": act, "opt": "sgd", "lr": lr, "wd": 0.0,
           "X": stream["X"][:STEPS], "Y": stream["Y"][:STEPS]}
    for i in range(2):
        inp[f"W{i}"] = np.asarray(p["params"][f"layer_{i}"]["kernel"]).T.copy()
        inp[f"b{i}"] = np.asarray(p["params"][f"layer_{i}"]["bias"])
    ip, op = out.replace(".npz", "_in.npz"), out.replace(".npz", "_raw.npz")
    np.savez(ip, **inp)
    subprocess.run([ORACLE_PY, RUNNER, "bp_traj", ip, op], check=True, capture_output=True)
    errs = np.load(op)["errs"]
    np.savez(out, errs=errs)
    os.remove(ip)
    os.remove(op)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--matched_seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for s in range(a.seeds):
        export("flip_one", s, STEPS)
    t0 = time.time()
    jobs = [(torch_init, arm, s) for arm in ARMS for s in range(a.seeds)]
    jobs += [(init_matched, arm, s) for arm in ARMS for s in range(a.matched_seeds)]
    with ThreadPoolExecutor(a.procs) as ex:
        list(ex.map(lambda j: j[0](j[1], j[2]), jobs))
    print(f"[INFO] oracle jobs done in {time.time() - t0:.0f}s", flush=True)
    ours = load("full_scr_b10")
    res = {}
    for arm in ARMS:
        o = np.asarray([np.load(os.path.join(OUT, f"torchinit_{arm}_seed{s}.npz"))["mse_per_10k"][:4].mean()
                        for s in range(a.seeds)])
        u = np.asarray([np.mean(ours[s]["arms"][arm]["mse_per_10k"][:4]) for s in range(a.seeds)])
        u100 = np.asarray([np.mean(ours[s]["arms"][arm]["mse_per_10k"][:4]) for s in sorted(ours)])
        # init-matched: our per-10k values from the run vs the official learner from our init
        mm = []
        for s in range(a.matched_seeds):
            e = np.load(os.path.join(OUT, f"ourinit_{arm}_seed{s}.npz"))["errs"]
            mm.append({"seed": s, "official_from_our_init_first_bin": float(e.mean()),
                       "ours_first_bin": float(np.mean(ours[s]["arms"][arm]["mse_per_10k"][:4])),
                       "per_10k_max_abs_diff": float(np.abs(e.reshape(4, -1).mean(1)
                                                            - np.asarray(ours[s]["arms"][arm]["mse_per_10k"][:4])).max())})
        res[arm] = {"official_torch_init_streams_0_29": mean_se(o), "ours_streams_0_29": mean_se(u),
                    "ours_streams_0_99": mean_se(u100), "paired_by_stream_ours_minus_official": paired_compare(u, o),
                    "init_matched": mm}
        pc = res[arm]["paired_by_stream_ours_minus_official"]
        print(f"[METRIC] first-bin {arm:14s} official(torch init) {o.mean():.3f}±{o.std(ddof=1) / np.sqrt(len(o)):.3f} "
              f"ours {u.mean():.3f}±{u.std(ddof=1) / np.sqrt(len(u)):.3f} diff {pc['diff_mean']:+.3f} p={pc['t_p']:.2g}; "
              f"init-matched max|Δ per 10k| {max(m['per_10k_max_abs_diff'] for m in mm):.2e}", flush=True)
    with open(os.path.join(REPO, "runs", "oracle_ref", "first_bin_probe.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
