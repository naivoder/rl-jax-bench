"""Reference run: the AUTHORS' filter (Octave dumps, drive_pmbm.m) vs OURS on the SAME measurement sets, scored with the same
metric code (src/eval.py: OSPA p=2 c=10 = the paper's metric, GOSPA alpha=2 = the official code's metric).

    python reference_run.py in.mat dump_dir truth.npz out.json [--runs 0 1 2 ...] [--truth-layout smoke|data]

truth.npz: either ref_meas.npz-style key 'truth' [4 target, 4 state, K] (scripts/official/gen_smoke_input.py) or data.py-style X[4,K,4]+alive.
Writes per-run per-step squared OSPA / GOSPA for official and ours (estimators 1-3), the largest estimate-set difference per
run, and the aggregate RMS (Table II statistic) + paired per-run statistics.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402

import diff_run  # noqa: E402
import parity_lib as pl  # noqa: E402
from src import eval as ev  # noqa: E402


def load_truth(path, K):
    t = np.load(path)
    if "truth" in t.files:                                   # [target, state, time]
        X = np.transpose(t["truth"], (0, 2, 1))              # -> [target, time, state]
        k = np.arange(1, K + 1)
        t_death = np.array([(K + 1) // 2, K + 1, K + 1, K + 1])
        alive = k[None, :] < t_death[:, None]
        return dict(X=X, alive=alive)
    return dict(X=t["X"], alive=t["alive"])


def pad(sets, K, E=16):
    m = np.zeros((K, E), bool)
    x = np.zeros((K, E, 4))
    for k, s in enumerate(sets):
        n = len(s)
        m[k, :n] = True
        x[k, :n] = s
    return m, x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("in_mat"), ap.add_argument("dump_dir"), ap.add_argument("truth"), ap.add_argument("out")
    ap.add_argument("--runs", type=int, nargs="*")
    ap.add_argument("--T", type=int, default=192), ap.add_argument("--H", type=int, default=200), ap.add_argument("--M", type=int, default=40)
    a = ap.parse_args(argv)
    P = loadmat(a.in_mat)
    K = np.asarray(P["z"]).shape[1]
    truth = load_truth(a.truth, K)
    runs = a.runs if a.runs else sorted(int(f[4:7]) for f in os.listdir(a.dump_dir) if f.startswith("run_"))
    res = {"runs": runs, "K": K, "per_run": {}}
    for r in runs:
        d = pl.load_run(a.dump_dir, r)
        rec = diff_run.run_ours(P, r, K, cfg_over=dict(T=a.T, H=a.H), M=a.M, record=False)
        out = {}
        maxdiff = 0.0
        ovf = np.max([x["ovf"] for x in rec], axis=0).tolist()
        for e in (1, 2, 3):
            off = [pl.official_estimates(d, e, k) for k in range(K)]
            ours = [rec[k]["est"][e] for k in range(K)]
            for k in range(K):
                maxdiff = max(maxdiff, pl.compare_est(off[k], ours[k]))
            mo, xo = pad(off, K)
            mu, xu = pad(ours, K)
            eo = ev.evaluate_run(mo, xo, truth)
            eu = ev.evaluate_run(mu, xu, truth)
            out[f"est{e}"] = dict(official=dict(ospa2=eo["ospa2"].tolist(), gospa2=eo["gospa2"].tolist()),
                                  ours=dict(ospa2=eu["ospa2"].tolist(), gospa2=eu["gospa2"].tolist()))
        out["max_estimate_diff"] = maxdiff
        out["our_overflow_counters_max"] = ovf
        res["per_run"][str(r)] = out
        print(f"run {r}: max |estimate difference| {maxdiff:.3e}; overflow counters {ovf}", flush=True)
    json.dump(res, open(a.out, "w"))
    # aggregate
    print("estimator  official RMS OSPA   ours RMS OSPA   official RMS GOSPA  ours RMS GOSPA   max per-step |d ospa2|")
    for e in (1, 2, 3):
        O = np.array([res["per_run"][str(r)][f"est{e}"]["official"]["ospa2"] for r in runs])
        U = np.array([res["per_run"][str(r)][f"est{e}"]["ours"]["ospa2"] for r in runs])
        GO = np.array([res["per_run"][str(r)][f"est{e}"]["official"]["gospa2"] for r in runs])
        GU = np.array([res["per_run"][str(r)][f"est{e}"]["ours"]["gospa2"] for r in runs])
        print(f"est{e}      {ev.joint_rms(O):.4f}             {ev.joint_rms(U):.4f}          {ev.joint_rms(GO):.4f}           {ev.joint_rms(GU):.4f}      {np.abs(O - U).max():.2e}")


if __name__ == "__main__":
    main()
