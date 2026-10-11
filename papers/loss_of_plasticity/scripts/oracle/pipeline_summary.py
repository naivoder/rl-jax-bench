"""Summarise the official end-to-end pipeline reference runs (scripts/oracle/official_pipeline.py) and set them next
to our small-scale runs.  Streams / task sequences differ between the two sides (their generator vs ours), so the
comparison is distributional: Welch t on the per-seed values (official seeds vs our seeds).

    uv run python -m scripts.oracle.pipeline_summary   ->  runs/oracle_pipeline/summary.json
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import subprocess

import numpy as np
from scipy import stats

from scripts.oracle.official_pipeline import ENV, PY, WORK

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "runs", "oracle_pipeline")
# official case -> (our small-scale tag, arm)
OURS = {"flip_relu_bp_0.01": ("small_scr_b10", "relu_0.01"), "flip_tanh_bp_0.01": ("small_scr_b10", "tanh_0.01"),
        "rs_relu_bp_0.01": ("small_scr_methods", "bp"), "rs_relu_l2_1e-2": ("small_scr_methods", "l2_0.01"),
        "rs_relu_snp_1e-3": ("small_scr_methods", "snp_0.001"),
        "rs_relu_cbp_1e-4_default": ("small_scr_methods", "cbp_0.0001"),
        "rs_relu_cbp_1e-4_mt100": (None, None),
        "mnist_w100_bp_0.003": ("small_mnist", "bp"), "mnist_w100_cbp_1e-4": ("small_mnist", "cbp_1e-4")}


def ms(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None]


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def scr_metrics(errs_list):
    """Same definitions as src/eval.py scr_arm_table: 10k periods, 40k bins."""
    pp = np.stack([e.reshape(-1, 10_000).mean(1) for e in errs_list])        # (S, P)
    P = pp.shape[1] // 4 * 4
    bins = pp[:, :P].reshape(len(pp), -1, 4).mean(-1)
    n1m = min(bins.shape[1], 25)
    per = {"whole_run": pp.mean(1), "first_bin": bins[:, 0], "last_bin": bins[:, -1],
           "min_bin_first_1M": bins[:, :n1m].min(1), "last_250k": pp[:, -25:].mean(1)}
    per["delta_lop"] = per["last_250k"] - per["min_bin_first_1M"]
    return per, bins.mean(0)


def our_scr(tag, arm):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(REPO, "runs", tag, "seed_*", "metrics.json")))]
    pp = np.asarray([r["arms"][arm]["mse_per_10k"] for r in runs])
    P = pp.shape[1] // 4 * 4
    bins = pp[:, :P].reshape(len(pp), -1, 4).mean(-1)
    n1m = min(bins.shape[1], 25)
    per = {"whole_run": pp.mean(1), "first_bin": bins[:, 0], "last_bin": bins[:, -1],
           "min_bin_first_1M": bins[:, :n1m].min(1), "last_250k": pp[:, -25:].mean(1)}
    per["delta_lop"] = per["last_250k"] - per["min_bin_first_1M"]
    return per


def mnist_convert(pkl):
    npz = pkl.replace(".pkl", ".npz")
    if not os.path.exists(npz):
        code = ("import pickle, numpy as np, sys; d=pickle.load(open(sys.argv[1],'rb')); "
                "np.savez(sys.argv[2], **{k: v.numpy() for k, v in d.items()})")
        subprocess.run([PY, "-c", code, pkl, npz], check=True, env=ENV)
    return dict(np.load(npz))


def mnist_metrics(ds, width=100, n_tasks=15):
    n_w = 784 * width + 2 * width * width + width * 10
    acc = np.stack([d["accuracies"].reshape(n_tasks, -1).mean(1) for d in ds])                 # (S, T)
    dead = np.stack([d["dead_neurons"].sum(1) / (3 * width) * 100 for d in ds])
    er = np.stack([d["effective_ranks"].mean(1) / width * 100 for d in ds])
    wm = np.stack([(d["weight_mag_sum"].sum(1) / n_w).reshape(n_tasks, -1).mean(1) for d in ds])
    k5 = 5
    peak5 = np.max(np.stack([acc[:, i:i + k5].mean(1) for i in range(n_tasks - k5 + 1)], 1), 1)
    per = {"acc_task1": acc[:, 0], "acc_last": acc[:, -1], "acc_last5": acc[:, -k5:].mean(1),
           "delta_peak5_minus_last5": peak5 - acc[:, -k5:].mean(1), "dead_pct_first": dead[:, 0],
           "dead_pct_last": dead[:, -1], "erank_first": er[:, 0], "erank_last": er[:, -1],
           "wmag_last_task_mean": wm[:, -1]}
    return per, acc.mean(0)


def our_mnist(arm):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(REPO, "runs", "small_mnist", "seed_*",
                                                                      "metrics.json")))]
    g = lambda k: np.asarray([r["arms"][arm][k] for r in runs])
    acc = g("acc")
    k5 = 5
    peak5 = np.max(np.stack([acc[:, i:i + k5].mean(1) for i in range(acc.shape[1] - k5 + 1)], 1), 1)
    return {"acc_task1": acc[:, 0], "acc_last": acc[:, -1], "acc_last5": acc[:, -k5:].mean(1),
            "delta_peak5_minus_last5": peak5 - acc[:, -k5:].mean(1), "dead_pct_first": g("dead_pct")[:, 0],
            "dead_pct_last": g("dead_pct")[:, -1], "erank_first": g("erank_scaled")[:, 0],
            "erank_last": g("erank_scaled")[:, -1], "wmag_last_task_mean": g("wmag_mean")[:, -1]}, acc.mean(0)


def main():
    res = {}
    for case in OURS:
        files = sorted(glob.glob(f"{WORK}/out/{case}/seed_*.pkl"))
        if not files:
            continue
        if case.startswith("mnist"):
            per, curve = mnist_metrics([mnist_convert(f) for f in files])
        else:
            per, curve = scr_metrics([pickle.load(open(f, "rb"))["errs"] for f in files])
        entry = {"n_seeds_official": len(files), "official": {k: ms(v) for k, v in per.items()},
                 "official_per_seed": {k: np.asarray(v).tolist() for k, v in per.items()},
                 "official_curve": curve.tolist()}
        tag, arm = OURS[case]
        if tag is not None:
            if case.startswith("mnist"):
                ou, ocurve = our_mnist(arm)
                entry["ours_curve"] = ocurve.tolist()
            else:
                ou = our_scr(tag, arm)
            entry["ours_tag"], entry["n_seeds_ours"] = f"{tag}:{arm}", len(next(iter(ou.values())))
            entry["ours"] = {k: ms(v) for k, v in ou.items()}
            entry["welch_p"] = {k: welch(per[k], ou[k]) for k in per if k in ou}
        res[case] = entry
        o = entry["official"]
        print(f"[METRIC] official_pipeline {case} n={len(files)} " +
              " ".join(f"{k}={v[0]:.4f}" for k, v in o.items()) +
              ("" if tag is None else " | ours " + " ".join(f"{k}={v[0]:.4f}" for k, v in entry["ours"].items())),
              flush=True)
    # paired within the official runs: CBP default (m=10) vs m=100, and CBP vs BP on identical streams
    pairs = [("rs_relu_cbp_1e-4_default", "rs_relu_cbp_1e-4_mt100"), ("rs_relu_bp_0.01", "rs_relu_cbp_1e-4_default"),
             ("rs_relu_bp_0.01", "rs_relu_snp_1e-3"), ("rs_relu_bp_0.01", "rs_relu_l2_1e-2"),
             ("mnist_w100_bp_0.003", "mnist_w100_cbp_1e-4")]
    res["paired_official"] = {}
    for a, b in pairs:
        if a in res and b in res:
            key = "whole_run" if not a.startswith("mnist") else "acc_last5"
            x, y = np.asarray(res[a]["official_per_seed"][key]), np.asarray(res[b]["official_per_seed"][key])
            n = min(len(x), len(y))
            d = x[:n] - y[:n]
            res["paired_official"][f"{a} - {b} ({key})"] = {
                "n": n, "diff": ms(d), "p_paired_t": float(stats.ttest_rel(x[:n], y[:n]).pvalue) if n > 1 else None}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for k, v in res["paired_official"].items():
        print(f"[METRIC] official_pipeline paired {k} diff={v['diff'][0]:+.4f}+-{v['diff'][1] or 0:.4f} "
              f"p={v['p_paired_t']}", flush=True)


if __name__ == "__main__":
    main()
