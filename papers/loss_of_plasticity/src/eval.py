"""Aggregation of per-seed metrics into reports (mean +- s.e. over seeds, paired tests, judgments).

    uv run python -m src.eval scr <tag> [<tag> ...]     # SCR reports
    uv run python -m src.eval mnist <tag>               # MNIST reports
    uv run python -m src.eval oracle <tag>              # oracle reference vs our matching arms
    uv run python -m src.eval collect                   # merge runs/<tag>/seed_<n> into runs/seed_<n>/metrics.json

The plasticity diagnostics themselves are in src/diagnostics.py (re-exported here).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402

from .diagnostics import (dead_units, effective_rank, layer_diagnostics, mean_abs_weight,  # noqa: E402
                          saturated_units, scaled_effective_rank)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs")


def load(tag):
    out = {}
    for p in sorted(glob.glob(os.path.join(RUNS, tag, "seed_*", "metrics.json"))):
        d = json.load(open(p))
        out[int(d["seed"])] = d
    return out


def judge(ours, paper, spread=0.0):
    if paper is None:
        return None
    rel = abs(ours - paper) / abs(paper)
    if abs(ours - paper) <= spread or rel <= 0.05:
        return "MATCH"
    return "CLOSE" if rel <= 0.20 else "DIVERGENT"


# ------------------------------------------------------------------------------------------- SCR
def scr_arm_table(tag, bin_periods=4):
    runs = load(tag)
    seeds = sorted(runs)
    arms = list(runs[seeds[0]]["arms"])
    tab = {}
    for a in arms:
        pp = np.asarray([runs[s]["arms"][a]["mse_per_10k"] for s in seeds])         # (S, P)
        P = pp.shape[1] // bin_periods * bin_periods
        bins = pp[:, :P].reshape(len(seeds), -1, bin_periods).mean(-1)             # (S, nbins) 40k bins
        n_first_1m = min(bins.shape[1], 1_000_000 // (10_000 * bin_periods))
        tab[a] = {
            "n_seeds": len(seeds),
            "first_bin": mean_se(bins[:, 0]),
            "last_bin": mean_se(bins[:, -1]),
            "min_bin_first_1M": mean_se(bins[:, :n_first_1m].min(1)),
            "whole_run": mean_se(pp.mean(1)),
            "last_250k": mean_se(pp[:, -25:].mean(1)),
            "bins_mean": bins.mean(0).tolist(),
            "bins_se": (bins.std(0, ddof=1) / np.sqrt(len(seeds))).tolist() if len(seeds) > 1 else None,
            "per_seed_whole_run": pp.mean(1).tolist(),
            "per_seed_last_250k": pp[:, -25:].mean(1).tolist(),
            "per_seed_min_bin_first_1M": bins[:, :n_first_1m].min(1).tolist(),
            "n_replacements_mean": float(np.mean([runs[s]["arms"][a]["n_replacements"] for s in seeds])),
        }
    return tab


def scr_report(tag):
    tab = scr_arm_table(tag)
    out = {"tag": tag, "arms": tab}
    with open(os.path.join(RUNS, tag, "report.json"), "w") as f:
        json.dump(out, f, indent=1)
    for a, v in tab.items():
        print(f"{tag:20s} {a:34s} first={v['first_bin'][0]:.3f} minfirst1M={v['min_bin_first_1M'][0]:.3f} "
              f"last={v['last_bin'][0]:.3f}+-{v['last_bin'][1] or 0:.3f} whole={v['whole_run'][0]:.3f}"
              f"+-{v['whole_run'][1] or 0:.3f} reps={v['n_replacements_mean']:.0f}")
    return out


def paired(tag, a, b, key="per_seed_whole_run"):
    tab = scr_arm_table(tag)
    return paired_compare(tab[a][key], tab[b][key])


# ------------------------------------------------------------------------------------------- MNIST
def mnist_report(tag):
    runs = load(tag)
    seeds = sorted(runs)
    arms = list(runs[seeds[0]]["arms"])
    tab = {}
    for a in arms:
        g = lambda k: np.asarray([runs[s]["arms"][a][k] for s in seeds])           # (S, tasks)
        acc = g("acc")
        k5 = min(5, acc.shape[1])
        peak = np.max(np.mean(acc, 0))
        tab[a] = {
            "n_seeds": len(seeds), "n_tasks": acc.shape[1],
            "acc_mean_per_task": acc.mean(0).tolist(),
            "acc_task1": mean_se(acc[:, 0]), "acc_last": mean_se(acc[:, -1]),
            "acc_peak_task": int(np.argmax(acc.mean(0))) + 1, "acc_peak": float(peak),
            "acc_last5": mean_se(acc[:, -k5:].mean(1)), "acc_all": mean_se(acc.mean(1)),
            "delta_peak5_minus_last5": mean_se(np.max(np.stack([acc[:, i:i + k5].mean(1) for i in
                                                                range(acc.shape[1] - k5 + 1)], 1), 1)
                                               - acc[:, -k5:].mean(1)),
            "dead_pct_first": mean_se(g("dead_pct")[:, 0]), "dead_pct_last": mean_se(g("dead_pct")[:, -1]),
            "erank_first": mean_se(g("erank_scaled")[:, 0]), "erank_last": mean_se(g("erank_scaled")[:, -1]),
            "erank_max": float(g("erank_scaled").mean(0).max()),
            "wmag_first": mean_se(g("wmag_before")[:, 0]),
            # per-step mean over the last task if tracked (track_wmag), else the mean of the before/after values
            "wmag_last_task_mean": mean_se(g("wmag_mean")[:, -1] if np.any(g("wmag_mean")) else
                                           0.5 * (g("wmag_before")[:, -1] + g("wmag_after")[:, -1])),
            "n_rep_per_task": float(g("n_rep").mean()),
            "per_seed_acc_last5": acc[:, -k5:].mean(1).tolist(),
        }
    out = {"tag": tag, "arms": tab}
    with open(os.path.join(RUNS, tag, "report.json"), "w") as f:
        json.dump(out, f, indent=1)
    for a, v in tab.items():
        print(f"{tag:18s} {a:10s} acc t1={v['acc_task1'][0]:.4f} peak={v['acc_peak']:.4f}@{v['acc_peak_task']} "
              f"last={v['acc_last'][0]:.4f} delta={v['delta_peak5_minus_last5'][0]:+.4f} "
              f"dead {v['dead_pct_first'][0]:.2f}->{v['dead_pct_last'][0]:.2f} erank {v['erank_first'][0]:.1f}->"
              f"{v['erank_last'][0]:.1f} |w| {v['wmag_first'][0]:.4f}->{v['wmag_last_task_mean'][0]:.4f} "
              f"reps/task={v['n_rep_per_task']:.1f}")
    return out


# ------------------------------------------------------------------------------------------- oracle reference
ORACLE_MATCH = {  # oracle reference case -> (our run tag suffix, arm)
    "relu_sgd_0.01_bp_flip": ("scr_b10", "relu_0.01"),
    "tanh_sgd_0.01_bp_flip": ("scr_b10", "tanh_0.01"),
    "tanh_adam_0.01_bp_flip": ("scr_d12", "bp"),
    "tanh_adam_0.01_cbp1e-4_flip": ("scr_d12", "overall_0.0001"),
    "relu_sgd_0.01_bp_resample": ("scr_methods", "bp"),
    "relu_sgd_0.01_cbp1e-4_resample": ("scr_methods", "cbp_0.0001"),
    "tanh_adam_0.01_cbp1e-4_resample": ("scr_d12_resample1m", "overall_0.0001"),
    "tanh_adam_0.01_adapt1e-4_resample": ("scr_d12_resample1m", "adaptation_0.0001"),
}


def oracle_compare(prefix, bin_periods=10):
    """Mean over seeds of 100k-step bins: ours (same seeds / streams) vs the oracle learner."""
    res = {}
    for case, (cfg, arm) in ORACLE_MATCH.items():
        files = sorted(f for f in glob.glob(os.path.join(RUNS, "oracle_ref", case, "seed_*.npz")) if "_in" not in f)
        if not files:
            continue
        seeds = [int(os.path.basename(f)[5:-4]) for f in files]
        o = np.asarray([np.load(f)["mse_per_10k"] for f in files])
        runs = load(f"{prefix}_{cfg}")
        u = np.asarray([runs[s]["arms"][arm]["mse_per_10k"][:o.shape[1]] for s in seeds if s in runs])
        if len(u) != len(o):
            continue
        nb = o.shape[1] // bin_periods
        ob = o[:, :nb * bin_periods].reshape(len(o), nb, -1).mean(-1)
        ub = u[:, :nb * bin_periods].reshape(len(u), nb, -1).mean(-1)
        se = np.sqrt(ob.var(0, ddof=1) / len(o) + ub.var(0, ddof=1) / len(u))
        z = (ub.mean(0) - ob.mean(0)) / se
        whole_o, whole_u = o.mean(1), u.mean(1)
        res[case] = {"seeds": seeds, "ours_bins": ub.mean(0).tolist(), "oracle_bins": ob.mean(0).tolist(),
                     "z_per_bin": z.tolist(), "max_abs_z": float(np.abs(z).max()),
                     "whole_ours": mean_se(whole_u), "whole_oracle": mean_se(whole_o),
                     "whole_rel_diff": float(abs(whole_u.mean() - whole_o.mean()) / whole_o.mean())}
        print(f"oracle {case:32s} ours={whole_u.mean():.4f}+-{res[case]['whole_ours'][1]:.4f} "
              f"oracle={whole_o.mean():.4f}+-{res[case]['whole_oracle'][1]:.4f} max|z|bin={np.abs(z).max():.2f}")
    with open(os.path.join(RUNS, "oracle_ref", f"compare_{prefix}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def oracle_mnist_compare(tag, arms=("bp", "cbp_1e-4")):
    """Official learners on our first tasks (same permutations/orders) vs our run ``tag`` (same seeds)."""
    res = {}
    runs = load(tag)
    for arm in arms:
        files = sorted(f for f in glob.glob(os.path.join(RUNS, "oracle_ref", f"mnist_{arm}", "seed_*.npz")) if "_in" not in f)
        if not files:
            continue
        seeds = [int(os.path.basename(f)[5:-4]) for f in files]
        o = [dict(np.load(f)) for f in files]
        T = o[0]["acc"].shape[0]
        oa = np.asarray([d["acc"] for d in o])
        od = np.asarray([100 * d["dead"].sum(1) / 300 for d in o])
        oe = np.asarray([d["erank"].mean(1) for d in o])                    # width 100: 100*erank/100
        ua = np.asarray([runs[s]["arms"][arm]["acc"][:T] for s in seeds])
        ud = np.asarray([runs[s]["arms"][arm]["dead_pct"][:T] for s in seeds])
        ue = np.asarray([runs[s]["arms"][arm]["erank_scaled"][:T] for s in seeds])
        z = lambda u, v: ((u.mean(0) - v.mean(0)) / np.sqrt(u.var(0, ddof=1) / len(u) + v.var(0, ddof=1) / len(v) + 1e-12)).tolist()
        res[arm] = {"seeds": seeds, "acc_ours": ua.mean(0).tolist(), "acc_oracle": oa.mean(0).tolist(), "z_acc": z(ua, oa),
                    "dead_ours": ud.mean(0).tolist(), "dead_oracle": od.mean(0).tolist(),
                    "erank_ours": ue.mean(0).tolist(), "erank_oracle": oe.mean(0).tolist(), "z_erank": z(ue, oe),
                    "acc_mean_diff_pp": float(100 * (ua.mean() - oa.mean()))}
        print(f"oracle mnist {arm}: acc ours {np.round(ua.mean(0), 4)} oracle {np.round(oa.mean(0), 4)}; "
              f"erank ours {np.round(ue.mean(0), 1)} oracle {np.round(oe.mean(0), 1)}; dead ours {np.round(ud.mean(0), 2)} "
              f"oracle {np.round(od.mean(0), 2)}")
    with open(os.path.join(RUNS, "oracle_ref", f"compare_mnist_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def collect():
    """Merge every runs/<tag>/seed_<n>/metrics.json into runs/seed_<n>/metrics.json (keyed by tag)."""
    merged = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "seed_*", "metrics.json"))):
        tag = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if tag.startswith("seed_"):
            continue
        d = json.load(open(p))
        merged.setdefault(int(d["seed"]), {})[tag] = d
    for s, d in merged.items():
        path = os.path.join(RUNS, f"seed_{s}", "metrics.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"seed": s, "experiments": d}, f)
    print(f"collected {len(merged)} seeds")


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "scr":
        for t in sys.argv[2:]:
            scr_report(t)
    elif kind == "mnist":
        for t in sys.argv[2:]:
            mnist_report(t)
    elif kind == "oracle":
        oracle_compare(sys.argv[2])
    elif kind == "oracle_mnist":
        oracle_mnist_compare(sys.argv[2])
    elif kind == "collect":
        collect()
