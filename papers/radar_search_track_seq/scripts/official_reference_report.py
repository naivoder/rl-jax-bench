"""Official-code check: three-way table for the reference run (paper / authors' toolchain on our env / ours at
small-scale scale) for the 'flatten' arm, plus per-iteration PPO diagnostics side by side.

Reads runs/official/sb3_ref/<setting>_seed<n>/{metrics.json, progress.csv} (scripts/official/sb3_reference_run.py) and
runs/small/flat/seed_<n>/metrics.json (our small-scale PPO-Flat). Writes runs/official/reference_report.json.
Seed = unit; mean +- SE over seeds; Welch t-test between toolchains (descriptive: not a pre-declared family).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as sps

from rl_jax_bench.stats import mean_se  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = dict(m_cov=dict(flat=727.0, flat_companion=790.63, random=948.15),
             gospa_d=dict(flat=1.25e7, random=1.25e7), gospa_miss=dict(flat=1.33e11, random=1.31e11))
KEYS = ("m_cov", "sr", "gospa_d", "gospa_miss", "mean_tracks", "ret_flip")


def sb3_runs(setting):
    out = []
    for f in sorted(glob.glob(os.path.join(REPO, "runs", "official", "sb3_ref", f"{setting}_seed*", "metrics.json"))):
        m = json.load(open(f))
        rows = list(csv.DictReader(open(os.path.join(os.path.dirname(f), "progress.csv"))))
        hist = [r for r in rows if r.get("train/entropy_loss")]
        m["hist"] = dict(entropy=[-float(r["train/entropy_loss"]) for r in hist],
                         ev=[float(r["train/explained_variance"]) for r in hist],
                         approx_kl=[float(r["train/approx_kl"]) for r in hist],
                         clip_frac=[float(r["train/clip_fraction"]) for r in hist],
                         vl=[float(r["train/value_loss"]) for r in hist])
        out.append(m)
    return out


def ours_flat():
    out = []
    for f in sorted(glob.glob(os.path.join(REPO, "runs", "small", "flat", "seed_*", "metrics.json"))):
        m = json.load(open(f))
        ev = m["eval_stochastic"]
        pe = m["per_episode"]
        out.append(dict(seed=m["seed"], eval_end={**{k: ev[k] for k in KEYS if k in ev},
                                                  "ret_flip": float(np.mean(pe["ret_flip"]))},
                        hist={k: m["train_hist"][k] for k in ("entropy", "ev", "approx_kl", "clip_frac", "vl")},
                        env_steps=m["env_steps"]))
    return out


def summarise(vals):
    mu, se = mean_se(vals)
    return dict(mean=float(mu), se=float(se), n=len(vals), per_seed=[float(v) for v in vals])


def main():
    rep = dict(paper=PAPER, settings={})
    ours = ours_flat()
    rep["ours_small_flat"] = {k: summarise([o["eval_end"][k] for o in ours]) for k in KEYS}
    rep["ours_small_flat"]["env_steps"] = [o["env_steps"] for o in ours]
    rep["ours_small_flat"]["hist"] = {k: np.mean([o["hist"][k] for o in ours], 0).tolist() for k in ours[0]["hist"]}
    for setting in ("paper", "sb3"):
        runs = sb3_runs(setting)
        if not runs:
            continue
        r = {}
        for phase in ("eval_init", "eval_after_bc", "eval_end"):
            r[phase] = {k: summarise([m[phase][k] for m in runs]) for k in KEYS + ("policy_entropy",)}
        r["delta_end_minus_bc"] = {k: summarise([m["eval_end"][k] - m["eval_after_bc"][k] for m in runs])
                                   for k in ("m_cov", "gospa_d", "ret_flip", "policy_entropy")}
        r["hist"] = {k: np.mean([m["hist"][k] for m in runs], 0).tolist() for k in runs[0]["hist"]}
        r["n_params"] = runs[0]["n_params"]
        r["ppo_env_steps"] = [m["ppo_env_steps"] for m in runs]
        r["wall_seconds"] = [m["wall_seconds"] for m in runs]
        # descriptive Welch t between the authors' toolchain and ours (end of run, seed = unit)
        r["vs_ours_welch"] = {}
        for k in ("m_cov", "gospa_d", "ret_flip"):
            a = [m["eval_end"][k] for m in runs]
            b = [o["eval_end"][k] for o in ours]
            t, p = sps.ttest_ind(a, b, equal_var=False)
            r["vs_ours_welch"][k] = dict(diff=float(np.mean(a) - np.mean(b)), t=float(t), p=float(p))
        # ratio to the paper
        r["ratio_to_paper_flat"] = {k: r["eval_end"][k]["mean"] / PAPER[k]["flat"] for k in ("m_cov", "gospa_d",
                                                                                            "gospa_miss")}
        rep["settings"][setting] = r
    rep["ours_ratio_to_paper_flat"] = {k: rep["ours_small_flat"][k]["mean"] / PAPER[k]["flat"]
                                       for k in ("m_cov", "gospa_d", "gospa_miss")}
    with open(os.path.join(REPO, "runs", "official", "reference_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    def fmt(s):
        return f"{s['mean']:.4g} ± {s['se']:.2g} (n={s['n']})"
    print("| quantity | paper Flat | SB3-fork paper-settings | SB3-fork defaults | ours small-scale |")
    for k in ("m_cov", "gospa_d", "gospa_miss", "sr", "mean_tracks", "ret_flip"):
        pv = PAPER.get(k, {}).get("flat", float("nan"))
        cells = [fmt(rep["settings"][s]["eval_end"][k]) if s in rep["settings"] else "-" for s in ("paper", "sb3")]
        print(f"| {k} | {pv:.4g} | {cells[0]} | {cells[1]} | {fmt(rep['ours_small_flat'][k])} |")
    for s in rep["settings"]:
        print(s, "entropy end", fmt(rep["settings"][s]["eval_end"]["policy_entropy"]),
              "delta(end-bc)", {k: round(v["mean"], 4) for k, v in rep["settings"][s]["delta_end_minus_bc"].items()},
              "welch", rep["settings"][s]["vs_ours_welch"])
        print(s, "hist", {k: [round(x, 4) for x in v] for k, v in rep["settings"][s]["hist"].items()})
    print("ours hist", {k: [round(x, 4) for x in v] for k, v in rep["ours_small_flat"]["hist"].items()})


if __name__ == "__main__":
    main()
