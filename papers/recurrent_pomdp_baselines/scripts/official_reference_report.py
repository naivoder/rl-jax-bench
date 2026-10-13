"""Official-code reference-run report: Pendulum-V sac-lstm-200-oar, separate vs shared, three sources on one axis.

  paper   : the authors' own logs behind Fig. 21 (numeric-results archive; 4 seeds, 50k steps)
  official: the UNMODIFIED official code run on our CPU (scripts/official/run_official.py; gym-0.18 Pendulum-v0 dynamics = the
            paper's runs), truncated to 20k env steps; plus a variant with the repo's current default gym-0.21
            Pendulum-v1 dynamics (scripts/official/run_official_v1.py)
  ours    : the JAX re-implementation at small stage scale (runs/small, 4 seeds, 20k env steps)

Per seed: mean eval return (metrics/return_eval_total) over env steps [16k, 20k] (the small stage window), curve points,
and medians over env steps [2k, 16k] of the training statistics all three log (alpha, entropy, Q losses, policy loss,
RNN grad norms; official metric = mean per-tensor l2 norm).  Unpaired comparisons (different RNG streams): Welch t and
the exact two-sided permutation test on the per-seed window means; n is 3-4 so these are descriptive.
Writes runs/official_ref/report.json and prints the tables.
    python scripts/official_reference_report.py
"""
import csv
import glob
import itertools
import json
import os
import re
import sys

import numpy as np
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from rl_jax_bench.stats import mean_se, paired_compare  # noqa: E402

from src.eval import AUTHOR_LOGS  # noqa: E402

OFF = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))
WINDOW = (16000, 20000)
DIAG_WIN = (2000, 16000)
POINTS = [2000, 4000, 8000, 12000, 16000, 20000]
# official column -> our train-log key
DIAG = {"separate": {"rl_loss/alpha": "alpha", "rl_loss/policy_entropy": "entropy", "rl_loss/qf1_loss": "qf1_loss",
                     "rl_loss/policy_loss": "policy_loss", "rl_loss/q_rnn_grad_norm": "off_q_rnn",
                     "rl_loss/pi_rnn_grad_norm": "off_pi_rnn"},
        "shared": {"rl_loss/alpha": "alpha", "rl_loss/policy_entropy": "entropy", "rl_loss/qf1_loss": "qf1_loss",
                   "rl_loss/policy_loss": "policy_loss", "rl_loss/rnn_grad_norm": "off_rnn"}}


def read_csv_run(path, arch):
    rows = list(csv.DictReader(open(path)))
    ev = np.array([(float(r["z/env_steps"]), float(r["metrics/return_eval_total"])) for r in rows
                   if r.get("metrics/return_eval_total") not in (None, "")])
    tr = [r for r in rows if r.get("rl_loss/alpha") not in (None, "")]
    diag = {}
    for col, key in DIAG[arch].items():
        v = [(1000 + 200 * i, float(r[col])) for i, r in enumerate(tr) if r.get(col) not in (None, "")]
        diag[key] = v
    return ev, diag


def ours_run(path, arch):
    m = json.load(open(path))
    ev = np.array(list(zip(m["eval_env_steps"], m["eval_returns"])), float)
    diag = {k: [(t["n_env"], float(t[k])) for t in m["train"] if k in t] for k in DIAG[arch].values()}
    return ev, diag


def summarise(ev, diag):
    if ev.ndim != 2 or len(ev) == 0:
        return None
    sel = (ev[:, 0] >= WINDOW[0]) & (ev[:, 0] <= WINDOW[1])
    out = {"max_env_steps": float(ev[:, 0].max()),
           "window_mean": float(ev[sel, 1].mean()) if sel.any() else None,
           "window_complete": bool(ev[:, 0].max() >= WINDOW[1]),
           "points": {p: float(ev[ev[:, 0] == p, 1][0]) for p in POINTS if (ev[:, 0] == p).any()},
           "diag": {}}
    for k, v in diag.items():
        x = [b for a, b in v if DIAG_WIN[0] <= a <= DIAG_WIN[1]]
        if x:
            out["diag"][k] = float(np.median(x))
    return out


def official_runs(prefix):
    """oracle/ref_logs/<prefix>_s<n>.log -> its progress.csv (the 'Logging to' line)."""
    out = {}
    for log in sorted(glob.glob(os.path.join(OFF, "oracle", "ref_logs", f"{prefix}_s*.log"))):
        m = re.search(r"Logging to (\S+)", open(log).read())
        if not m:
            continue
        p = os.path.join(OFF, "repo", m.group(1), "progress.csv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            out[os.path.basename(log)] = p
    return out


def perm_p(a, b):
    """exact two-sided permutation test on the difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pool, na = np.concatenate([a, b]), len(a)
    obs = abs(a.mean() - b.mean())
    cnt = tot = 0
    for idx in itertools.combinations(range(len(pool)), na):
        m = np.zeros(len(pool), bool)
        m[list(idx)] = True
        cnt += abs(pool[m].mean() - pool[~m].mean()) >= obs - 1e-9
        tot += 1
    return cnt / tot


def main():
    arch_dir = {"separate": "sac_lstm", "shared": "sac_lstm_shared"}
    ours_dir = {"separate": "P_sac_separate/separate", "shared": "P_sac_shared/official"}
    rep = {}
    for arch in ("separate", "shared"):
        src = {}
        src["paper_authors_logs"] = {os.path.relpath(f, AUTHOR_LOGS): summarise(*read_csv_run(f, arch)) for f in sorted(
            glob.glob(os.path.join(AUTHOR_LOGS, f"pomdp/Pendulum/V/{arch_dir[arch]}/len--1/*/*/*/oar/*/progress.csv")))}
        tag = "sep" if arch == "separate" else "shared"
        src["official_v0_ourhw"] = {k: summarise(*read_csv_run(p, arch)) for k, p in official_runs(f"ref20k_{tag}").items()}
        src["official_v1_ourhw"] = {k: summarise(*read_csv_run(p, arch))
                                    for k, p in official_runs(f"ref20k_v1_{tag}").items()}
        src["ours_small"] = {os.path.relpath(f, REPO): summarise(*ours_run(f, arch)) for f in sorted(
            glob.glob(os.path.join(REPO, "runs", "small", ours_dir[arch], "seed_*", "metrics.json")))}
        if arch == "shared":   # Official-code precision check (pre-declared in the README): fp32 'highest', seeds 0-7
            src["ours_fp32"] = {os.path.relpath(f, REPO): summarise(*ours_run(f, arch)) for f in sorted(
                glob.glob(os.path.join(REPO, "runs", "official_ref", "P_sac_shared_fp32", "official", "seed_*",
                                       "metrics.json")))}
        src = {s: {k: v for k, v in runs.items() if v is not None} for s, runs in src.items()}
        agg = {}
        for s, runs in src.items():
            w = [r["window_mean"] for r in runs.values() if r["window_complete"]]
            agg[s] = {"n_complete": len(w), "n_runs": len(runs), "window_seeds": w,
                      "window_mean_se": mean_se(w) if w else None,
                      "partial": {k: (r["max_env_steps"], r["window_mean"]) for k, r in runs.items()
                                  if not r["window_complete"]},
                      "points_mean": {p: float(np.mean([r["points"][p] for r in runs.values() if p in r["points"]]))
                                      for p in POINTS if any(p in r["points"] for r in runs.values())},
                      "diag_median_of_seed_medians": {k: float(np.median([r["diag"][k] for r in runs.values()
                                                                           if k in r["diag"]]))
                                                      for k in DIAG[arch].values()
                                                      if any(k in r["diag"] for r in runs.values())}}
        tests = {}
        for a, b in (("official_v0_ourhw", "paper_authors_logs"), ("ours_small", "official_v0_ourhw"),
                     ("ours_small", "paper_authors_logs"), ("official_v1_ourhw", "official_v0_ourhw"),
                     ("ours_fp32", "official_v0_ourhw"), ("ours_fp32", "paper_authors_logs"),
                     ("ours_fp32", "ours_small")):
            if a not in agg or b not in agg:
                continue
            x, y = agg[a]["window_seeds"], agg[b]["window_seeds"]
            if len(x) >= 2 and len(y) >= 2:
                tests[f"{a}_vs_{b}"] = {"diff": float(np.mean(x) - np.mean(y)),
                                        "welch_p": float(st.ttest_ind(x, y, equal_var=False).pvalue),
                                        "perm_p": perm_p(x, y)}
        if arch == "shared" and agg.get("ours_fp32", {}).get("n_complete", 0) >= 4:
            # paired: fp32 seeds 0-3 vs small stage TF32 seeds 0-3 (same keys -> same init and warm-up data)
            f32 = {k.split("seed_")[1].split("/")[0]: r["window_mean"] for k, r in src["ours_fp32"].items()}
            t32 = {k.split("seed_")[1].split("/")[0]: r["window_mean"] for k, r in src["ours_small"].items()}
            ks = sorted(set(f32) & set(t32))
            tests["paired_fp32_minus_tf32_seeds"] = {"seeds": ks, **paired_compare([f32[k] for k in ks],
                                                                                    [t32[k] for k in ks])}
        rep[arch] = {"runs": src, "agg": agg, "tests": tests}
    os.makedirs(os.path.join(REPO, "runs", "official_ref"), exist_ok=True)
    json.dump(rep, open(os.path.join(REPO, "runs", "official_ref", "report.json"), "w"), indent=1)
    for arch in rep:
        print(f"\n## {arch}  (window {WINDOW}, mean ± se over complete seeds)")
        for s, a in rep[arch]["agg"].items():
            ms = a["window_mean_se"]
            print(f"{s:22s} n={a['n_complete']}/{a['n_runs']} window={ms} seeds={np.round(a['window_seeds'], 1)} "
                  f"partial={a['partial']}")
            print(f"{'':22s} curve={ {k: round(v) for k, v in a['points_mean'].items()} }")
            print(f"{'':22s} diag={ {k: round(v, 4) for k, v in a['diag_median_of_seed_medians'].items()} }")
        for k, v in rep[arch]["tests"].items():
            if "welch_p" in v:
                print(f"  {k}: diff={v['diff']:.1f} welch_p={v['welch_p']:.3f} perm_p={v['perm_p']:.3f}")
            else:
                print(f"  {k}: diff={v['diff_mean']:.1f} se={v['diff_se']:.1f} t_p={v['t_p']:.3f} "
                      f"boot95={np.round(v['boot95'], 1)} per_seed={v['diff_per_seed']}")


if __name__ == "__main__":
    main()
