"""official-code check: grade reference run R-REF (pre-declared; see the README 'Checked against') -- CleanRL dqn_jax.py
(unmodified, own venv, stdout logs) vs our core (runs/official_ref/cartpole_ours.json), metric `final` per seed.
Writes runs/official_ref/compare.json. Pure numpy/scipy (no jax)."""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats as st

from rl_jax_bench import stats as S  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(ROOT, "external", "official"))
LOGS = os.path.join(OFFICIAL_ROOT, "ref_runs", "cleanrl_dqn_jax_cartpole_s{}.log")   # written by scripts/official/run_cleanrl_ref.sh
TOTAL, LS = 500_000, 10_000
PAT = re.compile(r"global_step=(\d+), episodic_return=\[([0-9.]+)\]")


def summarize(steps, rets):
    steps, rets = np.asarray(steps), np.asarray(rets)
    return dict(final=float(rets[steps >= 0.9 * TOTAL].mean()), auc=float(rets[steps >= LS].mean()), n_episodes=int(len(rets)))


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else np.inf
    hw = float(st.t.ppf(0.975, df) * se) if se > 0 else 0.0
    p = float(2 * st.t.sf(abs(d) / se, df)) if se > 0 else 1.0
    rng = np.random.default_rng(0)
    boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(10_000)]
    return dict(diff=float(d), ci=[float(d - hw), float(d + hw)], half_width=hw, df=float(df), p=p,
                boot_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])


def load_ours(tag):
    path = os.path.join(ROOT, "runs", "official_ref", f"cartpole_ours{tag}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def grade(clean, ours, seeds, label, declared):
    out = {}
    for metric in ("final", "auc"):
        c = [clean[s][metric] for s in seeds]
        out[f"cleanrl_{metric}"] = dict(zip(("mean", "se"), S.mean_se(c)))
        for variant in ("max", "double"):
            o = [ours[variant][str(s)][metric] for s in seeds]
            out[f"ours_{variant}_{metric}"] = dict(zip(("mean", "se"), S.mean_se(o)))
            w = welch(o, c)
            if metric == "final":
                ok = abs(w["diff"]) <= 25 or (w["ci"][0] <= 0 <= w["ci"][1] and w["half_width"] <= 50)
                w["verdict"] = ("PARITY" if ok else "DIVERGENT") + ("" if (declared and variant == "max") else " (descriptive only)")
            out[f"ours_{variant}_minus_cleanrl_{metric}"] = w
            print(f"[METRIC] {label} {metric} ours[{variant}]={np.mean(o):.2f}+-{np.std(o, ddof=1) / np.sqrt(len(o)):.2f} "
                  f"cleanrl={np.mean(c):.2f}+-{np.std(c, ddof=1) / np.sqrt(len(c)):.2f} diff={w['diff']:+.2f} "
                  f"CI=[{w['ci'][0]:+.2f},{w['ci'][1]:+.2f}] p={w['p']:.3f} {w.get('verdict', '')}")
    for v in ("max", "double"):
        out[f"per_seed_final_ours_{v}"] = {s: ours[v][str(s)]["final"] for s in seeds}
    return out


def main():
    all_seeds = list(range(1, 11))
    clean = {}
    for s in all_seeds:
        if os.path.exists(LOGS.format(s)):
            m = PAT.findall(open(LOGS.format(s)).read())
            clean[s] = summarize([int(x) for x, _ in m], [float(y) for _, y in m])
    out = dict(cleanrl=clean, rule="PARITY if |diff| <= 25 or (Welch 95% CI contains 0 and half-width <= 50)")
    s5 = [1, 2, 3, 4, 5]
    # declared test (seeds 1-5, max target): first as shipped (glorot), then the pre-declared our-code-first rerun (lecun)
    out["declared_glorot"] = grade(clean, load_ours(""), s5, "declared[glorot]", True)
    lec = load_ours("_lecun")
    if lec:
        out["declared_lecun_rerun"] = grade(clean, lec, s5, "rerun[lecun]", True)
    # descriptive: 10 seeds per side
    for init in ("glorot", "lecun"):
        a, b = load_ours("" if init == "glorot" else "_lecun"), load_ours(f"_{init}_s6-10")
        if a and b and all(s in clean for s in all_seeds):
            merged = {v: {**a[v], **b[v]} for v in ("max", "double")}
            out[f"descriptive10_{init}"] = grade(clean, merged, all_seeds, f"desc10[{init}]", False)
    print("[INFO] per-seed final cleanrl", {s: round(clean[s]["final"], 1) for s in clean})
    with open(os.path.join(ROOT, "runs", "official_ref", "compare.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
