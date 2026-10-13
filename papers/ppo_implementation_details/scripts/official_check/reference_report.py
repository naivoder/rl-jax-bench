"""authors'-code check reference-run report: the AUTHORS' code at the paper's E1 scale (500k steps) on our hardware (CPU),
summarized with the same function as our runs (src/eval.summarize), next to the paper's plotted values and ours.

Inputs: $OFFICIAL_ROOT/oracle_runs/reference_500k/*.json (harness/
launch_reference.py). Output: runs/official_check/reference_report.json + a markdown table on stdout.
Run: uv run python -m scripts.official_check.reference_report
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from src.eval import summarize  # noqa: E402
from rl_jax_bench.stats import mean_se  # noqa: E402

RUNS = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(REPO, "external", "official")), "oracle_runs", "reference_500k")
OUT = os.path.join(REPO, "runs", "official_check")
# paper §4.1 of (read from the figures; read precision in the last field)
PAPER = {("CartPole-v1", "separate"): (490, 10), ("CartPole-v1", "shared"): (277, 15),
         ("Acrobot-v1", "separate"): (-85, 10), ("Acrobot-v1", "shared"): (-380, 15),
         ("MountainCar-v0", "separate"): (-200, 0), ("MountainCar-v0", "shared"): (-200, 0)}
PAPER_RED = {("CartPole-v1", "separate"): (497, 10), ("CartPole-v1", "shared"): (210, 15),
             ("Acrobot-v1", "separate"): (-83, 10), ("Acrobot-v1", "shared"): (-363, 15)}
IMPL = {"post": {"separate": "post_ppo", "shared": "post_ppo_shared"},
        "cleanrl": {"separate": "ppo", "shared": "cleanrl_ppo_shared"},
        "baselines": {"separate": "baselines_run_separate_networks", "shared": "baselines_run"}}


def at(e, step, half=5000):
    w = np.abs(e[:, 0] - step) <= half
    return float(e[w, 1].mean()) if w.any() else float("nan")


def judge(x, ref, spread):
    if not np.isfinite(x):
        return "UNREPRODUCIBLE"
    d = abs(x - ref)
    if d <= max(spread, 0.05 * abs(ref)):
        return "MATCH"
    return "CLOSE" if d <= 0.20 * abs(ref) else "DIVERGENT"


def load(prefix, env, total):
    fs = sorted(glob.glob(os.path.join(RUNS, f"{prefix}__{env}__*.json")))
    rows = []
    for f in fs:
        d = json.load(open(f))
        if d["returncode"] != 0 or not d["episodes"]:
            continue
        e = np.array([[s, r, 1, 0] for s, r in d["episodes"]], float)
        sm = summarize(e, total, env)
        rows.append(dict(seed=d["seed"], final_window=sm["final_window"], auc=sm["auc"], at100k=at(e, 100000),
                         at150k=at(e, 150000), at500k=at(e, total, 10000), wall=d["wall_seconds"],
                         sts=sm["steps_to_threshold"]))
    return rows


def agg(rows, k):
    v = [r[k] for r in rows]
    return mean_se(v) if len(v) > 1 else (v[0] if v else float("nan"), float("nan"))


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {}
    print("| Env | Arm | Implementation | n | final 10 % window | at 100k | at 150k | paper (blue) | verdict vs paper |")
    print("|---|---|---|---|---|---|---|---|---|")
    for env in ("CartPole-v1", "Acrobot-v1", "MountainCar-v0"):
        for arm in ("separate", "shared"):
            for impl, m in IMPL.items():
                rows = load(m[arm], env, 500000)
                if not rows:
                    continue
                fw, fse = agg(rows, "final_window")
                a1, a1se = agg(rows, "at100k")
                a15, a15se = agg(rows, "at150k")
                ref = PAPER[(env, arm)] if impl != "baselines" else PAPER_RED.get((env, arm), PAPER[(env, arm)])
                v = judge(fw, *ref)
                rep[f"{env}/{arm}/{impl}"] = dict(n=len(rows), final_window=fw, se=fse, at100k=a1, at100k_se=a1se,
                                                  at150k=a15, at150k_se=a15se, paper=ref, verdict=v, per_seed=rows)
                print(f"| {env} | {arm} | {impl} | {len(rows)} | {fw:.1f} ± {fse:.1f} | {a1:.1f} ± {a1se:.1f} | "
                      f"{a15:.1f} ± {a15se:.1f} | {ref[0]} ± {ref[1]}{' (red)' if impl == 'baselines' else ''} | {v} |")
    rows = load("post_ppo_continuous_action", "Pendulum-v1", 200000)
    if rows:
        fw, fse = agg(rows, "final_window")
        rep["Pendulum-v1/post_continuous_200k"] = dict(n=len(rows), final_window=fw, se=fse, per_seed=rows)
        print(f"| Pendulum-v1 (200k, proxy) | default | post ppo_continuous_action.py | {len(rows)} | {fw:.1f} ± {fse:.1f} "
              f"| | | (no paper number) | – |")
    json.dump(rep, open(os.path.join(OUT, "reference_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
