"""Collect the per-seed headline metrics of every experiment/arm of a gate tag into runs/seed_<n>/metrics.json
(the layout the loop requires); the raw per-run files stay under runs/<tag>/<exp>/<arm>/seed_<n>/metrics.json.
Usage: JAX_PLATFORMS=cpu uv run python scripts/collect_seed_metrics.py toys small
"""
import glob
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402

from src import eval as E  # noqa: E402
from rl_jax_bench.jaxlog import write_json  # noqa: E402


def headline(d: dict) -> dict:
    exp = d.get("exp", "")
    if exp.startswith("e3_"):
        sem = d["config"].get("qoc_semantics", "literal")
        return dict(metric="exact value from (slot 0, t 0) of the final policy",
                    value=E.qoc_exact_value(np.asarray(d["per_update"]["p_danger"])[-1][: int(d["lane"]["T"])],
                                                 float(d["lane"]["gamma"]), sem))
    if exp.startswith("e5_"):  #: E5 dynamics runs (1e6 steps); headline = the E4 metric at 1e6
        return dict(metric="mean training return, last 10% of steps run (1e6-step E5 dynamics run)",
                    value=E.final_window_return(d))
    if exp.startswith("e4_"):
        return dict(metric="mean training return, last 10% of steps",
                    value=E.final_window_return(d))  # steps run
    if exp.startswith("e7_"):
        return dict(metric="discounted eval return (gamma 0.99, H 3000, 5 eps) at the final evaluation",
                    value=float(np.mean(d["evals"][-1]["disc"])))
    if exp.startswith("e9_"):
        r = np.asarray(d["bin_return_mean"])
        k = max(1, int(round(len(r) * E.FINAL_FRAC)))
        return dict(metric="online eps-greedy return, last 10% of bins", value=float(r[-k:].mean()))
    if exp.startswith("e2_"):
        return dict(metric="exact values of the learned greedy policy (T=3, inf)", value_T3=d["value_T3"],
                    value_inf=d["value_inf"])
    return {}


def main(tags):
    per_seed = {}
    for tag in tags:
        for p in glob.glob(os.path.join(REPO, "runs", tag, "**", "seed_*", "metrics.json"), recursive=True):
            d = json.load(open(p))
            rel = os.path.relpath(os.path.dirname(os.path.dirname(p)), os.path.join(REPO, "runs"))
            per_seed.setdefault(int(d["seed"]), {})[rel] = dict(headline(d), source=os.path.relpath(p, REPO))
    for s, v in per_seed.items():
        write_json(os.path.join(REPO, "runs", f"seed_{s}", "metrics.json"), dict(seed=s, runs=v))
    print("wrote", len(per_seed), "seed files")


if __name__ == "__main__":
    main(sys.argv[1:])
