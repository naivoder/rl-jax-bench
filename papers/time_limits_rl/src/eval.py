"""Metrics and summaries (paper's metrics per experiment).

* E2: exact values of learned greedy policies (src/exact.py) -- P1 / P2 statistics.
* E3: exact DP value from (slot 0, t = 0) of the learned PPO policy P(dangerous | slot, t) -- P3; Figure-2 table.
* E4: mean undiscounted training-episode return over the last 10 % of env steps -- P4 / P5.
* E5: critic value vs elapsed step on full-length episodes vs the closed forms of 
* E7: discounted (gamma 0.99) evaluation return at the final evaluation -- P6.
* E9: online eps-greedy episode return, mean over the last 10 % of bins -- P7 / P8.
Paired comparisons use rl_jax_bench.stats (paired t, bootstrap CI, Holm).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


FINAL_FRAC = 0.10  #  (last 10 % of steps)


def load_arm(root: str, arm: str) -> dict[int, dict]:
    out = {}
    for p in sorted(glob.glob(os.path.join(root, arm, "seed_*", "metrics.json"))):
        with open(p) as f:
            d = json.load(f)
        out[int(d["seed"])] = d
    return out


def steps_run(rec: dict) -> int:
    """Env steps actually run by a PPO lane: U * num_steps (U = total_timesteps // num_steps, as Baselines ppo2's
    nupdates), i.e. 98 304 for a nominal 1e5. Read from the per-update step record written by src/train.py."""
    return int(np.asarray(rec["per_update_global_step"])[-1])


def final_window_return(rec: dict, total: int | None = None, frac: float = FINAL_FRAC) -> float:
    """Mean undiscounted return of training episodes ending in the last `frac` of the env steps ACTUALLY RUN.

    total=None (default for our runs) takes the steps actually run (steps_run). Review fix: full scale passed
    the nominal config total (1e5) while PPO ran 48 x 2048 = 98 304 steps, so its 'last 10 %' window was the last
    8 304 steps (8.4 %); pass total explicitly only for records without per_update_global_step (Tonic runs, whose
    trainer runs exactly `steps`)."""
    ep = np.asarray(rec["episodes"], np.float64)
    if ep.size == 0:
        return float("nan")
    if total is None:
        total = steps_run(rec)
    m = ep[:, 0] > (1.0 - frac) * total
    return float(ep[m, 1].mean()) if m.any() else float("nan")


def qoc_exact_value(p_danger, gamma: float = 0.99, semantics: str = "literal") -> float:
    """Exact value from (slot 0, t = 0) of a policy table P(dangerous | t, slot) (T, 9)."""
    import jax.numpy as jnp
    from . import data as D
    from . import exact as X
    p = D.QOC_CRASH_FIRST if semantics == "crash_first" else D.QoCParams()
    return float(X.qoc_policy_value(jnp.asarray(p_danger, jnp.float32), D.qoc_outcomes(p), gamma)[0])


def paired(a, b) -> dict:
    from rl_jax_bench.stats import paired_compare
    return paired_compare(a, b)


def mean_se(x):
    x = np.asarray(x, np.float64)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
