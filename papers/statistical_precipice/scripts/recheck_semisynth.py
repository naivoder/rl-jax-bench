"""recheck: is M3's NOT SUPPORTED ('percentile CIs cover best from N ~ 10') an artefact of the
ceiling-bound M = 2 E6 pools? Fig 6 procedure on a SEMI-SYNTHETIC M = 26 pool (pre-declared in the README:

  tasks      the 26 Atari-100k games (sources.yaml atari100k_*), all in the 55-game ALE set
  location / scale per task: mean and SD (ddof 1, floor 1e-3) of C51's 5 final human-normalised Dopamine runs
             (unbounded, right-skewed ACROSS tasks, like the paper's scores)
  pool       R = 200 runs per task drawn ONCE (fixed seed) as mu_m + sd_m * e, e ~ N(0, 1) ('gauss') or a standardised
             log-normal (sigma 0.5; right-skewed within task, 'skew')
  procedure  K in {3, 5, 10, 20} runs per task WITHOUT replacement, 2,000 sets x 1,000 inner stratified-bootstrap reps,
             IQM (official mode a) and median of task means, percentile / basic / bc / bca; truth = statistic of the pool
  rule       M3 SUPPORTED on this pool iff at K in {10, 20} percentile is the closest method to the FPC target
             P(|Z| < z sqrt(R/(R-K))), or within 1 pp of the closest, in >= 6 of the 8 (stat x K x noise) cells

    python scripts/recheck_semisynth.py [--sets 2000 --inner 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src import eval as E  # noqa: E402

R, KS, METHODS = 200, (3, 5, 10, 20), ("percentile", "basic", "bc", "bca")
SA_STAT = 21


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=2000)
    ap.add_argument("--inner", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "full", "recheck_semisynth.json"))
    a = ap.parse_args()
    src = D.sources()
    g100k = None
    for sec in src.values():
        if isinstance(sec, dict) and "atari100k_human" in sec:
            g100k = list(sec["atari100k_human"]["value"])
    scores, games, _ = D.load_dopamine_ale(final_only=True, normalised=True, fetch=False)
    low = {g.lower(): i for i, g in enumerate(games)}
    idx = [low[g.lower()] for g in g100k]
    c51 = np.asarray(scores["C51"])[:, idx]                       # (5, 26)
    mu, sd = c51.mean(0), np.maximum(c51.std(0, ddof=1), 1e-3)
    rng = np.random.default_rng(20261016)
    z = rng.standard_normal((R, len(idx)))
    s_ln = np.sqrt((np.exp(0.25) - 1) * np.exp(0.25))
    pools = {"gauss": mu + sd * z, "skew": mu + sd * (np.exp(0.5 * z) - np.exp(0.125)) / s_ln}
    stats_ = {"iqm": A.agg_iqm, "median": A.agg_median}
    key = A.branch_key(0, A.EXP_IDS["e6"], 0, SA_STAT)
    out = dict(games=g100k, mu=mu.tolist(), sd=sd.tolist(), R=R, sets=a.sets, inner=a.inner, device=str(jax.devices()[0]),
               cells={})
    t0 = time.time()
    for ni, (nz, pool) in enumerate(pools.items()):
        for si, (sn, f) in enumerate(stats_.items()):
            for k in KS:
                kk = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(key, ni), si), k)
                res, truth, _ = A.coverage(kk, jnp.asarray(pool), f, k, a.sets, a.inner, methods=METHODS)
                tgt = E.fpc_target_coverage(k, R)
                cov = {m: res[m]["coverage"] for m in METHODS}
                closest = min(METHODS, key=lambda m: abs(cov[m] - tgt))
                gap = 100 * (abs(cov["percentile"] - tgt) - abs(cov[closest] - tgt))
                out["cells"][f"{nz}/{sn}/K{k}"] = dict(truth=truth, coverage=cov, se={m: res[m]["se"] for m in METHODS},
                                                       width={m: res[m]["mean_width"] for m in METHODS},
                                                       undefined={m: res[m]["undefined"] for m in METHODS},
                                                       fpc_target=tgt, d10_fpc=E.predicted_mean_coverage(k, 26, R=R),
                                                       closest=closest, pct_gap_pp=gap)
                print(f"[INFO] {nz:5s} {sn:6s} K={k:2d} " + " ".join(f"{m} {cov[m]:.3f}" for m in METHODS)
                      + f" | FPC target {tgt:.3f} closest {closest} pct gap {gap:.2f} pp  ({time.time() - t0:.0f}s)", flush=True)
    rule = [c for n, c in out["cells"].items() if n.endswith("K10") or n.endswith("K20")]
    ok = sum(c["closest"] == "percentile" or c["pct_gap_pp"] <= 1.0 for c in rule)
    out["rule"] = dict(cells=len(rule), percentile_ok=ok, M3_supported=bool(ok >= 6))
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out, "w"), indent=1)
    print("[INFO] rule:", out["rule"], "wrote", a.out)


if __name__ == "__main__":
    main()
