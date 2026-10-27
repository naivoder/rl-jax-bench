"""Oracle-exponent control (CPU, numpy, 0 GPU-h).  Writes runs/full/alg2_control.json.

PRE-DECLARED BEFORE THE FIRST RUN (this docstring is the declaration).
Why: the Algorithm-2 'MATCH' of the T^-1/3 rate (scripts/alg2.py) is near-tautological: the oracle's own schedule is
eps_t = min(1, (A/t)^(1/3)), and per layer the mean_t eps_t (A-1)/A ~ (3/2)(A/R)^(1/3)(A-1)/A.  So the fitted T-exponent is the exponent of the
chosen schedule.  Control: the SAME code (src/lock.py alg2_eps_greedy) with the schedule exponent changed to 1/2, eps_t = min(1, (A/t)^(1/2)).
Derived expectation (a derivation, not a fit): mean_t eps_t = 2 A^(1/2) R^(-1/2) + O(1/R), so S ~ H * R^(-1/2) and the slope of ln S in ln T is -1/2 (the
exponent of the oracle), while greedy mistakes are lower order.  Grade: the control PASSES (the fit tracks the oracle, so the earlier -1/3 is the
oracle's exponent and the test is an implementation / linear-propagation check) iff |slope + 1/2| <= max(2 SE, 5 % of 1/2) AND the slope differs
from the exponent-1/3 slope (-0.3328, runs/full/alg2.json) by more than 0.1.  H = 6, A = 2, T in {2.5e4, 5e4, 1e5, 2e5, 4e5}, N = 2000, same cells
and seeds layout as alg2.py (seeds 9300 + i).  Used with eps_exp = 0.5 (nothing else differs from the declared oracle).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import lock as L

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "full")
N = 2000


def wls(x, y, se):
    w = 1.0 / np.maximum(se, 1e-12) ** 2
    X = np.stack([np.ones_like(x), x], 1)
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ (X.T @ (w * y))
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


if __name__ == "__main__":
    t0 = time.time(); cells = []
    for i, T in enumerate((25000, 50000, 100000, 200000, 400000)):
        rng = np.random.default_rng(9300 + i)
        sub, S, _ = L.alg2_eps_greedy(6, 2, T, N, rng, eps_exp=0.5)
        cells.append(dict(T=T, S=float(S.mean()), S_se=float(S.std(ddof=1) / np.sqrt(N)), subopt=float(sub.mean())))
        print(cells[-1], round(time.time() - t0), "s", flush=True)
    x = np.log([c["T"] for c in cells]); y = np.log([c["S"] for c in cells]); se = np.array([c["S_se"] / c["S"] for c in cells])
    b, sb = wls(x, y, se)
    ref = json.load(open(os.path.join(OUT, "alg2.json")))["T1p_slope_lnS_vs_lnT"]["slope"]
    ok = abs(b + 0.5) <= max(2 * sb, 0.025) and abs(b - ref) > 0.1
    res = dict(cells=cells, slope=b, se=sb, derived=-0.5, ref_slope_eps_third=ref, tracks_oracle_exponent=bool(ok), secs=time.time() - t0)
    json.dump(res, open(os.path.join(OUT, "alg2_control.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=1))
