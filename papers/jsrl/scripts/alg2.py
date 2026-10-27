"""Algorithm 2 (paper Appendix A.5.3) implemented LITERALLY on the combination lock, and the rates of Theorem A.3 / Corollary A.4 measured on it
(the earlier theory toy used an explore-then-commit argmax oracle and a P(optimal) metric, i.e. it was NOT Algorithm 2 with the
Corollary-A.4 oracle).  CPU, numpy, 0 GPU-h.  Writes runs/full/alg2.json.

PRE-DECLARED BEFORE THE FIRST RUN (this docstring is the declaration; the script was run only after it was written and committed with it):
  Oracle: eps-greedy contextual bandit, eps_t = min(1, (A/t)^(1/3)) (Langford & Zhang's epoch-greedy schedule; Corollary A.4's O(T^-1/3) oracle),
          ceil(T/H) rounds per iteration h = H-1..0, pi_h = Unif({pi_t}) (line 5); guide reaches s*_h w.p. p^h (p = 1: C = 1); lock (H, A) with A = 2.
  Metrics: subopt = V* - V^pi = 1 - prod_l q_l (V* = 1, EXACT given the oracle's state); S = sum_l (1 - q_l) = the Theorem-A.3 sum of per-layer
           regrets (an upper bound on subopt).  N = 2000 independent learners per cell; SE of ln S from the across-learner SD.
  Derivation of what to expect (written before running; it is a derivation, not a fit): per layer the mean over rounds of eps_t (A-1)/A
           = 1.5 A^(1/3) (A-1)/A R^(-1/3) with R = T/H, greedy mistakes add a faster-decaying term, so S ~ H * R^(-1/3) = H^(4/3) T^(-1/3).
  T1' slope of ln S in ln T at H = 6, T in {2.5e4, 5e4, 1e5, 2e5, 4e5}.  Paper: -1/3 (Corollary A.4, an UPPER bound on the rate).
      CONSISTENT with the corollary iff slope <= -1/3 + 2 SE (the error decays at least as fast as the bound); MATCH iff within 2 SE or 5 % of -1/3
      from either side; CLOSE within 20 %; otherwise DIVERGENT-from-the-derivation (not a claim about the paper's upper bound).
  T2' exponent of ln S in ln H at T = 2e5, H in {2, 3, 4, 6, 8, 12}.  Paper: the bound grows like H^(7/3) (reward range H - h included; the lock has
      reward range 1, so a smaller exponent is allowed).  CONSISTENT iff exponent <= 7/3 + 2 SE; the derived value is 4/3: MATCH-to-derivation iff
      within 2 SE or 5 %, CLOSE within 20 %.
  Secondary (same cells): the fits of ln(subopt) restricted to the cells with subopt <= 0.2 (the linear regime), reported, not judged.
  NOT tested (stated): FALCON+ (T^-1/2, Corollary A.4 second part), general function approximation (Corollary A.5), the S^(1/3) A^(1/3) dependence.
Uses src/lock.py `alg2_eps_greedy`, tested against an independent python implementation (tests/test_lock.py::test_alg2_*).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import lock as L

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "full")
os.makedirs(OUT, exist_ok=True)
N = 2000


def wls(x, y, se):
    w = 1.0 / np.maximum(se, 1e-12) ** 2
    X = np.stack([np.ones_like(x), x], 1)
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ (X.T @ (w * y))
    return float(beta[1]), float(np.sqrt(cov[1, 1])), float(beta[0])


def cell(H, A, T, seed):
    rng = np.random.default_rng(seed)
    sub, S, _ = L.alg2_eps_greedy(H, A, T, N, rng)
    return dict(H=H, A=A, T=T, subopt=float(sub.mean()), subopt_se=float(sub.std(ddof=1) / np.sqrt(N)), S=float(S.mean()), S_se=float(S.std(ddof=1) / np.sqrt(N)))


def judge(slope, se, derived, bound, bound_is_upper=True):
    """bound_is_upper: the paper's exponent bounds the quantity from above (slope in T must be <= bound; exponent in H must be <= bound)."""
    rel = abs(slope - derived) / abs(derived)
    cons = (slope <= bound + 2 * se) if bound == -1 / 3 else (slope <= bound + 2 * se)
    match = rel <= 0.05 or abs(slope - derived) <= 2 * se
    return dict(consistent_with_paper_bound=bool(cons), vs_derivation='MATCH' if match else 'CLOSE' if rel <= 0.20 else 'DIVERGENT', rel_err_vs_derived=float(rel))


if __name__ == "__main__":
    res = {"cells_T": [], "cells_H": []}
    t0 = time.time()
    for i, T in enumerate((25000, 50000, 100000, 200000, 400000)):
        res["cells_T"].append(cell(6, 2, T, 9100 + i)); print("T", T, res["cells_T"][-1], round(time.time() - t0), "s", flush=True)
    for i, H in enumerate((2, 3, 4, 6, 8, 12)):
        res["cells_H"].append(cell(H, 2, 200000, 9200 + i)); print("H", H, res["cells_H"][-1], round(time.time() - t0), "s", flush=True)
    c = res["cells_T"]
    x = np.log([r["T"] for r in c]); y = np.log([r["S"] for r in c]); se = np.array([r["S_se"] / r["S"] for r in c])
    b, sb, _ = wls(x, y, se)
    res["T1p_slope_lnS_vs_lnT"] = dict(slope=b, se=sb, paper_bound=-1 / 3, derived=-1 / 3, **judge(b, sb, -1 / 3, -1 / 3))
    lin = [r for r in c if r["subopt"] <= 0.2]
    if len(lin) >= 3:
        b2, sb2, _ = wls(np.log([r["T"] for r in lin]), np.log([r["subopt"] for r in lin]), np.array([r["subopt_se"] / r["subopt"] for r in lin]))
        res["T1p_secondary_ln_subopt_linear_regime"] = dict(slope=b2, se=sb2, n_cells=len(lin))
    c = res["cells_H"]
    x = np.log([r["H"] for r in c]); y = np.log([r["S"] for r in c]); se = np.array([r["S_se"] / r["S"] for r in c])
    b, sb, _ = wls(x, y, se)
    res["T2p_exponent_lnS_vs_lnH"] = dict(exponent=b, se=sb, paper_bound=7 / 3, derived=4 / 3, **judge(b, sb, 4 / 3, 7 / 3))
    lin = [r for r in c if r["subopt"] <= 0.2]
    if len(lin) >= 3:
        b2, sb2, _ = wls(np.log([r["H"] for r in lin]), np.log([r["subopt"] for r in lin]), np.array([r["subopt_se"] / r["subopt"] for r in lin]))
        res["T2p_secondary_ln_subopt_linear_regime"] = dict(exponent=b2, se=sb2, n_cells=len(lin))
    res["secs"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "alg2.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("cells_T", "cells_H")}, indent=1))
