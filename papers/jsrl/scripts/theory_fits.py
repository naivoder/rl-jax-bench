"""Theory fits (CPU, numpy, no GPU): the EMPIRICAL scaling of the combination-lock quantities in Theorem 4.1 / A.3.

Pre-declared (T-family):
  T1  at fixed (H, A), ln(1 - P_opt) is linear in m = T/H with slope ln((A-1)/A) in the regime H q < 0.2 (the linearisation 1-(1-q)^H ~ H q; fit ranges fixed here before running: A=2 m=4..11, A=3 m=6..13) (Theorem A.3 with the explore-then-commit
      oracle: 1 - P = 1 - (1 - q)^H, q = ((A-1)/A)^(m+1)); fit the slope on Monte Carlo; MATCH iff within 2 SE or 5 % of the closed form.
  T2  at fixed (A, m) with H q << 1, ln(1 - P_opt) is linear in ln H with slope 1 (the bound C * H * f); fit the slope on Monte Carlo.
  T3  the from-scratch cost ln E[episodes to first reward] is linear in H with slope ln A (Theorem 4.1: A^H); fit.
  T4  the coverage constant: 1 - P vs C = p^-(H-1); closed form vs Monte Carlo ratio (already Lock-d) -- here the fitted exponent of
      -ln(1-P_opt) ~ p^h-weighted sum.  Reported as ln(1-P) slope in h*ln(1/p) at small q.
Uses src/lock.py's mc_etc / scratch_first_reward (the lock-tested oracle).  Writes runs/full/theory.json.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import lock as L

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "full")
os.makedirs(OUT, exist_ok=True)
N = 2_000_000


def wls(x, y, se):
    """Weighted least squares y = a + b x; returns b, se_b, a."""
    w = 1.0 / np.maximum(se, 1e-12) ** 2
    X = np.stack([np.ones_like(x), x], 1)
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ (X.T @ (w * y))
    return float(beta[1]), float(np.sqrt(cov[1, 1])), float(beta[0])


def fail_rate(H, A, m, seed, p=1.0, N=N):
    rng = np.random.default_rng(seed)
    f = 1.0 - L.mc_etc(H, A, m, N, rng, p).mean()
    se = np.sqrt(max(f * (1 - f), 1e-12) / N)
    return float(f), float(se)


res = {}
# T1: slope in m (H = 6, A = 2 and A = 3)
for A, H, ms in ((2, 6, [4, 5, 6, 7, 8, 9, 10, 11]), (3, 6, [6, 7, 8, 9, 10, 11, 12, 13])):
    x, y, se, ex = [], [], [], []
    for i, m in enumerate(ms):
        f, s = fail_rate(H, A, m, 7000 + 10 * A + i)
        x.append(m); y.append(np.log(f)); se.append(s / f); ex.append(np.log(1 - L.p_opt_closed(H, A, m)))
    b, sb, a = wls(np.array(x, float), np.array(y), np.array(se))
    pred = float(np.log((A - 1) / A))
    # exact closed-form curve has slope -> ln((A-1)/A) only when q is small (ln(1-(1-q)^H) ~ ln(Hq)); report the closed-form fit too
    be, _, _ = wls(np.array(x, float), np.array(ex), np.ones(len(x)))
    # the MC slope is a WEIGHTED fit; the closed-form curve must be fitted with the SAME weights for a like-for-like comparison
    # (the unweighted slope above stays reported as it was; the 1.5-1.6 % 'MC vs closed-form' gap was a fit-weighting artefact)
    bew, _, _ = wls(np.array(x, float), np.array(ex), np.array(se))
    res[f"T1_A{A}_H{H}"] = dict(m=x, ln_fail=y, fitted_slope=b, slope_se=sb, predicted_slope=pred, closed_form_curve_fitted_slope=be,
                                rel_err_vs_pred=abs(b - pred) / abs(pred), z=(b - pred) / sb,
                                rel_err_vs_closed_curve=abs(b - be) / abs(be),
                                closed_form_curve_slope_same_weights=bew, z_vs_closed_curve_same_weights=(b - bew) / sb,
                                rel_err_vs_closed_curve_same_weights=abs(b - bew) / abs(bew),
                                judgment_declared=('MATCH' if abs(b - pred) / abs(pred) <= 0.05 or abs(b - pred) <= 2 * sb else 'CLOSE' if abs(b - pred) / abs(pred) <= 0.20 else 'DIVERGENT'))
# T2: exponent in H at fixed A = 2, m = 12 (q = 1.2e-4)
Hs, y, se = [2, 4, 8, 16, 32], [], []
for i, H in enumerate(Hs):
    f, s = fail_rate(H, 2, 12, 7100 + i, N=4_000_000)
    y.append(np.log(f)); se.append(s / f)
b, sb, a = wls(np.log(np.array(Hs, float)), np.array(y), np.array(se))
res["T2_A2_m12"] = dict(H=Hs, ln_fail=y, fitted_exponent=b, exponent_se=sb, predicted_exponent=1.0, z=(b - 1.0) / sb)
# T3: ln E[episodes to first reward] vs H (A = 2), 100k learners per H
Hs, y, se = [2, 3, 4, 5, 6, 7, 8], [], []
for i, H in enumerate(Hs):
    rng = np.random.default_rng(7200 + i)
    e = L.scratch_first_reward(H, 2, 100_000, rng).astype(float)
    mu, s = e.mean(), e.std(ddof=1) / np.sqrt(len(e))
    y.append(float(np.log(mu))); se.append(float(s / mu))
b, sb, a = wls(np.array(Hs, float), np.array(y), np.array(se))
res["T3_A2"] = dict(H=Hs, ln_E_episodes=y, fitted_slope=b, slope_se=sb, predicted_slope=float(np.log(2)), z=(b - np.log(2)) / sb)
# T4: coverage. A = 2, m = 20, H = 6; fail = 1 - prod_h(1 - q_h) ~ sum_h (1 - p^h/2)^m / 2; compare MC to closed form per p
rows = []
for i, p in enumerate((1.0, 0.9, 0.8, 0.7, 0.6)):
    f, s = fail_rate(6, 2, 20, 7300 + i, p=p)
    cf = 1.0 - L.p_opt_cov_closed(6, 2, 20, p)
    rows.append(dict(p=p, C=p ** -(6 - 1), mc_fail=f, se=s, closed_fail=cf, z=(f - cf) / s))
C = np.array([r["C"] for r in rows]); f = np.array([r["mc_fail"] for r in rows])
# Corrected check (derivation, Remark A.7; written after seeing that the first version -- 'within C*H*q' -- failed for p < 1): the theorem's
# per-step term is C(h) * [d_g(s*_h) * P(wrong_h)] with C(h) = p^-h and d_g(s*_h) = p^h, so the bound is the UNION bound sum_h P(wrong_h)
# (tight up to the overlap), with P(wrong_h) = (1 - p^h/A)^m (A-1)/A.  Check: max_h P(wrong_h) <= fail <= sum_h P(wrong_h).
ub = [sum(L.q_stage(2, 20, r["p"] ** h) for h in range(6)) for r in rows]
lb = [max(L.q_stage(2, 20, r["p"] ** h) for h in range(6)) for r in rows]
res["T4_cov"] = dict(rows=rows, monotone=bool(np.all(np.diff(f) >= -1e-9)), union_bound=ub, max_stage=lb,
                     within_union_bound=[bool(r["mc_fail"] <= u + 5 * r["se"] and r["mc_fail"] >= l - 5 * r["se"]) for r, u, l in zip(rows, ub, lb)],
                     max_abs_z_vs_closed=float(max(abs(r["z"]) for r in rows)))
json.dump(res, open(os.path.join(OUT, "theory.json"), "w"), indent=1)
for k, v in res.items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a not in ("m", "H", "ln_fail", "ln_E_episodes", "rows")})
