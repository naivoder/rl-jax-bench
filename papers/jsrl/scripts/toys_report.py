"""Lock-check numbers (CPU, numpy): closed form vs exact enumeration vs Monte Carlo for the combination lock, Lock-c, Lock-cov.
usage: toys_report.py  ->  runs/toys/lock.json (+ printed table)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fractions import Fraction
import numpy as np
from src import lock as K

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "toys")
os.makedirs(OUT, exist_ok=True)
ROWS = [(10, 2, 10), (10, 2, 5), (8, 2, 4), (10, 3, 10), (20, 2, 10)]
res = dict(rows=[], exact=[], cov={}, scratch=[])
t0 = time.time()
for i, (H, A, m) in enumerate(ROWS):
    rng = np.random.default_rng(20261027 + i)
    N, hits = 2_000_000, 0
    for _ in range(N // 200_000):
        hits += int(K.mc_etc(H, A, m, 200_000, rng).sum())
    pe = hits / N; se = float(np.sqrt(pe * (1 - pe) / N))
    cf = K.p_opt_closed(H, A, m)
    q = K.q_stage(A, m)
    T = m * H
    scratch_any = 1 - (1 - A ** -H) ** T
    res["rows"].append(dict(H=H, A=A, m=m, T=T, q=q, closed=cf, mc=pe, mc_se=se, z=(pe - cf) / se, N=N, bound_Hq=H * q,
                            suboptimality=1 - cf, scratch_P_any_reward_le=scratch_any, scratch_expected_episodes=A ** H))
    print(f"H={H} A={A} m={m} T={T}: closed {cf:.6f}  MC {pe:.6f} +- {se:.6f}  z={(pe - cf) / se:+.2f}  1-P={1 - cf:.6f} <= Hq={H * q:.6f}  scratch P(reward in T)<={scratch_any:.4f}", flush=True)
# independent re-check of any row whose single-run |z| exceeds 2.5: pooled 8 x 1M trajectories from fresh seeds (reported, not a replacement)
for r in res["rows"]:
    if abs(r["z"]) > 2.5:
        tot = 0
        for sd in range(8):
            rng = np.random.default_rng(1000 + sd)
            tot += sum(int(K.mc_etc(r["H"], r["A"], r["m"], 250_000, rng).sum()) for _ in range(4))
        n8 = 8_000_000; pe = tot / n8; se = float(np.sqrt(pe * (1 - pe) / n8))
        r["recheck_8x1M"] = dict(mc=pe, mc_se=se, z=(pe - r["closed"]) / se)
        print(f"  re-check H={r['H']} A={r['A']} m={r['m']}: pooled 8x1M MC {pe:.6f} +- {se:.6f} vs closed {r['closed']:.6f}  z={(pe - r['closed']) / se:+.2f}", flush=True)
for (H, A, m) in [(1, 2, 1), (2, 2, 2), (3, 2, 3), (3, 3, 2), (4, 2, 4), (5, 2, 2), (6, 2, 2), (6, 3, 1), (3, 3, 3)]:
    ex = K.exact_etc(H, A, m)
    res["exact"].append(dict(H=H, A=A, m=m, exact=str(ex), exact_float=float(ex), closed=K.p_opt_closed(H, A, m), abs_err=abs(float(ex) - K.p_opt_closed(H, A, m))))
    print(f"exact enumeration H={H} A={A} m={m}: {float(ex):.15f} vs closed {K.p_opt_closed(H, A, m):.15f}  |err|={res['exact'][-1]['abs_err']:.2e}", flush=True)
# Lock-cov
H, A, m, p = 6, 2, 20, 0.7
rng = np.random.default_rng(20261027 + 90)
N, hits = 2_000_000, 0
for _ in range(N // 200_000):
    hits += int(K.mc_etc(H, A, m, 200_000, rng, p).sum())
pe = hits / N; se = float(np.sqrt(pe * (1 - pe) / N)); cf = K.p_opt_cov_closed(H, A, m, p)
ex = [K.exact_etc(h, a, mm, Fraction(pp)) for (h, a, mm, pp) in [(3, 2, 3, "0.7"), (4, 2, 2, "0.5")]]
res["cov"] = dict(H=H, A=A, m=m, p=p, C=p ** (-(H - 1)), closed=cf, mc=pe, mc_se=se, z=(pe - cf) / se,
                  exact_checks=[dict(closed=K.p_opt_cov_closed(3, 2, 3, 0.7), exact=float(ex[0])), dict(closed=K.p_opt_cov_closed(4, 2, 2, 0.5), exact=float(ex[1]))])
print(f"Lock-cov H=6 A=2 m=20 p=0.7: C={p ** (-5):.2f} closed {cf:.6f}  MC {pe:.6f} +- {se:.6f} z={(pe - cf) / se:+.2f}; exact checks {res['cov']['exact_checks']}", flush=True)
# Lock-c
for (H, A, n) in [(6, 2, 200_000), (8, 2, 200_000), (5, 3, 200_000)]:
    rng = np.random.default_rng(20261027 + 50 + H)
    ep = K.scratch_first_reward(H, A, n, rng)
    mean, se = float(ep.mean()), float(ep.std(ddof=1) / np.sqrt(n))
    res["scratch"].append(dict(H=H, A=A, n=n, mean=mean, se=se, expected=A ** H, z=(mean - A ** H) / se))
    print(f"Lock-c scratch H={H} A={A}: mean episodes to first reward {mean:.2f} +- {se:.2f} vs A^H={A ** H}  z={(mean - A ** H) / se:+.2f}", flush=True)
res["secs"] = time.time() - t0
json.dump(res, open(os.path.join(OUT, "lock.json"), "w"), indent=1)
