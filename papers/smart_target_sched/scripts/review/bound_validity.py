"""Review: which Lagrangian bound is wrong?  The paper's near-optimality headline (SS IV-C) needs a valid denominator.
Test on instances whose EXACT optimum is computable (open-loop enumeration, deterministic dynamics => open loop is optimal; idle allowed, as in the relaxation):
  V_ta  = finite-horizon TIME-AVERAGED discounted dual (the paper's eqs 13-20 relaxation: one multiplier, price lam*beta^t), lam maximised;
  V_ps  = PER-SLOT dual (one multiplier per slot; entropy-smoothed ascent, hard-min evaluation), a valid lower bound on the EXACT optimum by weak duality;
  J*    = exact optimum;  J_idx = the index policy (A.simulate).
Checks: V_ta <= V_ps <= J* <= J_idx on every instance (weak duality), and how much of J_idx/V_ta - 1 is bound looseness (J* / V_ta - 1) vs policy suboptimality (J_idx / J* - 1).
CPU only.  Output: stdout and runs/tables/review_bound_validity.md."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import jax, jax.numpy as jnp, numpy as np
from scipy.optimize import minimize_scalar
import model as M, algorithm as A, data as D
import bound_tests as BT
CASES = {"n2_k1_T10": (("reckless", "cautious"), (4.0, 4.0), (5.0, 1.0), 1, 10),
         "n3_k1_T8": (("reckless", "cautious", "reckless"), (2.0, 4.0, 6.0), (5.0, 1.0, 1.0), 1, 8),
         "n4_k2_T6": (("reckless", "reckless", "cautious", "cautious"), (2.0, 5.0, 2.0, 5.0), (5.0, 5.0, 1.0, 1.0), 2, 6)}
NREP = 10
beta = M.BETA
out = []
for name, (types, qct, d, K, T) in CASES.items():
    N = len(types); bank = M.scalar_bank(list(types), list(qct), list(d))
    rk = D.rep_keys(0, "X2", 50 + N, K, 40); p0k, tie, mode = D.split_rep_keys(rk)
    P0 = jnp.maximum(jax.vmap(lambda k: jax.random.uniform(k, (N,), minval=0.0, maxval=2.0))(p0k), 1e-12)[..., None, None]
    Jw = np.asarray(A.simulate(bank, P0, tie, mode, "whittle", K, A.Cfg(T=T, guard=False))["J"])
    tabs = [BT.Tab(types[n], qct[n], d[n]) for n in range(N)]
    for r in range(NREP):
        p0 = np.asarray(P0[r])[:, 0, 0]
        Jstar, _, _ = A.exact_enumerate(bank, P0[r], T, K, allow_idle=True)
        f = lambda lam: -BT.soft_dual(tabs, p0, lam * beta ** np.arange(T), K, T, beta, 0.0)[0]
        lam = minimize_scalar(f, bounds=(0.0, 400.0), method="bounded", options=dict(xatol=1e-4)).x
        Vta = -f(lam)
        Vps = BT.perslot_dual(list(types), list(qct), list(d), p0, K, T=T, lam0=lam)["V_perslot_lb"]
        out.append((name, r, Vta, Vps, Jstar, float(Jw[r])))
        print(f"{name} rep {r}: V_ta {Vta:.4f} <= V_ps {Vps:.4f} <= J* {Jstar:.4f} <= J_idx {Jw[r]:.4f}  valid={Vta <= Vps + 1e-6 and Vps <= Jstar + 1e-6 and Jstar <= Jw[r] + 1e-9}", flush=True)
rows = []
for name in CASES:
    o = np.array([x[2:] for x in out if x[0] == name])
    ok = bool(np.all(o[:, 0] <= o[:, 1] + 1e-6) and np.all(o[:, 1] <= o[:, 2] + 1e-6) and np.all(o[:, 2] <= o[:, 3] + 1e-9))
    m = o.mean(0)
    rows.append(f"| {name} | {len(o)} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} | {100 * (m[3] / m[0] - 1):.2f} % | {100 * (m[2] / m[0] - 1):.2f} % | {100 * (m[3] / m[2] - 1):.2f} % | {100 * (m[3] / m[1] - 1):.2f} % | {'yes' if ok else 'NO'} |")
hdr = ["| case | reps | V_ta | V_ps | J* (exact) | J_idx | gap to V_ta (the paper's kind of gap) | of which bound looseness J*/V_ta-1 | of which policy suboptimality J_idx/J*-1 | gap to V_ps | chain V_ta<=V_ps<=J*<=J_idx in every rep |", "|---|---|---|---|---|---|---|---|---|---|---|"]
txt = "\n".join(hdr + rows)
print(txt)
os.makedirs(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs'), 'tables'), exist_ok=True)
open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs'), 'tables', 'review_bound_validity.md'), 'w').write(txt + "\n")
