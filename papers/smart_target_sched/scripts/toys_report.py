"""Toy-check numbers (printed as markdown): toy values, enumeration table, g plateaus, Fig-4 points, oracle deviations, lower-bound two-path check."""
import os, sys
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, jax, jax.numpy as jnp
from rl_jax_bench import jaxlog
import model as M, algorithm as A, data as D, oracle as O, bound as B
jaxlog.check_device()
U = {"reckless": ((0.9, 0.1), (0.2, 0.8)), "cautious": ((0.95, 0.05), (0.6, 0.4))}
b = M.scalar_bank(["reckless", "cautious"], [4.0, 4.0], 1.0); P = jnp.ones((2, 1, 1))
p0 = np.asarray(M.phi(b, P, jnp.zeros(2))).ravel(); p1 = np.asarray(M.phi(b, P, jnp.ones(2))).ravel()
my = np.asarray(A.index_myopic(b, P)[0]); mp = np.asarray(A.index_whittle(b, P, A.Cfg())[0])
ref = dict(p0=[2.558, 2.384], p1=[1.393851, 1.221866], my=[1.164149, 1.162134], mp=[1.149438, 1.206511])
print("## toy (reckless, cautious) at P=1: ours vs the hand derivation")
for k, v in dict(p0=p0, p1=p1, my=my, mp=mp).items():
    print(f"{k}: ours {v.round(6).tolist()} derivation {ref[k]} max|d|={np.abs(v-np.array(ref[k])).max():.2e}")
T = {2: (5.400066, 5.401880, 5.400066, 5.400066), 3: (8.817281, 8.817281, 9.033796, 9.033796), 4: (12.259849, 12.259849, 12.322772, 12.322772),
     8: (22.840805, 22.840805, 22.939443, 22.939443), 12: (29.806160, 29.806160, 29.921768, 29.921768)}
_, tie, mode = D.split_rep_keys(D.rep_keys(0, "G1", 99, 1, 1))
print("\n| T | exact (ours / derivation) | MP | myopic | TEV |\n|---|---|---|---|---|")
worst = 0
for t, e in T.items():
    ex = A.exact_enumerate(b, P, t, 1, 0.9, True)[0]
    js = [float(A.simulate(b, P[None], tie, mode, pol, 1, A.Cfg(T=t, guard=False, tie_first=True))["J"][0]) for pol in ("whittle", "myopic", "tev")]
    worst = max(worst, abs(ex - e[0]), *[abs(x - y) for x, y in zip(js, e[1:])])
    print(f"| {t} | {ex:.6f} / {e[0]:.6f} | {js[0]:.6f} / {e[1]:.6f} | {js[1]:.6f} / {e[2]:.6f} | {js[2]:.6f} / {e[3]:.6f} |")
print(f"max |ours - derivation| over the table: {worst:.2e}")
Ps = np.array([0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 15.0]); bb = M.scalar_bank(["reckless"] * 10, [4.0] * 10, 1.0)
f, g = A.fg_marginal(bb, jnp.asarray(Ps)[:, None, None], jnp.full(10, 4.0))
print("\ng(P,4) reckless Q=4 at P =", Ps.tolist(), "->", np.asarray(g).round(6).tolist(), " f(15,4)=", round(float(f[-1]), 3))
for typ, P_, q, plot in [("reckless", 1, 40, 4.05), ("cautious", 1, 40, 2.7), ("reckless", 10, 2, 31.6), ("cautious", 10, 2, 31.9), ("reckless", 15, 4, 55), ("cautious", 15, 4, 57)]:
    v = float(A.index_whittle(M.scalar_bank([typ], [q], 1.0), jnp.full((1, 1, 1), float(P_)), A.Cfg())[0][0])
    print(f"mp*({P_}) {typ} Q^CT={q}: ours {v:.3f}  Fig 4 reading {plot}")
rng = np.random.default_rng(3); N = 6
types = list(rng.choice(["reckless", "cautious"], N)); qct = list(rng.uniform(1.5, 12, N)); d = list(rng.choice([1., 5.], N)); P0 = rng.uniform(0.05, 2, N)
bk = M.scalar_bank(types, qct, d); tg = [O.scalar_tgt(*U[t], q, dd) for t, q, dd in zip(types, qct, d)]
Pj = jnp.asarray(P0)[:, None, None]
dev = {"whittle": np.max(np.abs(np.asarray(A.index_whittle(bk, Pj, A.Cfg(tau=100))[0]) / np.array([O.mp_index(t, np.array([[p]]), 100) for t, p in zip(tg, P0)]) - 1)),
       "passive_cont": np.max(np.abs(np.asarray(A.index_passive(bk, Pj, A.Cfg())[0]) / np.array([O.passive_cont_index(t, np.array([[p]]), 100) for t, p in zip(tg, P0)]) - 1)),
       "myopic": np.max(np.abs(np.asarray(A.index_myopic(bk, Pj)[0]) / np.array([O.myopic_index(t, np.array([[p]])) for t, p in zip(tg, P0)]) - 1))}
print("\nmax relative deviation JAX vs literal-equation numpy oracle (tau=100, N=6 random scalar targets):", {k: f"{v:.1e}" for k, v in dev.items()})
print("environment checksum (frozen 32-step trajectories, 6 files) max dev:", f"{M.reference_trajectory_error():.2e}")
types = ["reckless", "cautious", "reckless", "cautious"]; qc = [4.0] * 4; dd = [5., 1., 5., 1.]
for P0 in ([[0.5, 1.2, 0.01, 1.9]], [[1.0, 1.0, 1.0, 1.0]], [[3.0, 0.2, 7.0, 0.9]]):
    a = B.vi_dual(types, qc, dd, np.array(P0), K=1); bth = B.threshold_dual(types, qc, dd, np.array(P0), K=1)
    va, vb = float(a["V_D"][0]), float(bth["V_D"][0])
    print(f"LB P0={P0[0]}: VI path V_D={va:.4f} (lam*={float(a['lam'][0]):.3f}, max visited {float(a['pvis'][0]):.2f}, cap {B.P_CAP:g}); threshold-min path {vb:.4f}; rel diff {(va-vb)/vb:+.2e}")
