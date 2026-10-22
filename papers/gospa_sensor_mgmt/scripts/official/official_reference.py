"""The authors' own pipeline (their numpy RNG, their rollouts) run at small scale as a REFERENCE for our JAX runs.

Each seed is one MC run of the official closed loop (tests/official_loop.py with injection disabled, so np.random drives measurements
and rollouts exactly as in main.py).  Reports the Table III statistic (mean over steps of the per-step RMS over seeds) per algorithm.
Usage: python scripts/official/official_reference.py <alg-name> <n-seeds> <clutter> [seed0]   (alg names as in src/model.py ALGS, GD only)."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from src import model as M
import official_loop as OL

alg = M.ALGS[sys.argv[1]]
n = int(sys.argv[2]); lam = float(sys.argv[3]); s0 = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
kind = "GOSPA" if alg.cost == "GD" else "KLD"
truth = M.load_truth()
g2 = []
t0 = time.time()
for i in range(n):
    rec = OL.run(kind, alg.T, alg.b_joint, alg.b_ind, truth, None, None, lam=lam, n_steps=200, seed=s0 + i, inject=False)
    g2.append(rec["gospa"])
    print(f"[INFO] {alg.name} lam={lam} seed={s0 + i} mean_d={np.sqrt(rec['gospa']).mean():.3f} wall={time.time() - t0:.0f}s", flush=True)
g2 = np.array(g2)
G = float(np.mean(np.sqrt(g2.mean(axis=0))))
print(f"[METRIC] official-code reference {alg.name} lam={lam} n={n}: G={G:.3f} (paper {M.PAPER_TABLE3[alg.name][M.CLUTTER_RATES.index(lam)]})")
OUT = os.path.join(os.environ.get("GOSPA_RUNS", "runs"), "ref_official")
os.makedirs(OUT, exist_ok=True)
json.dump(dict(alg=alg.name, lam=lam, n=n, seed0=s0, G=G, gospa_sq=g2.tolist(), commit=M.OFFICIAL_COMMIT),
          open(os.path.join(OUT, f"{alg.name}_lam{lam}_s{s0}.json"), "w"))
