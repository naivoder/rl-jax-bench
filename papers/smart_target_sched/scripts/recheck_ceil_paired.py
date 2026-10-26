"""Recheck of the ceil-grid hypothesis (CPU only; prediction declared before the run).  The T5 declared criterion ('ONE (scheme, step) reproduces V_auth within 0.5 % in all four scenarios at K = 2')
re-graded with PAIRED per-replication ratios in Fig 6 (10 replications): ratio_r = V_ceil0.075_r / (J_r / (1 + gap_auth)), reported as mean +- SE.  Also the unmodified V^D
(route A) per replication, and the same for ceil at the paper's stated step 0.01.  PREDICTION: Fig 6 ceil-0.075 mean ratio ~0.991 (the earlier 10-rep ratio of means) with SE < 0.5 %,
so the 0.5 % criterion fails by more than 1.5 SE; V^D ratio ~0.968-0.972.
    JAX_PLATFORMS=cpu nice -n 19 ionice -c3 python scripts/recheck_ceil_paired.py  -> runs/recheck/ceil_paired/ceil_paired.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import bound_tests as BT, bound as B
from rl_jax_bench import jaxlog
from scipy.optimize import minimize_scalar
T, beta = 100, 0.9
OUT = os.path.join(HERE, "..", "runs", "recheck", "ceil_paired")


def v_ceil(types, qct, d, P0r, K, step):
    cache = {}; tabs = []
    for n in range(len(types)):
        k = (types[n], float(qct[n]), float(d[n]))
        if k not in cache:
            cache[k] = BT.TabGrid(types[n], qct[n], d[n], step, "ceil", pmax=130.0)
        tabs.append(cache[k])
    f = lambda lam: -BT.soft_dual_grid(tabs, P0r, lam * beta ** np.arange(T), K, T, beta)
    return -minimize_scalar(f, bounds=(0.0, 300.0 * max(1.0, max(d))), method="bounded", options=dict(xatol=1e-4)).fun


def main():
    BT.NREP = 10
    scn, P0, J, gap_auth = BT.setup("fig6", 2)
    types, qct, d = list(scn.types), list(scn.qct), list(scn.d)
    Jr = np.asarray(BT.ACTS_J["fig6"]); rows = []
    for r in range(len(Jr)):
        t0 = time.time(); va = float(Jr[r]) / (1 + gap_auth / 100)
        VD = float(np.asarray(B.vi_dual(types, qct, d, P0[r:r + 1], 2)["V_D"])[0])
        rows.append(dict(rep=r, J=float(Jr[r]), V_auth=va, V_D=VD, V_ceil075=v_ceil(types, qct, d, P0[r], 2, 0.075), V_ceil001=v_ceil(types, qct, d, P0[r], 2, 0.01), sec=time.time() - t0))
        jaxlog.info("ceil_paired", **rows[-1]); json.dump(dict(rows=rows), open(os.path.join(OUT, "ceil_paired.json"), "w"), indent=1)
    s = {}
    for k in ("V_D", "V_ceil075", "V_ceil001"):
        rt = np.array([x[k] / x["V_auth"] for x in rows])
        s[k] = dict(mean=float(rt.mean()), se=float(rt.std(ddof=1) / np.sqrt(len(rt))), ratio_of_means=float(np.mean([x[k] for x in rows]) / np.mean([x["V_auth"] for x in rows])), n=len(rt))
    json.dump(dict(rows=rows, summary=s), open(os.path.join(OUT, "ceil_paired.json"), "w"), indent=1); jaxlog.info("ceil_paired summary", **{k: v["mean"] for k, v in s.items()})


if __name__ == "__main__":
    main()
