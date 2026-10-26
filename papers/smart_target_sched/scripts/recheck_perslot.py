"""Recheck probes of the bound tests, CPU only.  Predictions declared BEFORE the run:

reps10: re-state the T1 per-slot-bound ratio for Fig 6 K=2 on ALL 10 replications (the first pass used 3).  V_auth_r = J_r/(1+gap_auth) per replication (J = Whittle arm),
     V_ps_r = per-slot LB per replication.  Declared criterion = the one in this file's sibling docstring (bound_tests.py T1): per-slot reproduces V_auth if |ratio-1| <= 0.5 %.
     PREDICTION: ratio ~ 1.04 (the 3-rep value 1.0204 moves with V_auth 823.2 -> 807.8, x1.019), i.e. T1 stays REJECTED (|dev| > 0.5 %).
gridres: the per-slot dual on Fig 5a K=2 at three grid resolutions (uniform step/geometric ratio: coarse 0.02/1.02, base 0.01/1.01, fine 0.005/1.005).
     PREDICTION: base-vs-fine differ by < 0.1 % so the by-product 'index within 0.3 % of the per-slot bound' (317.03 vs J = 318.0) is not grid noise.
    JAX_PLATFORMS=cpu nice -n 19 ionice -c3 python scripts/recheck_perslot.py gridres reps10   -> runs/recheck/perslot/{reps10,gridres}.json
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import bound_tests as G
import model as M, bound as B
from rl_jax_bench import jaxlog
import jax.numpy as jnp

OUT = os.path.join(HERE, "..", "runs", "recheck", "perslot"); os.makedirs(OUT, exist_ok=True)
_BaseTab = G.Tab


def make_tab(step, ratio):
    class TabR(_BaseTab):
        def __init__(self, typ, q, d):
            uni = np.arange(step, 20.0 + 1e-9, step)
            geo = 20.0 * ratio ** np.arange(1, int(np.ceil(np.log(100.0) / np.log(ratio))) + 1)
            self.g = np.concatenate([uni, geo]); Gn = self.g.size
            bank = M.scalar_bank([typ], [float(q)], [float(d)]); self.bank = bank; self.d = float(d)
            f = lambda a, x: np.asarray(M.phi(M.scalar_bank([typ] * len(x), [float(q)] * len(x), [float(d)] * len(x)), jnp.asarray(x)[:, None, None], jnp.full(len(x), a)))[:, 0, 0]  # noqa: E731
            x0, x1 = f(0.0, self.g), f(1.0, self.g)
            self.out0 = x0 > self.g[-1]
            def idx(x):
                j = np.clip(np.searchsorted(self.g, np.minimum(x, self.g[-1]), side="right") - 1, 0, Gn - 2)
                return j, (np.minimum(x, self.g[-1]) - self.g[j]) / (self.g[j + 1] - self.g[j])
            self.j0, self.w0 = idx(x0); self.j1, self.w1 = idx(x1)
            self.c = self.d * self.g
    return TabR


def sa2():
    rows = []
    scn, P0, J, gap_auth = G.setup("fig5a", 2)
    types, qct, d = list(scn.types), list(scn.qct), list(scn.d)
    a = B.vi_dual(types, qct, d, P0[:1], 2); lam = float(np.asarray(a["lam"])[0])
    Jw = float(G.ACTS_J["fig5a"][0])
    for name, (st, ra) in dict(coarse=(0.02, 1.02), base=(0.01, 1.01), fine=(0.005, 1.005)).items():
        G.Tab = make_tab(st, ra); t0 = time.time()
        o = G.perslot_dual(types, qct, d, P0[0], 2, lam0=lam)
        rows.append(dict(res=name, step=st, ratio=ra, V_perslot_lb=o["V_perslot_lb"], J_whittle=Jw, J_over_LB=Jw / o["V_perslot_lb"], sec=time.time() - t0, trace=o["trace"]))
        jaxlog.info("gridres", **{k: v for k, v in rows[-1].items() if k != "trace"})
        json.dump(rows, open(os.path.join(OUT, "gridres.json"), "w"), indent=1)
    G.Tab = _BaseTab


def sa1():
    rows = []
    scn, P0, J, gap_auth = G.setup("fig6", 2)           # NREP = 10 replications
    types, qct, d = list(scn.types), list(scn.qct), list(scn.d)
    Jr = np.asarray(G.ACTS_J["fig6"]); R = len(Jr)
    for r in range(R):
        t0 = time.time()
        a = B.vi_dual(types, qct, d, P0[r:r + 1], 2); lam = float(np.asarray(a["lam"])[0]); Vd = float(np.asarray(a["V_D"])[0])
        o = G.perslot_dual(types, qct, d, P0[r], 2, lam0=lam)
        rows.append(dict(rep=r, J=float(Jr[r]), V_timeavg=Vd, V_perslot=o["V_perslot_lb"], V_auth=float(Jr[r]) / (1 + gap_auth / 100), sec=time.time() - t0))
        jaxlog.info("reps10", **rows[-1])
        json.dump(dict(rows=rows, gap_auth=gap_auth), open(os.path.join(OUT, "reps10.json"), "w"), indent=1)
    ps = np.array([x["V_perslot"] for x in rows]); va = np.array([x["V_auth"] for x in rows]); td = np.array([x["V_timeavg"] for x in rows]); Jn = np.array([x["J"] for x in rows])
    rat = ps / va; rat_t = td / va
    summ = dict(n=R, ratio_perslot_over_auth_ratio_of_means=float(ps.mean() / va.mean()), per_rep_mean=float(rat.mean()), per_rep_se=float(rat.std(ddof=1) / np.sqrt(R)),
                ratio_timeavg_over_auth=float(td.mean() / va.mean()), per_rep_timeavg_mean=float(rat_t.mean()), per_rep_timeavg_se=float(rat_t.std(ddof=1) / np.sqrt(R)),
                J_over_perslot=float(Jn.mean() / ps.mean()), V_auth_mean=float(va.mean()), V_auth_first3=float(va[:3].mean()), perslot_mean=float(ps.mean()), perslot_first3=float(ps[:3].mean()))
    json.dump(dict(rows=rows, gap_auth=gap_auth, summary=summ), open(os.path.join(OUT, "reps10.json"), "w"), indent=1)
    jaxlog.info("reps10 summary", **summ)


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["gridres", "reps10"]):
        {"reps10": sa1, "gridres": sa2}[w]()
