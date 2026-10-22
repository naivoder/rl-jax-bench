"""T2 + T3 of the pre-declared theory checks P6.

T2: decision regret of the JAX MCTS (T = 2, joint tree, GD, CFG-P semantics) against the exhaustive two-step reduced-tree values on 30 random two-Bernoulli
geometries (tests/test_search_oracle.py scenario()), as a function of the budget B and the UCT exploration constant eps.
    V_min(a1)  = C1(a1) + lam * min_{a2} C2(a1, a2)      (what eq. 40's nested min asks for, reduced tree)
    V_mean(a1) = C1(a1) + lam * mean_{a2} C2(a1, a2)     (what a uniform-random rollout's mean-backup estimates)
  regret_min  = V_min(chosen)  - min_a V_min(a);   regret_mean = V_mean(chosen) - min_a V_mean(a).
T3: reduced tree (merge at every node) vs full tree (eq. 40 expectation over the 2^S branches) two-step value for a single Bernoulli, 60 random cases,
  in both hypothesis-probability modes (paper eq. 38, exact).
Usage: python scripts/theory_t2_t3.py --json runs/report/t2t3.json
"""
import argparse, json, os, sys, time
import numpy as np

from rl_jax_bench import gpu_bootstrap
gpu_bootstrap.setup()
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
sys.path.insert(0, os.path.join(HERE, ".."))
import oracle_toy as O
from src import planner as PL
from src import model as M
import test_search_oracle as S


def t3(n_cases=60, seed=7):
    rng = np.random.default_rng(seed)
    lam = 0.9
    out = {"paper": [], "exact": []}
    for _ in range(n_cases):
        r = rng.uniform(0.1, 0.95)
        a = float(np.exp(rng.uniform(0, np.log(500))))
        P0 = np.diag([a, 1.0, a, 1.0])
        pd1 = list(rng.uniform(0.1, 0.99, 2))
        pd2 = list(rng.uniform(0.1, 0.99, 2))
        for mode in ("paper", "exact"):
            c1, rbar, Pbar = O.node(r, P0, pd1, mode, "pos")
            r2, x2, P2 = O.predict(rbar, np.zeros(4), Pbar)
            c2r = O.node(r2, P2, pd2, mode, "pos")[0]
            Vr = c1 + lam * c2r
            c2f = 0.0
            for h, p, rc, Pc in O.hypotheses(r, P0, pd1, mode):
                rr, xx, PP = O.predict(rc, np.zeros(4), Pc)
                c2f += p * O.node(rr, PP, pd2, mode, "pos")[0]
            Vf = c1 + lam * c2f
            out[mode].append((Vr, Vf))
    res = {}
    for mode, v in out.items():
        v = np.asarray(v)
        res[mode] = dict(frac_reduced_ge_full=float(np.mean(v[:, 0] >= v[:, 1] - 1e-9)), mean_ratio=float(np.mean(v[:, 0] / np.maximum(v[:, 1], 1e-12))),
                         median_ratio=float(np.median(v[:, 0] / np.maximum(v[:, 1], 1e-12))), n=len(v))
    res["verdict"] = "PASS" if res["paper"]["frac_reduced_ge_full"] >= 0.9 else "FAIL"
    return res


def t2(n_geom=30, n_streams=20, budgets=(49, 98, 196, 392, 784, 1568), epss=(0.05, 300.0)):
    T = 2
    lam = S.LAM
    geoms = []
    for g in range(n_geom):
        bel, pos = S.scenario(1000 + g)
        fe = S.feas_joint(pos)
        Vmin, Vmean = {}, {}
        for a1 in np.nonzero(fe)[0]:
            p1 = S.pos_of(pos, a1)
            c1, b1 = S.expand(bel, p1, 0)
            f2 = S.feas_joint(p1)
            c2 = np.array([S.expand(b1, S.pos_of(p1, a2), 1)[0] for a2 in np.nonzero(f2)[0]])
            Vmin[a1] = c1 + lam * c2.min()
            Vmean[a1] = c1 + lam * c2.mean()
        geoms.append((bel, pos, fe, Vmin, Vmean))
    res = {}
    for eps in epss:
        cfg = S.CFG.replace(uct_eps=eps)
        for B in budgets:
            plan = jax.jit(PL.make_mcts(cfg, 2, T, B, "GD"))
            rmin, rmean, hit_min, hit_mean = [], [], [], []
            for g, (bel, pos, fe, Vmin, Vmean) in enumerate(geoms):
                root = S._root(bel, T)
                bm, bmean = min(Vmin, key=Vmin.get), min(Vmean, key=Vmean.get)
                for s in range(n_streams):
                    rng = np.random.default_rng(10_000 * g + s)
                    o = plan(root, jnp.asarray(pos), jnp.asarray(rng.uniform(size=(B, 1))), jnp.asarray(rng.uniform(size=B)))
                    a = int(o["action"])
                    rmin.append(Vmin[a] - Vmin[bm]); rmean.append(Vmean[a] - Vmean[bmean])
                    hit_min.append(a == bm); hit_mean.append(a == bmean)
            res[f"eps={eps}|B={B}"] = dict(eps=eps, B=B, regret_min=float(np.mean(rmin)), regret_mean=float(np.mean(rmean)),
                                           hit_min=float(np.mean(hit_min)), hit_mean=float(np.mean(hit_mean)), n=len(rmin))
            print("[METRIC] T2 eps=%g B=%d regret_min=%.3f regret_mean=%.3f hit_min=%.3f hit_mean=%.3f" %
                  (eps, B, res[f"eps={eps}|B={B}"]["regret_min"], res[f"eps={eps}|B={B}"]["regret_mean"], res[f"eps={eps}|B={B}"]["hit_min"], res[f"eps={eps}|B={B}"]["hit_mean"]), flush=True)
    # power-law exponent of mean regret against B
    for eps in epss:
        bs = np.array(budgets, float)
        for key in ("regret_min", "regret_mean"):
            y = np.array([res[f"eps={eps}|B={int(B)}"][key] for B in bs])
            ok = y > 1e-9
            slope = float(np.polyfit(np.log(bs[ok]), np.log(y[ok]), 1)[0]) if ok.sum() >= 3 else None
            res[f"slope_{key}_eps={eps}"] = slope
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    r3 = t3()
    print("[METRIC] T3", json.dumps(r3), flush=True)
    r2 = t2(n_geom=4, n_streams=3, budgets=(49, 98, 196)) if a.quick else t2()
    print("[INFO] done", round(time.time() - t0, 1), "s")
    if a.json:
        json.dump(dict(T3=r3, T2=r2), open(a.json, "w"), indent=1)
