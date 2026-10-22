"""T1 (pre-declared theory check P6): does the Lemma-1 one-step bound dominate the TRUE expected single-Bernoulli GOSPA cost?

Generative model (no clutter, 2 sensors, the planning assumptions of the paper): the target exists w.p. r with state ~ N(0, P0); sensor s detects it
w.p. p_s and returns z_s = H x + N(0, 2 I).  The exact Bayesian posterior (r', x_hat, P') is formed from the outcome, the Bayes-optimal report rule
(min-form of eq. 31) is applied and the REALIZED cost is c^2/2 (missed or false) or min(|e|^2, c^2) (reported, exists).  The MC mean is compared with
B_exact = sum_h p*(h) C(r^h, P^h) (exact hypothesis probabilities) and B_paper = same with the paper's eq.-38 product probabilities.
Pure numpy; uses the independent oracle in tests/oracle_toy.py.  Usage: python scripts/theory_t1.py [--n 400000] [--json out.json]
"""
import argparse, itertools, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import oracle_toy as O

C2 = O.C2


def mc_cost(r, a, pds, n, rng):
    P0 = np.diag([a, 1.0, a, 1.0])
    exists = rng.random(n) < r
    x = rng.multivariate_normal(np.zeros(4), P0, n)
    det = np.stack([rng.random(n) < p for p in pds], axis=1) & exists[:, None]
    Ri = np.linalg.inv(O.R)
    HtRiH = O.H.T @ Ri @ O.H
    cost = np.empty(n)
    # group by detection pattern
    for h in itertools.product((False, True), repeat=len(pds)):
        sel = np.all(det == np.array(h), axis=1)
        idx = np.nonzero(sel)[0]
        if len(idx) == 0:
            continue
        nd = sum(h)
        if nd == 0:
            pn = np.prod([1 - p for p in pds])
            rp = r * pn / (1 - r + r * pn)
            Pp = P0
            xh = np.zeros((len(idx), 4))
        else:
            rp = 1.0
            Pp = np.linalg.inv(np.linalg.inv(P0) + nd * HtRiH)
            zs = np.zeros((len(idx), 2))
            for s, hs in enumerate(h):
                if hs:
                    zs += x[idx] @ O.H.T + rng.multivariate_normal(np.zeros(2), O.R, len(idx))
            xh = (Pp @ O.H.T @ Ri @ zs.T).T
        tr = Pp[0, 0] + Pp[2, 2]
        report = (1 - rp) * C2 / 2 + rp * min(tr, C2) < rp * C2 / 2
        e2 = (x[idx][:, 0] - xh[:, 0]) ** 2 + (x[idx][:, 2] - xh[:, 2]) ** 2
        ex = exists[idx]
        # BUT the pattern selection above conditions on exists for detections; targets that do not exist are handled below
        if report:
            cost[idx] = np.where(ex, np.minimum(e2, C2), C2 / 2)
        else:
            cost[idx] = np.where(ex, C2 / 2, 0.0)
    # non-existing targets produce the all-miss pattern h=(0,..,0); they are inside that group (det False) -> handled via `ex` above
    return float(cost.mean()), float(cost.std(ddof=1) / np.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(20261022)
    rows = []
    for r in (0.2, 0.5, 0.8):
        for av in (1.0, 10.0, 100.0, 1500.0):
            for pds in ((0.9, 0.5), (0.99, 0.99), (0.3, 0.3)):
                P0 = np.diag([av, 1.0, av, 1.0])
                mc, se = mc_cost(r, av, pds, a.n, rng)
                be = O.node(r, P0, list(pds), "exact", "pos")[0]
                bp = O.node(r, P0, list(pds), "paper", "pos")[0]
                rows.append(dict(r=r, a=av, pds=list(pds), mc=mc, se=se, B_exact=be, B_paper=bp,
                                 exact_ok=bool(be >= mc - 3 * se), paper_ok=bool(bp >= mc - 3 * se),
                                 rel_exact=(be - mc) / mc, rel_paper=(bp - mc) / mc))
                print("r=%.1f a=%6.0f pd=%s MC=%9.2f+-%.2f B_exact=%9.2f B_paper=%9.2f %s %s" %
                      (r, av, pds, mc, se, be, bp, "ok " if rows[-1]["exact_ok"] else "VIOL", "ok " if rows[-1]["paper_ok"] else "VIOL(paper)"), flush=True)
    n_ok = sum(r_["exact_ok"] for r_ in rows)
    n_pv = sum(not r_["paper_ok"] for r_ in rows)
    out = dict(cases=len(rows), exact_bound_holds=n_ok, paper_eq38_violations=n_pv,
               median_rel_exact=float(np.median([r_["rel_exact"] for r_ in rows])), median_rel_paper=float(np.median([r_["rel_paper"] for r_ in rows])),
               verdict="PASS" if n_ok >= 35 else "FAIL", rows=rows)
    print({k: v for k, v in out.items() if k != "rows"})
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
