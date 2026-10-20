"""Generators for the two full-run side files cited in the README plus the reverse budget match:

  proj_summary.json : equal-budget check. Per learner lane set (Case I and II, every variant), the test-time projected
                      dwells (learn_*_proj.json: sum tau <= Theta_max T0) vs OA at delta = 0.9, seed-level.
  kendall.json      : Kendall tau-b of our 5-method ranking (mean U) vs Table VII per case and calibration, and of the
                      3 baselines alone (paper ties, e.g. Case II DQL = OA, are kept as ties).
  oa_matched.json   : (written by scripts/oa_matched.py) OA given each learner seed's realised share.

Usage: python scripts/full_aux.py [--tag full|full_v2]   (numpy/scipy only; reads runs/<tag>/eval and report.json)
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_report import PAPER_U, Learn, seedvals_base  # noqa: E402
from rl_jax_bench.stats import mean_se, paired_compare, t_interval  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def proj_summary(ev_dir):
    out = {}
    for case in ("case1", "case2"):
        for algo in ("ddpg", "dql"):
            fp = os.path.join(ev_dir, f"learn_{algo}_{case}_{case}_proj.json")
            fn = os.path.join(ev_dir, f"learn_{algo}_{case}_{case}.json")
            if not os.path.exists(fp):
                continue
            lp, ln = Learn(fp), Learn(fn)
            for v in sorted({x for x, _ in lp.lanes}):
                ck = "A04a" if v.startswith("A04a") else "A04"
                b = json.load(open(os.path.join(ev_dir, f"base_{case}_{ck}.json")))
                uo, ud, ua = (seedvals_base(b[k], "U") for k in ("oa", "da", "ea"))
                bh = ud if ud.mean() >= ua.mean() else ua
                u = lp.vals(v, "U")
                pc = paired_compare(u, uo)
                G = (u - bh) / (uo - bh)
                key = f"{algo}[{v}]" if case == "case1" else f"case2|{algo}[{v}]"
                out[key] = dict(U=list(mean_se(u)), U_noproj=list(mean_se(ln.vals(v, "U"))),
                                vsOA=dict(diff=pc["diff_mean"], t95=pc["t95"], p=pc["t_p"], wins=int((u > uo).sum())),
                                G=list(mean_se(G)), G_med=float(np.median(G)),
                                share=float(lp.vals(v, "track_share").mean()), sat=float(lp.vals(v, "sat_rate").mean()),
                                per_seed=np.round(u).tolist())
    return out


def kendall(report):
    out = {}
    for case in ("case1", "case2", "case3f", "case3s", "case4"):
        for ck in ("A04", "A04a"):
            tab = report["cases"].get(f"{case}|{ck}")
            if not tab:
                continue
            ours = {k: tab[k]["U"]["mean"] for k in ("ea", "da", "oa")}
            for algo in ("ddpg", "dql"):
                nm = f"{algo}[{ck}]"
                if nm in tab:
                    ours[algo] = tab[nm]["U"]["mean"]
            meth = [m for m in ("oa", "ddpg", "da", "dql", "ea") if m in ours]
            paper = PAPER_U[case]
            t5 = kendalltau([ours[m] for m in meth], [paper[m] for m in meth])[0]
            tb = kendalltau([ours[m] for m in ("ea", "da", "oa")], [paper[m] for m in ("ea", "da", "oa")])[0]
            out[f"{case}|{ck}"] = dict(tau5=float(t5), tau_baselines=float(tb),
                                       ours_rank=sorted(meth, key=lambda m: -ours[m]),
                                       paper_rank=sorted(meth, key=lambda m: -paper[m]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full_v2")
    ap.add_argument("--check", action="store_true", help="compare with the existing files instead of writing")
    a = ap.parse_args()
    ev_dir = os.path.join(REPO, "runs", a.tag, "eval")
    rep = json.load(open(os.path.join(REPO, "runs", a.tag, "report.json")))
    res = dict(proj_summary=proj_summary(ev_dir), kendall=kendall(rep))
    for name, obj in res.items():
        p = os.path.join(ev_dir, f"{name}.json")
        if a.check:
            old = json.load(open(p))
            worst = 0.0
            for k in old:
                # compare the fields the existing file has (the full-run case2 rows had no share/sat; paper ranks
                # with ties may list tied methods in either order, so only the taus are compared for kendall)
                keep = ("tau5", "tau_baselines") if name == "kendall" else tuple(old[k])
                o_, n_ = {f: old[k][f] for f in keep}, {f: obj[k][f] for f in keep}
                a_, b_ = json.dumps(o_, sort_keys=True), json.dumps(n_, sort_keys=True, default=float)
                fa = np.asarray([float(x) for x in __import__("re").findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", a_)])
                fb = np.asarray([float(x) for x in __import__("re").findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", b_)])
                if fa.shape != fb.shape:
                    print(name, k, "SHAPE MISMATCH"); worst = np.inf; continue
                d = np.max(np.abs(fa - fb) / np.maximum(np.abs(fa), 1e-12)) if fa.size else 0.0
                worst = max(worst, d)
                txt_a = [x for x in __import__("re").findall(r'"([a-z]+)"', a_)]
                txt_b = [x for x in __import__("re").findall(r'"([a-z]+)"', b_)]
                if txt_a != txt_b:
                    print(name, k, "LABEL MISMATCH", txt_a, txt_b); worst = np.inf
            print(f"{name}: {len(old)} entries, max rel diff vs existing file {worst:.2e}")
        else:
            with open(p, "w") as f:
                json.dump(obj, f, indent=1, default=float)
            print("wrote", p)


if __name__ == "__main__":
    main()
