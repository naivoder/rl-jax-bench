"""Reverse budget match (verification): OA run at delta_s = learner seed s's realised mean track share,
on the same 20 k_test replica draws as that seed. The complement of the test-time projection (full_eval --project),
which cuts the learner to Theta_max instead and moves it off its training distribution.

Usage: python scripts/oa_matched.py --tag full_v2 [--lanes run:case:variant:calib,...] [--replica v2 --key_scheme v2]
Writes runs/<tag>/eval/oa_matched.json (per lane set: shares, learner U, OA@share U, OA@0.9 U, paired test).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import common  # noqa: E402
from src.common import info, load_calibration, load_yaml, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import data, eval as ev  # noqa: E402
import full_eval as G  # noqa: E402
from rl_jax_bench.stats import paired_compare  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = ("dql_case1:case1:A04a_dual50:A04a,dql_case1:case1:dual50:A04,ddpg_case2:case2:A04a:A04a,"
           "dql_case2:case2:A04a:A04a,dql_case1:case1:A04:A04,ddpg_case1:case1:A04a:A04a,ddpg_case1:case1:A04:A04,"
           "ddpg_case2:case2:A04:A04,dql_case2:case2:A04:A04,dql_case1:case1:A04a:A04a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full_v2")
    ap.add_argument("--lanes", default=DEFAULT)
    ap.add_argument("--replica", default="v2", choices=["v2", "v1", "v2o"])
    ap.add_argument("--key_scheme", default="v2", choices=["v2", "legacy"])
    ap.add_argument("--out_name", default="oa_matched.json")
    a = ap.parse_args()
    data.REPLICA, ev.SLOT_KEY = a.replica, a.key_scheme
    common.check_device()
    y, cal = load_yaml(), load_calibration()
    rk, _ = G.keys(jax.random.PRNGKey(20261019))
    evd = os.path.join(REPO, "runs", a.tag, "eval")
    out = {}
    for item in a.lanes.split(","):
        run, case, v, ck = item.split(":")
        L = json.load(open(os.path.join(evd, f"learn_{run}_{case}.json")))
        lanes = [tuple(x) for x in L["lanes"]]
        if not any(x[0] == v for x in lanes):
            continue
        sh = np.array([np.mean(L["replica"]["track_share"][lanes.index((v, s))]) for s in range(5)])
        Ul = np.array([np.mean(L["replica"]["U"][lanes.index((v, s))]) for s in range(5)])
        cfg = G.case_cfg(y, case)
        one = ev.make_evaluator(cfg, ev.make_lut(cfg), ev.make_policy("oa", cfg, y), G.T)
        scen = G.bcast(G.case_scen(cfg, case), 20)
        c0 = G.base_consts(cal, ck)
        f = jax.jit(jax.vmap(lambda d, k: one(None, dict(c0, delta=d), scen, k)))
        r = jax.device_get(f(jnp.asarray(sh), rk.reshape(5, 20, *rk.shape[1:])))
        Uo = np.asarray(r["U"]).mean(1)
        b = json.load(open(os.path.join(evd, f"base_{case}_{ck}.json")))["oa"]["replica"]["U"]
        Uo09 = np.asarray(b).reshape(5, 20).mean(1)
        pc = paired_compare(Ul, Uo)
        out[f"{run}|{case}|{v}"] = dict(share=sh.tolist(), U_learner=Ul.tolist(), U_oa_matched=Uo.tolist(),
                                        U_oa_09=Uo09.tolist(), diff=(Ul - Uo).tolist(), diff_mean=pc["diff_mean"],
                                        t95=pc["t95"], p_two=pc["t_p"], boot95=pc["boot95"],
                                        wins_vs_matched=int((Ul > Uo).sum()), wins_vs_09=int((Ul > Uo09).sum()))
        info("oa_matched", lane=f"{run}|{case}|{v}", diff=round(pc["diff_mean"], 1), p=round(pc["t_p"], 4),
             wins=int((Ul > Uo).sum()))
    write_json(os.path.join(evd, a.out_name), out)


if __name__ == "__main__":
    main()
