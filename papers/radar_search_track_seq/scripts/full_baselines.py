"""full-scale, fixed policies on the 100 final-evaluation scenarios (CPU; no training). Static / Random use the per-seed
action keys fold_in(fold_in(k_scen_i, 0xBA5E), seed) for seeds 0..4; Coverage (row-major and boustrophedon), the
tuned hand rule (frozen at small-scale: k_rev 12, P_rev 4500) and the declared ceiling  are deterministic given the
scenario, so one run serves every seed. Also E1-gamma (scan-value sensitivity, descriptive): SR of Static / Random /
Coverage at gamma_sv in {0.005, 0.01, 0.02, 0.05} x N_SV in {50, 100}.

    JAX_PLATFORMS=cpu python scripts/full_baselines.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import Config, K_EVAL_SCEN, K_BASE_ACT  # noqa: E402
from src import baselines as B, eval as E, data  # noqa: E402
sys.path.insert(0, os.path.join(C.REPO, "scripts"))
from full_learners import eval_scen_sha  # noqa: E402


def run(pinit, pol, sk, cfg, seed_idx=0):
    pk = jax.vmap(lambda k: jax.random.fold_in(jax.random.fold_in(k, K_BASE_ACT), seed_idx))(sk)
    return jax.tree.map(np.asarray, jax.jit(lambda a, b: E.eval_many(pinit, pol, a, b, cfg))(sk, pk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(C.REPO, "runs", "full", "baselines.json"))
    ap.add_argument("--skip-gamma", action="store_true")
    args = ap.parse_args()
    C.warn("full-scale fixed policies run on CPU BY DESIGN (no learning; keeps the GPU for the learners)")
    C.check_device()
    cfg_all = Config()
    cfg = cfg_all.env
    root = jax.random.PRNGKey(cfg_all.seed_root)
    sk = E.scenario_keys(root, K_EVAL_SCEN, cfg_all.n_eval_scenarios)
    sha = eval_scen_sha(cfg_all, sk)
    C.info("eval scenarios", **sha)
    g2 = json.load(open(os.path.join(C.REPO, "runs", "small", "baselines.json")))
    k_rev, p_rev = int(g2["hand"]["k_rev"]), float(g2["hand"]["p_rev"])
    tm = C.Timer()
    res = dict(eval_scenarios=sha, env_version=data.env_version(cfg), hand=dict(k_rev=k_rev, p_rev=p_rev), arms={})
    plan = [("static", B.make_static(cfg), range(5)), ("random", B.make_random(cfg), range(5)),
            ("coverage", B.make_coverage(cfg), [0]), ("coverage_boustro", B.make_coverage(cfg, True), [0]),
            ("hand", B.make_hand_rule(cfg, k_rev, p_rev), [0]), ("ceiling", B.make_ceiling(cfg), [0])]
    for name, (pi, po), seeds in plan:
        res["arms"][name] = {}
        for s in seeds:
            out = run(pi, po, sk, cfg, s)
            red = E.reduce_episodes(out)
            red["per_episode"] = {k: np.asarray(v).tolist() for k, v in out.items() if k != "life_hist"}
            res["arms"][name][str(s)] = red
            C.metric("full baseline", arm=name, seed=s, sr=red["sr"], m_cov=red["m_cov"], gospa_d=red["gospa_d"],
                     gospa_miss=red["gospa_miss"], mean_tracks=red["mean_tracks"], sec=round(tm.lap(), 1))
        C.write_json(args.out, res)
    if not args.skip_gamma:
        res["gamma_sweep"] = {}
        cg = replace(cfg, gospa_metrics=False)
        for gam in (0.005, 0.01, 0.02, 0.05):
            for nsv in (50, 100):
                c2 = replace(cg, gamma_sv=gam, n_sv=nsv)
                row = {}
                for name, mk in (("static", B.make_static), ("random", B.make_random), ("coverage", B.make_coverage)):
                    pi, po = mk(c2)
                    row[name] = float(np.mean(run(pi, po, sk, c2, 0)["sr"]))
                row["order_ok"] = bool(row["coverage"] > row["random"] > row["static"])
                res["gamma_sweep"][f"{gam}_{nsv}"] = row
                C.metric("gamma sweep", gamma=gam, n_sv=nsv, **row, sec=round(tm.lap(), 1))
                C.write_json(args.out, res)
    C.info("full baselines done")


if __name__ == "__main__":
    main()
