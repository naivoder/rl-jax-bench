"""small-scale, step 1 (no learning): tune the hand rule on the 20 tuning scenarios, evaluate the pre-declared dt
conditional on the 16 validation scenarios, and evaluate every fixed policy on the small-scale scenarios.

Declared before running (README 'small-scale'): hand-rule objective = mean episode sum of GOSPA distance (lower is
better), ties broken by M_cov; grid k_rev in {2, 4, 8, 12} x P_rev in {3000, 4000, 4500}.
The final-evaluation scenarios (fold_in(root, 0xE7A1)) are NOT touched.

    python scripts/small_baselines.py [--n-small 100] [--uniform-speed]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import EnvConfig, K_VALID, K_HAND_TUNE, K_SMALL, K_BASE_ACT  # noqa: E402
from src import baselines as B, eval as E  # noqa: E402

GRID_K = (2, 4, 8, 12)
GRID_P = (3000.0, 4000.0, 4500.0)


def run(pinit, pol, sk, cfg, seed_idx=0):
    pk = jax.vmap(lambda k: jax.random.fold_in(jax.random.fold_in(k, K_BASE_ACT), seed_idx))(sk)
    return jax.tree.map(np.asarray, jax.jit(lambda a, b: E.eval_many(pinit, pol, a, b, cfg))(sk, pk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-small", type=int, default=100)
    ap.add_argument("--uniform-speed", action="store_true", help="original-reading scene (scene audit only)")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--out", default=os.path.join(C.REPO, "runs", "small", "baselines.json"))
    args = ap.parse_args()
    C.check_device()
    cfg = EnvConfig()
    if args.uniform_speed:
        cfg = replace(cfg, truth_vel="uniform_speed")
    if args.dt is not None:
        cfg = replace(cfg, dt=args.dt)
    root = jax.random.PRNGKey(15)
    tm = C.Timer()
    # 1. hand-rule tuning (training-distribution scenarios only)
    sk_tune = E.scenario_keys(root, K_HAND_TUNE, 20)
    tune = {}
    for k_rev, p_rev in itertools.product(GRID_K, GRID_P):
        pi, po = B.make_hand_rule(cfg, k_rev, p_rev)
        r = E.reduce_episodes(run(pi, po, sk_tune, cfg))
        tune[f"{k_rev}_{int(p_rev)}"] = dict(gospa_d=r["gospa_d"], m_cov=r["m_cov"], gospa_miss=r["gospa_miss"],
                                            mean_tracks=r["mean_tracks"], life_median=r["life_median"])
        C.metric("hand tune", k_rev=k_rev, p_rev=p_rev, gospa_d=r["gospa_d"], m_cov=r["m_cov"],
                 life=r["life_median"])
    best = min(tune, key=lambda k: (tune[k]["gospa_d"], tune[k]["m_cov"]))
    k_rev, p_rev = int(best.split("_")[0]), float(best.split("_")[1])
    C.info("hand rule tuned", k_rev=k_rev, p_rev=p_rev, sec=round(tm.lap(), 1))
    # 2. dt conditional (validation scenarios): median lifetime of the tuned hand rule, in updates after initiation
    sk_val = E.scenario_keys(root, K_VALID, 16)
    pi, po = B.make_hand_rule(cfg, k_rev, p_rev)
    rv = E.reduce_episodes(run(pi, po, sk_val, cfg))
    switch = bool(rv["life_median"] < 2)
    C.metric("dt conditional", life_median=rv["life_median"], switch_to_5ms=switch)
    # 3. all fixed policies on the small-scale scenarios
    sk = E.scenario_keys(root, K_SMALL, args.n_small)
    pols = dict(static=B.make_static(cfg), random=B.make_random(cfg), coverage=B.make_coverage(cfg),
                coverage_boustro=B.make_coverage(cfg, boustrophedon=True),
                hand=B.make_hand_rule(cfg, k_rev, p_rev), ceiling=B.make_ceiling(cfg))
    res = {}
    for n, (pi, po) in pols.items():
        out = run(pi, po, sk, cfg)
        res[n] = E.reduce_episodes(out)
        res[n]["per_episode"] = {k: np.asarray(v).tolist() for k, v in out.items() if k != "life_hist"}
        C.metric("baseline", arm=n, sr=res[n]["sr"], m_cov=res[n]["m_cov"], gospa_d=res[n]["gospa_d"],
                 gospa_miss=res[n]["gospa_miss"], gospa_loc=res[n]["gospa_loc"], gospa_false=res[n]["gospa_false"],
                 mean_tracks=res[n]["mean_tracks"], life=res[n]["life_median"])
    C.info("baselines done", sec=round(tm.lap(), 1))
    C.write_json(args.out, dict(cfg=dict(dt=cfg.dt, truth_vel=cfg.truth_vel), tune=tune, hand=dict(k_rev=k_rev, p_rev=p_rev),
                                dt_conditional=dict(life_median=rv["life_median"], switch=switch, validation=rv),
                                small=res, n_small=args.n_small))


if __name__ == "__main__":
    main()
