"""review resolution: re-evaluate the full-scale learners' FINAL parameters (runs/full/train_<family>/
params_<enc>.pkl, saved by train.py) on the same 100 final-evaluation scenarios with the same per-lane policy keys
(stochastic, the pre-registered mode), with the library-reading GOSPA switched on. The live metrics must reproduce the
stored full-scale per-episode values (discrete counts exactly; continuous to the CPU/GPU round-off of RESULTS 'Scenario
identity'), so the library GOSPA is read off the very same rollouts. CPU by design (no GPU budget).

    JAX_PLATFORMS=cpu taskset -c 0-2 python scripts/review_learner_lib.py --family flat
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import Config, K_EVAL_SCEN  # noqa: E402
from src import train as TR, eval as E  # noqa: E402
from src.model import ActorCritic  # noqa: E402

ENC = {"flat": "flat", "bigru": "bigru", "mhsa": "mhsa"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(ENC))
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--arms", default="flat,bigru,mhsa", help="lanes to re-evaluate (default: the three primary arms)")
    args = ap.parse_args()
    C.warn("learner library-GOSPA re-evaluation on CPU BY DESIGN")
    C.check_device()
    cfg = Config()
    enc = ENC[args.family]
    cfg = replace(cfg, net=replace(cfg.net, encoder=enc))
    pk_ = pickle.load(open(os.path.join(C.REPO, "runs", "full", f"train_{args.family}", f"params_{enc}.pkl"), "rb"))
    meta = [tuple(x) for x in pk_["lanes"]]
    params = jax.tree.map(jnp.asarray, pk_["params"])
    ecfg = replace(cfg.env, gospa_metrics=True, gospa_lib=True)
    net = ActorCritic(cfg.net, replace(cfg.env, gospa_metrics=False))
    sk = E.scenario_keys(jax.random.PRNGKey(cfg.seed_root), K_EVAL_SCEN, cfg.n_eval_scenarios)
    keys = jnp.stack([TR.lane_key(cfg, a, s) for a, s in meta])

    def one(p, k_lane):
        pinit, pol = E.make_learned(net, p, False)
        pk = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(k_lane, 6), i))(jnp.arange(sk.shape[0]))
        return E.eval_many(pinit, pol, sk, pk, ecfg)
    f = jax.jit(one)
    res = {}
    for i, (a, s) in enumerate(meta):
        if a not in args.arms.split(","):
            continue
        tm = C.Timer()
        out = jax.tree.map(np.asarray, f(jax.tree.map(lambda x: x[i], params), keys[i]))
        stored = json.load(open(os.path.join(C.REPO, "runs", "full", a, f"seed_{s}", "metrics.json")))["per_episode"]
        disc = all(np.array_equal(np.asarray(stored[k]), out[k]) for k in ("n_init", "n_del", "n_hit", "n_det"))
        rel = float(np.nanmax(np.abs(np.asarray(stored["m_cov"], float) - out["m_cov"]) /
                              np.abs(np.asarray(stored["m_cov"], float))))
        gd = float(np.max(np.abs(np.asarray(stored["gospa_d"]) - out["gospa_d"]) / np.asarray(stored["gospa_d"])))
        res[f"{a}/{s}"] = dict(arm=a, seed=s, discrete_identical=bool(disc), max_rel_m_cov=rel, max_rel_gospa_d=gd,
                               per_episode={k: out[k].tolist() for k in
                                            ("gospa_d", "gospa_miss", "gospa_false", "gospa_lib_d", "gospa_lib_loc",
                                             "gospa_lib_miss", "gospa_lib_false", "lib_kill_ovf", "m_cov", "m_cov_lib")})
        C.metric("learner lib", arm=a, seed=s, discrete_identical=disc, max_rel_m_cov=rel, max_rel_gospa_d=gd,
                 gospa_d=float(out["gospa_d"].mean()), gospa_lib_d=float(out["gospa_lib_d"].mean()),
                 lib_kill_ovf=int(out["lib_kill_ovf"].sum()), sec=round(tm.lap(), 1))
        C.write_json(os.path.join(C.REPO, "runs", "review", f"learner_lib_{args.family}.json"), res)


if __name__ == "__main__":
    main()
