"""Amendment (README, paper-budget run): DESCRIPTIVE evaluation of each lane's FINAL parameters (not the
selected best model) on the same 100 evaluation scenarios, same scenario and policy keys, both modes. Never graded.

    python scripts/paper_budget_final_eval.py      # -> runs/paper_budget/final_params/<arm>/seed_<n>/metrics.json
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.configs.config import Config, K_EVAL_SCEN  # noqa: E402
from src import train as TR, eval as E  # noqa: E402
from src.model import ActorCritic  # noqa: E402
from scripts.full_learners import eval_scen_sha, eval_lanes  # noqa: E402

PB = os.path.join(C.REPO, "runs", "paper_budget")


def main():
    t0 = time.time()
    plat = C.check_device()
    cfg = Config()
    root = jax.random.PRNGKey(cfg.seed_root)
    sk = E.scenario_keys(root, K_EVAL_SCEN, cfg.n_eval_scenarios)
    sha = eval_scen_sha(cfg, sk)
    for fam in ("mhsa", "bigru", "flat"):
        d = pickle.load(open(os.path.join(PB, "train_" + fam, f"params_{fam}.pkl"), "rb"))
        meta = [tuple(x) for x in d["lanes"]]
        cfgf = replace(cfg, net=replace(cfg.net, encoder=fam))
        net = ActorCritic(cfgf.net, replace(cfgf.env, gospa_metrics=False))
        params = jax.tree.map(jnp.asarray, d["params"])
        keys = jnp.stack([TR.lane_key(cfg, a, s) for a, s in meta])
        ev_s = eval_lanes(cfgf, net, params, keys, sk, False, 5)
        ev_d = eval_lanes(cfgf, net, params, keys, sk, True, 5)
        for i, (a, s) in enumerate(meta):
            red_s = E.reduce_episodes({k: v[i] for k, v in ev_s.items()})
            red_d = E.reduce_episodes({k: v[i] for k, v in ev_d.items()})
            C.write_json(os.path.join(PB, "final_params", a, f"seed_{s}", "metrics.json"),
                         dict(arm=a, seed=s, params="final (iteration 180)", eval_stochastic=red_s,
                              eval_deterministic=red_d, eval_scenarios=sha, device=plat,
                              per_episode={k: v[i].tolist() for k, v in ev_s.items() if k != "life_hist"},
                              per_episode_det={k: v[i].tolist() for k, v in ev_d.items() if k != "life_hist"}))
            C.metric("final-params eval", arm=a, seed=s, m_cov=red_s["m_cov"], gospa_d=red_s["gospa_d"],
                     gospa_miss=red_s["gospa_miss"], det_m_cov=red_d["m_cov"])
    C.info("final-params eval done", wall_s=round(time.time() - t0, 1), gpu_h=round((time.time() - t0) / 3600, 4))


if __name__ == "__main__":
    main()
