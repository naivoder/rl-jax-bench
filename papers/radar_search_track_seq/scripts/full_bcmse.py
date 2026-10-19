"""full-scale, E1-BCmse (descriptive, pre-training only, no PPO): the printed-MSE BC reading, 5 seeds, evaluated on the
100 final-evaluation scenarios with stochastic actions. P-BCmse: GO if > 90 % of looks fall in the central 3x3 cells.
CPU by design (pre-training only; keeps the GPU budget for the learners).

    JAX_PLATFORMS=cpu python scripts/full_bcmse.py
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import Config, K_EVAL_SCEN, to_dict  # noqa: E402
from src import train as TR, eval as E  # noqa: E402
sys.path.insert(0, os.path.join(C.REPO, "scripts"))
from full_learners import eval_lanes, eval_scen_sha  # noqa: E402


def main():
    if os.environ.get("JAX_PLATFORMS", "") == "cpu":
        C.warn("E1-BCmse runs on CPU BY DESIGN (pre-training only)")
    C.check_device()
    t0 = time.time()
    cfg = Config()
    cfgm = replace(cfg, net=replace(cfg.net, encoder="mhsa"))
    net, pretrain_lane = TR.make_pretrain(cfgm)
    meta = [("mhsa_bcmse", s) for s in range(5)]
    keys = jnp.stack([TR.lane_key(cfgm, a, s) for a, s in meta])
    flags = TR.Flags(*(jnp.array([float(TR.ARMS[a][f]) for a, _ in meta]) for f in TR.Flags._fields))
    params, pre = jax.jit(jax.vmap(pretrain_lane))(keys, flags)
    root = jax.random.PRNGKey(cfg.seed_root)
    sk = E.scenario_keys(root, K_EVAL_SCEN, cfg.n_eval_scenarios)
    ev = eval_lanes(cfgm, net, params, keys, sk, False, 5)
    for i, (a, s) in enumerate(meta):
        red = E.reduce_episodes({k: v[i] for k, v in ev.items()})
        C.write_json(os.path.join(C.REPO, "runs", "full", a, f"seed_{s}", "metrics.json"),
                     dict(arm=a, seed=s, stage="full", eval_stochastic=red, eval_scenarios=eval_scen_sha(cfg, sk),
                          pretrain={k: np.asarray(v[i]).tolist() for k, v in pre.items()}, config=to_dict(cfgm),
                          device=jax.devices()[0].platform, wall_s=time.time() - t0))
        C.metric("full bcmse", seed=s, central1=red["central1"], central3=red["central3"], sr=red["sr"])


if __name__ == "__main__":
    main()
