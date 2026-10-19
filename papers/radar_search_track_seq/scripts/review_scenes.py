"""review resolution (README/): the fixed policies on the 100
final-evaluation scenarios under co-primary and sensitivity scenes, with every M_cov / GOSPA reading side by side.

Scenes (all other constants as the primary scene):
  primary   dt 50 ms (paper §2.1), trace deleter (companion), live + library GOSPA          -- golden vs full-scale
  dt5       dt 5 ms + trace 5000 (the companion's Table IV pairing)
  frob      dt 50 ms + Frobenius deleter (this paper's §2.1 text)
  dt5_frob  dt 5 ms + Frobenius deleter
  gnn       primary with the paper-text GNN (Munkres) associator 
  clutter   primary with 0.1 false alarms per look (NN associator)
  clutter5  primary with 0.5 false alarms per look 
  n12       12 targets (the other N/c pair of README). GOSPA NOT computed (exact enumeration is infeasible
            at 12 x 17 and c = 5 km only changes GOSPA), so M_cov / track counts only
  na20      20 x 20 action grid (full coverage per paper §2.3)
Policies: Static and Random x 3 action seeds (0, 1, 2; the full-scale key tree), Coverage, hand rule (12, 4500, frozen at
small-scale, not re-tuned per scene) and the declared ceiling. Descriptive sensitivity: nothing here is a learner result.

    JAX_PLATFORMS=cpu taskset -c 0-2 python scripts/review_scenes.py --scenes primary,dt5,...
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
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import Config, K_EVAL_SCEN, K_BASE_ACT  # noqa: E402
from src import baselines as B, eval as E, data  # noqa: E402

SCENES = {
    "primary": dict(),
    "dt5": dict(dt=0.005),
    "frob": dict(del_stat="frob"),
    "dt5_frob": dict(dt=0.005, del_stat="frob"),
    "gnn": dict(associator="gnn"),
    "clutter": dict(n_fa_slots=2, clutter_rate=0.1),
    "clutter5": dict(n_fa_slots=4, clutter_rate=0.5),
    "n12": dict(n_tgt=12, gospa_metrics=False),
    "na20": dict(n_a=20),
    # recheck fix stage (README, declared before running)
    "uniform_speed": dict(truth_vel="uniform_speed"),               #: the frozen original truth (descriptive)
    "filt_sv10": dict(init_vel_sigma=10.0, truth_vel_sigma=30.0),   #: filter prior only (truth as primary)
    "dt5_svrule": dict(dt=0.005, init_vel_sigma=18.0),              #: sigma_v by the declared coast-life rule
}
SEEDS = (0, 1, 2)


def run_seeds(pinit, pol, sk, cfg, seeds):
    """One jit for all seeds: tile the scenario keys and give each block its seed's action keys."""
    n = sk.shape[0]
    skk = jnp.concatenate([sk] * len(seeds), 0)
    pkk = jnp.concatenate([jax.vmap(lambda k: jax.random.fold_in(jax.random.fold_in(k, K_BASE_ACT), s))(sk)
                           for s in seeds], 0)
    out = jax.tree.map(np.asarray, jax.jit(lambda a, b: E.eval_many(pinit, pol, a, b, cfg))(skk, pkk))
    return [{k: v[i * n:(i + 1) * n] for k, v in out.items()} for i in range(len(seeds))]


def summarise(out, cfg):
    red = E.reduce_episodes(out)
    pe = {k: np.asarray(v) for k, v in out.items() if k != "life_hist"}
    der = E.derived_episode_metrics(pe, cfg.T, cfg.n_tgt, cfg.gospa_c)
    for k, v in der.items():
        fin = v[np.isfinite(v)]
        red[k] = float(fin.mean()) if fin.size else float("nan")
    lh = np.asarray(out["life_hist"])                      # [n, 64] tracks by number of updates
    nz = lh > 0
    max_upd = np.where(nz.any(1), (nz * np.arange(lh.shape[1])[None]).max(1), -1)
    red["max_upd_per_episode_mean"] = float(max_upd.mean())
    red["frac_ep_track_ge10_upd"] = float((max_upd >= 10).mean())
    red["frac_ep_track_ge50_upd"] = float((max_upd >= 50).mean())
    red["per_episode"] = {k: v.tolist() for k, v in pe.items()}
    red["per_episode_derived"] = {k: v.tolist() for k, v in der.items()}
    return red


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, "runs", "review"))
    args = ap.parse_args()
    C.warn("review scene grid runs on CPU BY DESIGN (fixed policies)")
    C.check_device()
    cfg_all = Config()
    root = jax.random.PRNGKey(cfg_all.seed_root)
    sk = E.scenario_keys(root, K_EVAL_SCEN, cfg_all.n_eval_scenarios)
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.scenes.split(","):
        kw = dict(SCENES[name])
        cfg = replace(cfg_all.env, gospa_lib=kw.get("gospa_metrics", True), **kw)
        tm = C.Timer()
        res = dict(scene=name, overrides=kw, env_version=data.env_version(cfg), arms={})
        plan = [("static", B.make_static(cfg), SEEDS), ("random", B.make_random(cfg), SEEDS),
                ("coverage", B.make_coverage(cfg), (0,)), ("hand", B.make_hand_rule(cfg, 12, 4500.0), (0,)),
                ("ceiling", B.make_ceiling(cfg), (0,))]
        for arm, (pi, po), seeds in plan:
            outs = run_seeds(pi, po, sk, cfg, seeds)
            res["arms"][arm] = {str(s): summarise(o, cfg) for s, o in zip(seeds, outs)}
            r0 = res["arms"][arm][str(seeds[0])]
            C.metric("review scene", scene=name, arm=arm, sr=r0["sr"], m_cov=r0["m_cov"], m_cov_lib=r0["m_cov_lib"],
                     no_track=r0["n_episodes_no_track"], mean_tracks=r0["mean_tracks"],
                     gospa_d=r0.get("gospa_d"), gospa_lib_d=r0.get("gospa_lib_d"), sec=round(tm.lap(), 1))
            C.write_json(os.path.join(args.out_dir, f"scene_{name}.json"), res)
        C.info("scene done", scene=name)


if __name__ == "__main__":
    main()
