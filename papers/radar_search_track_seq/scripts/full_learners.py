"""full-scale: train one encoder family (all its arms x 5 seeds vmapped in ONE process), then evaluate every lane ONCE
on the 100 final-evaluation scenarios fold_in(root, 0xE7A1) (stochastic primary, deterministic descriptive).

Declared before launch in README. Budget control: the per-lane step count may be
lowered (never raised) through runs/full/train_<family>/n_it_override_<enc>, a timing-only decision.

    python scripts/full_learners.py --family mhsa --steps 491520 --out runs/full
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import Config, K_EVAL_SCEN  # noqa: E402
from src import train as TR, eval as E, data  # noqa: E402

FAMILIES = {"mhsa": ["mhsa", "mhsa_nobc", "mhsa_noae", "mhsa_lit"], "flat": ["flat"], "bigru": ["bigru"]}


def eval_scen_sha(cfg, sk):
    truth = np.asarray(jax.vmap(lambda k: data.sample_truth(k, cfg.env))(sk), dtype=np.float64)
    return dict(truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
                keys_sha256=hashlib.sha256(np.asarray(jax.random.key_data(sk) if hasattr(jax.random, "key_data")
                                                      else sk).tobytes()).hexdigest())


def eval_lanes(cfg, net, params, keys, sk, deterministic, chunk):
    ecfg = replace(cfg.env, gospa_metrics=True)

    def one(p, k_lane):
        pinit, pol = E.make_learned(net, p, deterministic)
        pk = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(k_lane, 6), i))(jnp.arange(sk.shape[0]))
        return E.eval_many(pinit, pol, sk, pk, ecfg)
    f = jax.jit(jax.vmap(one))
    outs = []
    L = keys.shape[0]
    for i in range(0, L, chunk):
        sl = slice(i, min(L, i + chunk))
        outs.append(jax.tree.map(np.asarray, f(jax.tree.map(lambda x: x[sl], params), keys[sl])))
    return jax.tree.map(lambda *xs: np.concatenate(xs, 0), *outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--arms", default=None, help="subset of the family's arms")
    ap.add_argument("--curve-every", type=int, default=49_152)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(C.REPO, "runs", "full"))
    args = ap.parse_args()
    t_start = time.time()
    plat = C.check_device()
    cfg = Config()
    arms = args.arms.split(",") if args.arms else FAMILIES[args.family]
    seeds = [int(s) for s in args.seeds.split(",")]
    lm = [(a, s) for a in arms for s in seeds]
    tr_dir = os.path.join(args.out, "train_" + args.family)
    net, lanes, meta, summary = TR.run_family(cfg, arms, seeds, tr_dir, args.steps, args.curve_every,
                                              ckpt_every_it=2, lanes_meta=lm)
    t_train = time.time() - t_start
    # ---- final evaluation (the only use of the 0xE7A1 scenarios)
    root = jax.random.PRNGKey(cfg.seed_root)
    sk = E.scenario_keys(root, K_EVAL_SCEN, cfg.n_eval_scenarios)
    sha = eval_scen_sha(cfg, sk)
    C.info("eval scenarios", **sha)
    keys = jnp.stack([TR.lane_key(cfg, a, s) for a, s in meta])
    cfgf = replace(cfg, net=replace(cfg.net, encoder=TR.ARMS[arms[0]]["encoder"]))
    tm = C.Timer()
    ev_s = eval_lanes(cfgf, net, lanes.params, keys, sk, False, args.chunk)
    ev_d = eval_lanes(cfgf, net, lanes.params, keys, sk, True, args.chunk)
    t_eval = tm.lap()
    n_it = len(summary[0]["train_hist"].get("rew", []))
    steps_done = n_it * cfg.ppo.n_steps * cfg.ppo.n_envs
    for i, (a, s) in enumerate(meta):
        red_s = E.reduce_episodes({k: v[i] for k, v in ev_s.items()})
        red_d = E.reduce_episodes({k: v[i] for k, v in ev_d.items()})
        d = summary[i]
        m = dict(arm=a, seed=s, stage="full", env_steps=steps_done, iterations=n_it, eval_stochastic=red_s,
                 eval_deterministic=red_d,
                 per_episode={k: v[i].tolist() for k, v in ev_s.items() if k != "life_hist"},
                 per_episode_det={k: v[i].tolist() for k, v in ev_d.items() if k != "life_hist"},
                 curve=d["curve"], pretrain=d["pretrain"], train_hist=d["train_hist"], timing=d["timing"],
                 env_version=d["env_version"], eval_scenarios=sha, n_eval=int(sk.shape[0]), device=plat,
                 init_style=cfg.net.init_style)
        C.write_json(os.path.join(args.out, a, f"seed_{s}", "metrics.json"), m)
        C.metric("full learner", arm=a, seed=s, sr=red_s["sr"], m_cov=red_s["m_cov"], gospa_d=red_s["gospa_d"],
                 gospa_miss=red_s["gospa_miss"], mean_tracks=red_s["mean_tracks"], det_m_cov=red_d["m_cov"],
                 det_gospa_d=red_d["gospa_d"])
    wall = time.time() - t_start
    C.write_json(os.path.join(args.out, f"timing_{args.family}.json"),
                 dict(family=args.family, lanes=len(meta), steps_per_lane=steps_done, wall_s=wall, train_s=t_train,
                      eval_s=t_eval, gpu_h_process=wall / 3600, t_start=t_start, t_end=time.time(), device=plat))
    C.info("full family done", family=args.family, wall_s=round(wall, 1), gpu_h=round(wall / 3600, 4))


if __name__ == "__main__":
    main()
