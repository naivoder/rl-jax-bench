"""small-scale, step 2: all learner arms at the small-scale proxy, one process, families run in sequence; then each lane is
evaluated on the small-scale scenarios (stochastic primary, deterministic descriptive) and the MSE-BC pre-training-only arm
is evaluated for look concentration (P-BCmse, descriptive).

    python scripts/small_learners.py --steps 100000 --n-eval 32 --out runs/small
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import Config, K_SMALL, to_dict  # noqa: E402
from src import train as TR, eval as E, data  # noqa: E402

PLAN = [("mhsa", ["mhsa", "mhsa_nobc", "mhsa_noae", "mhsa_lit"]), ("flat", ["flat"]), ("bigru", ["bigru"])]
SEEDS = {"mhsa": [0, 1, 2], "mhsa_nobc": [0, 1], "mhsa_noae": [0, 1], "mhsa_lit": [0, 1], "flat": [0, 1],
         "bigru": [0, 1], "mhsa_bcmse": [0, 1]}


def eval_lanes(cfg, net, params, keys, sk, deterministic):
    ecfg = replace(cfg.env, gospa_metrics=True)

    def one(p, k_lane):
        pinit, pol = E.make_learned(net, p, deterministic)
        pk = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(k_lane, 6), i))(jnp.arange(sk.shape[0]))
        return E.eval_many(pinit, pol, sk, pk, ecfg)
    return jax.tree.map(np.asarray, jax.jit(jax.vmap(one))(params, keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(C.REPO, "runs", "small"))
    ap.add_argument("--only", default=None, help="comma list of families to run")
    args = ap.parse_args()
    C.check_device()
    cfg = Config()
    root = jax.random.PRNGKey(cfg.seed_root)
    sk = E.scenario_keys(root, K_SMALL, args.n_eval)
    scen_sha = hashlib.sha256(np.asarray(jax.vmap(lambda k: data.sample_truth(k, cfg.env))(sk)).tobytes()).hexdigest()
    plan = [p for p in PLAN if args.only is None or p[0] in args.only.split(",")]
    for fam, arms in plan:
        tm = C.Timer()
        lm = [(a, s) for a in arms for s in SEEDS[a]]
        net, lanes, meta, summary = TR.run_family(cfg, arms, [], os.path.join(args.out, "train"), args.steps,
                                                  args.steps, ckpt_every_it=0, lanes_meta=lm)
        keys = jnp.stack([TR.lane_key(cfg, a, s) for a, s in meta])
        cfgf = replace(cfg, net=replace(cfg.net, encoder=TR.ARMS[arms[0]]["encoder"]))
        ev_s = eval_lanes(cfgf, net, lanes.params, keys, sk, False)
        ev_d = eval_lanes(cfgf, net, lanes.params, keys, sk, True)
        for i, (a, s) in enumerate(meta):
            red_s = E.reduce_episodes({k: v[i] for k, v in ev_s.items()})
            red_d = E.reduce_episodes({k: v[i] for k, v in ev_d.items()})
            d = summary[i]
            m = dict(arm=a, seed=s, stage="small", env_steps=args.steps, eval_stochastic=red_s,
                     eval_deterministic=red_d, per_episode={k: v[i].tolist() for k, v in ev_s.items() if k != "life_hist"},
                     curve=d["curve"], pretrain=d["pretrain"], train_hist=d["train_hist"], timing=d["timing"],
                     env_version=d["env_version"], small_scenarios_sha256=scen_sha, n_eval=args.n_eval)
            C.write_json(os.path.join(args.out, a, f"seed_{s}", "metrics.json"), m)
            C.metric("small learner", arm=a, seed=s, sr=red_s["sr"], m_cov=red_s["m_cov"], gospa_d=red_s["gospa_d"],
                     gospa_miss=red_s["gospa_miss"], mean_tracks=red_s["mean_tracks"], det_m_cov=red_d["m_cov"],
                     det_gospa_d=red_d["gospa_d"])
        C.info("family done", family=fam, sec=round(tm.lap(), 1))
    if args.only is None or "bcmse" in args.only.split(","):
        # MSE-BC pre-training only (P-BCmse, descriptive)
        tm = C.Timer()
        cfgm = replace(cfg, net=replace(cfg.net, encoder="mhsa"))
        net, pretrain_lane = TR.make_pretrain(cfgm)
        lanes_meta = [("mhsa_bcmse", s) for s in SEEDS["mhsa_bcmse"]]
        keys = jnp.stack([TR.lane_key(cfgm, a, s) for a, s in lanes_meta])
        flags = TR.Flags(*(jnp.array([float(TR.ARMS[a][f]) for a, _ in lanes_meta]) for f in TR.Flags._fields))
        params, pre = jax.jit(jax.vmap(pretrain_lane))(keys, flags)
        ev = eval_lanes(cfgm, net, params, keys, sk[:16], False)
        for i, (a, s) in enumerate(lanes_meta):
            red = E.reduce_episodes({k: v[i] for k, v in ev.items()})
            C.write_json(os.path.join(args.out, a, f"seed_{s}", "metrics.json"),
                         dict(arm=a, seed=s, eval_stochastic=red, pretrain={k: np.asarray(v[i]).tolist() for k, v in pre.items()},
                              config=to_dict(cfgm)))
            C.metric("small bcmse", seed=s, central1=red["central1"], central3=red["central3"], sr=red["sr"],
                     bc_curve=np.asarray(pre["bc_curve"][i]).tolist())
        C.info("bcmse done", sec=round(tm.lap(), 1))


if __name__ == "__main__":
    main()
