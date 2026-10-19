"""Train + evaluate the descriptive DRQN encoder arms (src/drqn_arm.py), lanes vmapped per encoder.

    python scripts/run_drqn.py --encoders flat,bigru,mhsa --seeds 0,1 --steps 50000 --n-eval 32 --out runs/small
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from dataclasses import replace  # noqa: E402

from src.configs.config import Config, K_SMALL  # noqa: E402
from src import drqn_arm as DA, eval as E, data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="flat,bigru,mhsa")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(C.REPO, "runs", "small"))
    args = ap.parse_args()
    C.check_device()
    cfg = Config()
    st = DA.DRQNSettings()
    seeds = [int(s) for s in args.seeds.split(",")]
    sk = E.scenario_keys(jax.random.PRNGKey(cfg.seed_root), K_SMALL, args.n_eval)
    for enc in args.encoders.split(","):
        arm = "drqn_" + enc
        tm = C.Timer()
        net, init, run_chunk, n_iter = DA.make(cfg, st, enc, args.steps)
        keys = jnp.stack([DA.lane_key(cfg, arm, s) for s in seeds])
        carry = jax.jit(jax.vmap(init))(keys)
        chunk = jax.jit(jax.vmap(lambda c, k, g0: run_chunk(c, g0, args.chunk, k), in_axes=(0, 0, None)))
        hist = []
        for c0 in range(0, n_iter, args.chunk):
            t0 = time.perf_counter()
            carry, info = chunk(carry, keys, c0)
            info = jax.tree.map(lambda x: np.asarray(x).mean(1), info)
            hist.append({k: v.tolist() for k, v in info.items()})
            C.metric("drqn chunk", arm=arm, env_steps=(c0 + args.chunk) * st.n_envs, sec=round(time.perf_counter() - t0, 2),
                     r=float(info["r"].mean()), loss=float(info["loss"].mean()), q=float(info["q"].mean()),
                     eps=float(info["eps"].mean()), cov=float(info["cov"].mean()))
        params = carry[0].params
        ecfg = replace(cfg.env, gospa_metrics=True)

        def one(p):
            pinit, pol = DA.make_policy(net, p)
            return E.eval_many(pinit, pol, sk, sk, ecfg)
        ev = jax.tree.map(np.asarray, jax.jit(jax.vmap(one))(params))
        for i, s in enumerate(seeds):
            red = E.reduce_episodes({k: v[i] for k, v in ev.items()})
            C.write_json(os.path.join(args.out, arm, f"seed_{s}", "metrics.json"),
                         dict(arm=arm, seed=s, env_steps=args.steps, eval_greedy=red, train_hist=hist,
                              settings=st.__dict__, env_version=data.env_version(cfg.env)))
            C.metric("drqn eval", arm=arm, seed=s, sr=red["sr"], m_cov=red["m_cov"], gospa_d=red["gospa_d"],
                     gospa_miss=red["gospa_miss"], mean_tracks=red["mean_tracks"])
        C.info("drqn encoder done", arm=arm, sec=round(tm.lap(), 1))


if __name__ == "__main__":
    main()
