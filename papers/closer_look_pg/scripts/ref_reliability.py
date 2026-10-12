"""Recheck split-half reliability of the E1 1e6-pair reference gradient, per (arm, it, seed).

The full-scale reference is 100 fresh 1e4-step actors from key k_ref (src.eval.grad_collect). This script re-collects
exactly those actors (same keys, same frozen normaliser, the run's own env mode), recomputes the full reference
(checked against the saved <tag>_gref_it<k>.npz), and splits the actors into two disjoint halves of 5e5 pairs:
  rho   = cos(g_half_A, g_half_B)                      (split-half agreement)
  r_ref = sqrt(2 rho / (1 + rho))                      estimate of cos(g_ref_full, g_true); 0 if rho <= 0)
A to_true cell whose r_ref < 0.5 is labelled "not measurable at a 1e6 reference" (pre-declared).

Usage: python scripts/ref_reliability.py --run runs/full/hopper_ppo --its 0 150 300 450 --gref_tags full_e1a full_e1b
       -> runs/recheck/ref_reliability/<arm>.json
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402  (bootstrap first)
from src.common import info, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src import eval as E  # noqa: E402
from src.configs.default import INSTRUMENTS  # noqa: E402
from src.train import load_ckpt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--its", type=int, nargs="+", required=True)
    ap.add_argument("--gref_tags", nargs="+", default=["full_e1a", "full_e1b"])
    ap.add_argument("--ref_pairs", type=int, default=INSTRUMENTS["grad_reference_pairs"])
    ap.add_argument("--chunk", type=int, default=INSTRUMENTS["rollout_chunk"])
    ap.add_argument("--seed_idx", type=int, nargs="*", default=None, help="subset of the stacked seeds (indices)")
    ap.add_argument("--actor_batch", type=int, default=0, help="CPU placement: actors per lax.map group")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    E.ACTOR_BATCH = args.actor_batch
    common.check_device()
    run_dir = os.path.join(common.REPO, args.run)
    env_name, alg = E.parse_run_name(run_dir)
    ctx = E.Ctx(env_name, alg, run_dir=run_dir)
    seeds = sorted(int(d.split("_")[1]) for d in os.listdir(run_dir) if d.startswith("seed_"))
    S = len(seeds)
    st0 = ctx.init_state(jax.random.PRNGKey(0))
    like = {"ts": jax.tree_util.tree_map(lambda x: jnp.stack([x] * S), st0),
            "norm": jax.tree_util.tree_map(lambda x: jnp.stack([x] * S), D.norm_init(ctx.spec.obs_dim)),
            "it": jnp.int32(0)}
    n_act = args.ref_pairs // args.chunk
    half = n_act // 2

    def one(pi, v, norm, root, it):
        k_ref = common.k_instr(root, it, common.I_REF)                       # == eval.grad_collect
        tr = E.collect(ctx, pi, norm, jax.random.split(k_ref, n_act), args.chunk)
        sl = lambda a, b: jax.tree_util.tree_map(lambda x: x[a:b], tr)
        return (E.grad_of(ctx, pi, v, tr, "gae"), E.grad_of(ctx, pi, v, sl(0, half), "gae"),
                E.grad_of(ctx, pi, v, sl(half, n_act), "gae"))

    f = jax.jit(lambda p, v, nm, r, it: jax.lax.map(lambda a: one(*a), (p, v, nm, r, it)))
    out = {"run": args.run, "ref_pairs": args.ref_pairs, "chunk": args.chunk, "halves_pairs": half * args.chunk,
           "env_post_step_kinematics": bool(ctx.cfg["env_post_step_kinematics"]), "cells": {}}
    for it in args.its:
        ck = load_ckpt(os.path.join(run_dir, "ckpt", f"it{it:04d}"), like)
        idx = args.seed_idx if args.seed_idx is not None else list(range(S))
        sub = lambda t: jax.tree_util.tree_map(lambda x: x[jnp.asarray(idx)], t)
        roots = jnp.stack([jax.random.PRNGKey(seeds[i]) for i in idx])
        t0 = common.Timer()
        g, ha, hb = f(sub(ck["ts"].pi), sub(ck["ts"].v), sub(ck["norm"]), roots, jnp.full((len(idx),), it, jnp.int32))
        jax.block_until_ready(g)
        for j, s in enumerate([seeds[i] for i in idx]):
            saved = None
            for tag in args.gref_tags:
                p = os.path.join(run_dir, f"seed_{s}", f"{tag}_gref_it{it}.npz")
                if os.path.exists(p):
                    saved = np.load(p)["g_ref"]
            rho = float(A.cosine(ha[j], hb[j]))
            r_ref = float(np.sqrt(2 * rho / (1 + rho))) if rho > 0 else 0.0
            cell = dict(rho_split_half=rho, r_ref=r_ref, measurable=bool(r_ref >= 0.5),
                        cos_full_vs_saved=None if saved is None else float(A.cosine(g[j], jnp.asarray(saved))),
                        cos_halfA_full=float(A.cosine(ha[j], g[j])), cos_halfB_full=float(A.cosine(hb[j], g[j])))
            out["cells"][f"it{it}/seed{s}"] = cell
            info("ref reliability", run=args.run, it=it, seed=s, **{k: v for k, v in cell.items() if v is not None})
        info("timing", it=it, seconds=round(t0.lap(), 1))
    od = os.path.join(common.REPO, "runs", "recheck", "ref_reliability")
    os.makedirs(od, exist_ok=True)
    write_json(os.path.join(od, f"{os.path.basename(run_dir)}{args.tag}.json"), {**out, **common.git_info()})


if __name__ == "__main__":
    main()
