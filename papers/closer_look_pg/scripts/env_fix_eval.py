"""Recheck before/after on FIXED policies: each full-scale agent (checkpoint, frozen normaliser) is rolled out in
the legacy env and in the post-step-kinematics env with IDENTICAL keys; mean first-episode return (unbiased:
A = 64 fresh 1000-step actors, first episode each), episode length and the per-step |forward-reward difference| are
compared per agent (paired over the 5 training seeds).

Usage: python scripts/env_fix_eval.py --its 450 -> runs/recheck/env_fix/env_fix_eval.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402  (bootstrap first)
from src.common import info, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import data as D  # noqa: E402
from src import eval as E  # noqa: E402
from src.train import load_ckpt  # noqa: E402

from rl_jax_bench.stats import paired_compare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--its", type=int, nargs="+", default=[450])
    ap.add_argument("--actors", type=int, default=64)
    args = ap.parse_args()
    common.check_device()
    out = {}
    for arm in ("hopper_ppo", "hopper_trpo", "walker2d_ppo", "walker2d_trpo"):
        env_name, alg = arm.split("_")
        run_dir = os.path.join(common.REPO, "runs", "full", arm)
        ctxs = {m: E.Ctx(env_name, alg, {"env_post_step_kinematics": m == "post_step"}) for m in ("legacy", "post_step")}
        c0 = ctxs["legacy"]
        st0 = c0.init_state(jax.random.PRNGKey(0))
        like = {"ts": jax.tree_util.tree_map(lambda x: jnp.stack([x] * 5), st0),
                "norm": jax.tree_util.tree_map(lambda x: jnp.stack([x] * 5), D.norm_init(c0.spec.obs_dim)),
                "it": jnp.int32(0)}
        for it in args.its:
            ck = load_ckpt(os.path.join(run_dir, "ckpt", f"it{it:04d}"), like)
            res = {}
            for m, ctx in ctxs.items():
                def agent(pi, nm, s):
                    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(s), 424242), args.actors)
                    tr = E.collect(ctx, pi, nm, keys, int(ctx.cfg["episode_length"]))
                    sel, rets = D.unbiased_episode_returns(tr, int(ctx.cfg["episode_length"]))
                    r = jnp.sum(jnp.where(sel, rets, 0.0)) / jnp.maximum(jnp.sum(sel), 1)
                    l = jnp.sum(jnp.where(sel, tr.ep_len, 0.0)) / jnp.maximum(jnp.sum(sel), 1)
                    return r, l
                r, l = jax.jit(jax.vmap(agent))(ck["ts"].pi, ck["norm"], jnp.arange(5))
                res[m] = (np.asarray(r).tolist(), np.asarray(l).tolist())
            d = paired_compare(res["post_step"][0], res["legacy"][0])
            out[f"{arm}/{it}"] = {"legacy_return": res["legacy"][0], "post_step_return": res["post_step"][0],
                                  "legacy_len": res["legacy"][1], "post_step_len": res["post_step"][1],
                                  "diff_post_minus_legacy": d}
            info("env fix eval", arm=arm, it=it, legacy=float(np.mean(res["legacy"][0])),
                 post_step=float(np.mean(res["post_step"][0])), diff=d["diff_mean"], se=d["diff_se"], p=d["t_p"])
    write_json(os.path.join(common.REPO, "runs", "recheck", "env_fix", "env_fix_eval.json"), {**out, **common.git_info()})


if __name__ == "__main__":
    main()
