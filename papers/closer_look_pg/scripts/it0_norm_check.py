""" divergence test (standing rule 1): Hopper PPO iteration-0 gradient cosines are far above the paper's (ours and the
authors' code agree with each other). Hypothesis: the paper's iteration-0 agents carried EMPTY (unwarmed) normaliser
statistics (obs ZFilter std 0 -> clipped +-10 inputs; PPO RewardFilter std 0 -> every reward clipped to +-10), whereas
our checkpoint stores the statistics after rollout 0 . Same E1 instrument, same keys, two normaliser states:
'ckpt' (as in full-scale) and 'empty' (norm_init). Reference 1e5 pairs (scaled; labelled), m = 10, n in {1e2,1e3,2e3,1e4}.
Usage: JAX_PLATFORMS=cpu python scripts/it0_norm_check.py hopper_ppo [hopper_trpo ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as A, data as D  # noqa: E402
from src import eval as E  # noqa: E402
from src.train import load_ckpt  # noqa: E402

E.ACTOR_BATCH = 64
SIZES = (100, 1000, 2000, 10000)
out = {}
for arm in sys.argv[1:]:
    env_name, alg = arm.split("_")
    ctx = E.Ctx(env_name, alg)
    st0 = ctx.init_state(jax.random.PRNGKey(0))
    like = {"ts": jax.tree_util.tree_map(lambda x: jnp.stack([x] * 5), st0),
            "norm": jax.tree_util.tree_map(lambda x: jnp.stack([x] * 5), D.norm_init(ctx.spec.obs_dim)), "it": jnp.int32(0)}
    ck = load_ckpt(os.path.join(common.REPO, "runs", "full", arm, "ckpt", "it0000"), like)
    f = jax.jit(lambda pi, v, nm, root: E.gradient_block(ctx, pi, v, nm, root, jnp.int32(0), SIZES, 10, 10000, 100000,
                                                         raw_ablation=False))
    for s in range(5):
        g = lambda x: jax.tree_util.tree_map(lambda y: y[s], x)
        nm0 = g(ck["norm"])
        for state, nm in (("ckpt", nm0), ("empty", D.norm_init(ctx.spec.obs_dim)),
                          ("rew_empty", D.NormState(nm0.obs, D.runstat_init(())))):
            o = f(g(ck["ts"]).pi, g(ck["ts"]).v, nm, jax.random.PRNGKey(s))
            for n in SIZES:
                c = A.cos_stats(np.asarray(o["G"]["gae"][n]), np.asarray(o["g_ref"]), 10, 0)
                out[f"{arm}/{state}/seed{s}/{n}"] = {"to_true": c["to_true"], "pairwise": c["pairwise"]}
                common.metric("it0 norm test", arm=arm, state=state, seed=s, n=n, to_true=round(c["to_true"], 4),
                              pairwise=round(c["pairwise"], 4))
    for state in ("ckpt", "empty", "rew_empty"):
        for n in SIZES:
            tt = [out[f"{arm}/{state}/seed{s}/{n}"]["to_true"] for s in range(5)]
            pw = [out[f"{arm}/{state}/seed{s}/{n}"]["pairwise"] for s in range(5)]
            common.info("summary", arm=arm, state=state, n=n, to_true=f"{np.mean(tt):.3f}+-{np.std(tt, ddof=1) / 5 ** .5:.3f}",
                        pairwise=f"{np.mean(pw):.3f}+-{np.std(pw, ddof=1) / 5 ** .5:.3f}")
common.write_json(os.path.join(common.REPO, "runs", "full", "it0_norm_test.json"), out)
