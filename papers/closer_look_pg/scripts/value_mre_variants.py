"""E2 divergence test (standing rule 1): is the high held-out Returns-MRE an artefact of HOW the metric is built?
On fresh held-out rollouts (10 x 2048 pairs, frozen normalisers, key k_instr(root, it, I_HELDOUT) -> fold_in 8, disjoint
from the training log's held-out rollout and from the value-floor rollouts) of every checkpoint, report V_theta's
Returns-MRE (as logged) next to:
  * complete-episode-only (no batch-end truncation of G),
  * the median instead of the mean of |V-G|/|G| (heavy tail),
  * the share of states with |G| < 0.1 * median|G| and their share of the mean.
Usage: JAX_PLATFORMS=cpu python scripts/value_mre_variants.py runs/full/<arm> [its...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402  (gpu_bootstrap before jax)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as A, data as D  # noqa: E402
from src import eval as E  # noqa: E402
from src.train import load_ckpt  # noqa: E402

run = sys.argv[1]
its = [int(x) for x in sys.argv[2:]] or [150, 450]
env_name, alg = os.path.basename(run.rstrip("/")).split("_")
ctx = E.Ctx(env_name, alg)
E.ACTOR_BATCH = 10
seeds = [0, 1, 2, 3, 4]
st0 = ctx.init_state(jax.random.PRNGKey(0))
like = {"ts": jax.tree_util.tree_map(lambda x: jnp.stack([x] * 5), st0),
        "norm": jax.tree_util.tree_map(lambda x: jnp.stack([x] * 5), D.norm_init(ctx.spec.obs_dim)), "it": jnp.int32(0)}


@jax.jit
def one(pi, v, norm, root, it):
    k = jax.random.fold_in(common.k_instr(root, it, common.I_HELDOUT), 8)
    tr = E.collect(ctx, pi, norm, jax.random.split(k, 10), 2048)
    vals = ctx.nets.v_apply(v, tr.obs)
    _, ret = E.advantages(ctx, tr, vals, "gae")
    rel = jnp.abs(vals - ret) / jnp.abs(ret)
    return rel, ret, tr.not_done, A.complete_episode_mask(tr.not_done)


out = {}
for it in its:
    ck = load_ckpt(os.path.join(common.REPO, run, "ckpt", f"it{it:04d}"), like)
    for i, s in enumerate(seeds):
        g = lambda x: jax.tree_util.tree_map(lambda y: y[i], x)
        rel, ret, nd, comp = map(np.asarray, one(g(ck["ts"]).pi, g(ck["ts"]).v, g(ck["norm"]), jax.random.PRNGKey(s), jnp.int32(it)))
        m_all = nd > 0.5
        m_cmp = m_all & comp
        small = m_all & (np.abs(ret) < 0.1 * np.median(np.abs(ret[m_all])))
        out[f"it{it}/seed{s}"] = r = dict(
            mre_logged_def=float(rel[m_all].mean()), mre_complete_only=float(rel[m_cmp].mean()),
            median_rel=float(np.median(rel[m_all])), median_rel_complete=float(np.median(rel[m_cmp])),
            frac_complete=float(m_cmp.sum() / m_all.sum()), frac_small_G=float(small.sum() / m_all.sum()),
            share_of_mean_from_small_G=float(rel[small].sum() / rel[m_all].sum()))
        common.metric("value variants", arm=os.path.basename(run), it=it, seed=s, **{k: round(v, 4) for k, v in r.items()})
common.write_json(os.path.join(common.REPO, run, "value_mre_variants.json"), out)
