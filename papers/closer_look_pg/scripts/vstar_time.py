"""Recheck follow-up (decisive variant): is V_theta's held-out Returns-MSE gap irreducible return noise?
Refit V* on the SAME 1e6 pairs as full-scale (k_vstar) three ways and score all on the C7 held-out states (value_floor keys):
  vstar      : the full-scale V* ([64,64], obs only)          -- must reproduce full_e3*.json value_floor
  vstar_t    : [64,64] with t/1000 (episode step) as an extra input -> removes time-limit aliasing
  vstar_wide : [256,256], obs only                        -> capacity
If V*(s,t) or the wide V* sits far below V_theta, the gap is fit/aliasing, not noise.
Usage: python scripts/vstar_time.py runs/full/<arm> <it> <seed...>   (writes runs/full_extra/vstar_time_<arm>_it<it>.json)"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402  (bootstrap first)
from src.common import info, metric, write_json  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src.configs.default import INSTRUMENTS as I  # noqa: E402
from src import eval as E  # noqa: E402
from src.eval import Ctx, advantages, collect, parse_run_name  # noqa: E402

E.ACTOR_BATCH = 50   # CPU placement only (numbers unchanged)
from src.model import Nets  # noqa: E402
from src.train import load_ckpt  # noqa: E402

run, it = sys.argv[1], int(sys.argv[2])
seeds_sel = [int(x) for x in sys.argv[3:]]
env_name, alg = parse_run_name(run)
ctx = Ctx(env_name, alg)
SEEDS = [0, 1, 2, 3, 4]
st0 = ctx.init_state(jax.random.PRNGKey(0))
like = {"ts": jax.tree_util.tree_map(lambda x: jnp.stack([x] * 5), st0),
        "norm": jax.tree_util.tree_map(lambda x: jnp.stack([x] * 5), D.norm_init(ctx.spec.obs_dim)), "it": jnp.int32(0)}
ck = load_ckpt(os.path.join(common.REPO, run, "ckpt", f"it{it:04d}"), like)
od, ad = ctx.spec.obs_dim, ctx.spec.act_dim
nets_t = Nets(od + 1, ad, ctx.cfg)
nets_w = Nets(od, ad, {**ctx.cfg, "value_hidden": (256, 256)})
EPL = float(ctx.cfg["episode_length"])
PAIRS, CHUNK = int(I["vstar_fit_pairs"]), int(I["rollout_chunk"])


def tfeat(tr):
    return ((tr.ep_len - 1.0) / EPL)[..., None]


@jax.jit
def fit_all(pi, norm, root):
    k = common.k_instr(root, it, common.I_VSTAR)
    k_data, k_fit = jax.random.split(k)
    tr = collect(ctx, pi, norm, jax.random.split(k_data, PAIRS // CHUNK), CHUNK)
    _, ret = advantages(ctx, tr, jnp.zeros_like(tr.rew), "zero")
    mask = A.complete_episode_mask(tr.not_done).reshape(-1)
    obs = tr.obs.reshape((-1, od))
    obs_t = jnp.concatenate([tr.obs, tfeat(tr)], -1).reshape((-1, od + 1))
    y = ret.reshape(-1)
    out = {}
    for name, nets, x in (("vstar", ctx.nets, obs), ("vstar_t", nets_t, obs_t), ("vstar_wide", nets_w, obs)):
        vs, mu, sd, inf = A.fit_value(nets, k_fit, x, y, mask, I["vstar_lr"], I["vstar_steps"], I["vstar_batch"])
        out[name] = (vs, mu, sd, inf["heldout_r2"], inf["best_step"])
    return out


@jax.jit
def score(pi, v, norm, root, fits):
    k = jax.random.fold_in(common.k_instr(root, it, common.I_HELDOUT), 7)
    tr = collect(ctx, pi, norm, jax.random.split(k, 10), int(I["heldout_pairs"]))
    vals = ctx.nets.v_apply(v, tr.obs)
    _, ret = advantages(ctx, tr, vals, "gae")
    nd = tr.not_done
    comp = A.complete_episode_mask(tr.not_done) * nd
    preds = {"vtheta": vals}
    for name, nets in (("vstar", ctx.nets), ("vstar_t", nets_t), ("vstar_wide", nets_w)):
        vs, mu, sd = fits[name][:3]
        x = jnp.concatenate([tr.obs, tfeat(tr)], -1) if name == "vstar_t" else tr.obs
        preds[name] = mu + sd * nets.v_apply(vs, x)
    res = {}
    for m_name, m in (("all", nd), ("complete", comp)):
        for p_name, pv in preds.items():
            res[f"{m_name}/mse/{p_name}"] = A.masked_mean((pv - ret) ** 2, m)
            res[f"{m_name}/mre/{p_name}"] = A.masked_mean(jnp.abs(pv - ret) / jnp.abs(ret), m)
    return res


out = {}
for si in seeds_sel:
    s = SEEDS[si]
    root = jax.random.PRNGKey(s)
    pi, v = ck["ts"].pi[si], ck["ts"].v[si]
    norm = jax.tree_util.tree_map(lambda x: x[si], ck["norm"])
    fits = fit_all(pi, norm, root)
    r = {k: float(x) for k, x in score(pi, v, norm, root, fits).items()}
    r.update({f"r2/{k}": float(f[3]) for k, f in fits.items()})
    r.update({f"best_step/{k}": int(f[4]) for k, f in fits.items()})
    out[f"seed{s}"] = r
    metric("vstar_time", arm=os.path.basename(run), it=it, seed=s, **{k: round(x, 5) for k, x in r.items() if "/mse/" in k or "r2" in k})
write_json(os.path.join(common.REPO, "runs", "full_extra", f"vstar_time_{os.path.basename(run.rstrip('/'))}_it{it}.json"),
           {"run": run, "it": it, **common.git_info(), **out})
info("done")
