"""Gradient-quality instrument on the OFFICIAL agent's exact checkpoints (official-code check, standing rule 2).

official_grad_ref.py trains the unmodified official Trainer (gymnasium MuJoCo v4) and, at checkpoints {0,15,45},
exports the policy/value weights and the frozen normaliser state, then measures gradient quality with the official
primitives. This script loads those SAME checkpoints into our JAX nets / NormState and runs OUR instrument
(src.eval.gradient_block + algorithm.cos_stats: small-scale sizes, m=10, reference 1e5, chunk 2000) on our MJX proxy.
Identical parameters, so any difference is env physics  + sampling design ( chunking, nested
prefixes) + Monte-Carlo noise, NOT a difference in training trajectory.

Usage: python scripts/official_ckpt_grad.py [--device cpu|gpu]   -> runs/oracle/official_ckpt_grad.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402  (bootstrap first)
from src.common import info, metric, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src.eval import Ctx, gradient_block  # noqa: E402

OFF = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), "grad_ref_ck")
SIZES = (100, 1000, 2000, 10000)


def flat_from(nets, d, prefix, policy=True):
    tree = nets.pi_unravel(nets.pi_flat0) if policy else nets.v_unravel(nets.v_flat0)
    sub = tree["params"]["mean" if policy else "v"]
    for i in range(3):
        sub[f"Dense_{i}"]["kernel"] = jnp.asarray(d[f"{prefix}W{i}"], jnp.float32)
        sub[f"Dense_{i}"]["bias"] = jnp.asarray(d[f"{prefix}b{i}"], jnp.float32)
    if policy:
        tree["params"]["log_std"] = jnp.asarray(d[f"{prefix}log_std"], jnp.float32)
    return ravel_pytree(tree)[0]


def norm_from(d, obs_dim):
    obs = D.RunStat(jnp.float32(d["obs_n"]), jnp.asarray(d["obs_M"], jnp.float32), jnp.asarray(d["obs_S"], jnp.float32))
    rew = (D.RunStat(jnp.float32(d["rew_n"]), jnp.asarray(d["rew_M"], jnp.float32).reshape(()),
                     jnp.asarray(d["rew_S"], jnp.float32).reshape(())) if "rew_n" in d else D.runstat_init(()))
    return D.NormState(obs, rew)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    args = ap.parse_args()
    common.check_device()
    dev = jax.devices("cpu")[0] if args.device == "cpu" else jax.devices()[0]
    out = {}
    ctxs = {}
    with jax.default_device(dev):
        for f in sorted(glob.glob(os.path.join(OFF, "*_s[0-9].npz"))):
            name = os.path.basename(f)[:-4]
            env, alg, s = name.split("_")
            seed = int(s[1:])
            off = json.load(open(f[:-4] + ".json"))
            arr = dict(np.load(f))
            if (env, alg) not in ctxs:
                ctx = Ctx(env, alg)
                ctxs[(env, alg)] = (ctx, jax.jit(lambda p, v, nm, root, it, ctx=ctx: gradient_block(
                    ctx, p, v, nm, root, it, SIZES, 10, 2000, 100000)))
            ctx, jg = ctxs[(env, alg)]
            for it in (0, 15, 45):
                d = {k[len(f"it{it}_"):]: v for k, v in arr.items() if k.startswith(f"it{it}_")}
                pi, v = flat_from(ctx.nets, d, "pi_"), flat_from(ctx.nets, d, "v_", False)
                nm = norm_from(d, ctx.spec.obs_dim)
                root = jax.random.PRNGKey(50_000 + seed)
                o = jg(pi, v, nm, root, jnp.int32(it))
                row = {}
                for n in SIZES:
                    st = A.cos_stats(np.asarray(o["G"]["gae"][n]), np.asarray(o["g_ref"]), n_boot=10)
                    oc = off["cells"][str(it)][str(n)]
                    row[n] = dict(ours_pw=st["pairwise"], ours_tt=st["to_true"], off_pw=oc["pairwise"],
                                  off_tt=oc["to_true"])
                out.setdefault(f"{env}_{alg}", {}).setdefault(str(seed), {})[it] = row
                metric("ckpt_grad", arm=f"{env}_{alg}", seed=seed, it=it, **{
                    f"n{n}": f"pw {r['ours_pw']:.3f}/{r['off_pw']:.3f} tt {r['ours_tt']:.3f}/{r['off_tt']:.3f}"
                    for n, r in row.items()})
    # paired summary over seeds: ours (MJX, our instrument) minus official (gymnasium, official primitives)
    summ = {}
    for arm, by_seed in out.items():
        for it in (0, 15, 45):
            for n in SIZES:
                for k in ("pw", "tt"):
                    a = np.array([by_seed[s][it][n][f"ours_{k}"] for s in sorted(by_seed)])
                    b = np.array([by_seed[s][it][n][f"off_{k}"] for s in sorted(by_seed)])
                    dlt = a - b
                    summ.setdefault(arm, {}).setdefault(str(it), {}).setdefault(str(n), {})[k] = dict(
                        ours=[float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))],
                        official=[float(b.mean()), float(b.std(ddof=1) / np.sqrt(len(b)))],
                        diff=[float(dlt.mean()), float(dlt.std(ddof=1) / np.sqrt(len(dlt)))])
    write_json(os.path.join(common.REPO, "runs", "oracle", "official_ckpt_grad.json"),
               {"per_seed": {a: {s: {str(i): {str(n): r for n, r in row.items()} for i, row in v.items()}
                                 for s, v in bs.items()} for a, bs in out.items()},
                "summary": summ, **common.git_info()})
    for arm, by_it in summ.items():
        for it, by_n in by_it.items():
            print(arm, it, " ".join(f"n={n}: tt ours {c['tt']['ours'][0]:.3f} off {c['tt']['official'][0]:.3f} "
                                    f"d {c['tt']['diff'][0]:+.3f}±{c['tt']['diff'][1]:.3f}" for n, c in by_n.items()))


if __name__ == "__main__":
    main()
