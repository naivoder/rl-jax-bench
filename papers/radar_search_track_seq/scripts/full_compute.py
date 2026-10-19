"""Compute parity (descriptive): parameter count and XLA-counted FLOPs per policy forward (batch 1) and per PPO
minibatch gradient (batch 64) for each encoder, plus the env step. CPU only.

    JAX_PLATFORMS=cpu python scripts/full_compute.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import Config  # noqa: E402
from src.model import ActorCritic, dummy_obs  # noqa: E402
from src import algorithm as alg, data  # noqa: E402


def flops(fn, *args):
    ca = jax.jit(fn).lower(*args).compile().cost_analysis()
    ca = ca[0] if isinstance(ca, (list, tuple)) else ca
    return float(ca.get("flops", float("nan")))


def main():
    cfg = Config()
    ecfg = replace(cfg.env, gospa_metrics=False)
    out = {}
    for enc in ("flat", "bigru", "mhsa"):
        net = ActorCritic(replace(cfg.net, encoder=enc), ecfg)
        o1 = dummy_obs(ecfg, (1,))
        p = net.init(jax.random.PRNGKey(0), o1)
        n = int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))
        o64 = dummy_obs(ecfg, (64,))
        a = jnp.zeros((64, 2), jnp.int32)
        z = jnp.zeros((64,), jnp.float32)

        def loss(pp):
            la, le, v = net.apply(pp, o64)
            pg, vl, el, _ = alg.ppo_loss_terms(la, le, v, a, z, z, z, 0.2)
            return pg + 0.5 * vl
        f_fwd = flops(lambda pp: net.apply(pp, o1), p)
        f_grad = flops(jax.grad(loss), p)
        out[enc] = dict(params=n, flops_forward_b1=f_fwd, flops_grad_b64=f_grad)
        C.metric("compute", encoder=enc, params=n, flops_fwd=f_fwd, flops_grad64=f_grad)
    s = data.reset(jax.random.PRNGKey(0), ecfg)
    out["env_step"] = flops(lambda s: data.step(s, jnp.array([9, 9], jnp.int32), ecfg)[0].tr.P, s)
    C.metric("compute", env_step_flops=out["env_step"])
    C.write_json(os.path.join(C.REPO, "runs", "full", "compute.json"), out)


if __name__ == "__main__":
    main()
