"""toy-check extras that need training (GPU): the auto-encoder floor (README, toy C3 / P-AE part 1).

  C3-lin: a rank-64 LINEAR auto-encoder on x ~ U(-1,1)^{17x7} trained by Adam must reach (1/3)(1 - 64/119) +- 2 %
  P-AE: the shipped AE (BiGRU / BiGRU+MHSA extractor + Dense 256 ReLU decoder, 2e4 Adam steps, batch 256, lr 1e-3)
           must end within +-10 % of the same floor (README P-AE)
Writes runs/toys/toys.json.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.linen as nn  # noqa: E402
import optax  # noqa: E402
from dataclasses import replace  # noqa: E402

from src.configs.config import Config  # noqa: E402
from src import train as TR  # noqa: E402

FLOOR = (1.0 / 3.0) * (1.0 - 64.0 / 119.0)


def linear_ae(key, steps=30_000, batch=1024, lr=3e-3):
    enc, dec = nn.Dense(64, use_bias=False), nn.Dense(119, use_bias=False)
    k1, k2 = jax.random.split(key)
    p = dict(e=enc.init(k1, jnp.zeros((1, 119))), d=dec.init(k2, jnp.zeros((1, 64))))
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    st = opt.init(p)

    def loss(p, x):
        return jnp.mean((dec.apply(p["d"], enc.apply(p["e"], x)) - x) ** 2)

    def body(c, i):
        p, st = c
        x = jax.random.uniform(jax.random.fold_in(key, i), (batch, 119), jnp.float32, -1, 1)
        l, g = jax.value_and_grad(loss)(p, x)
        u, st = opt.update(g, st, p)
        return (optax.apply_updates(p, u), st), l
    (p, _), ls = jax.lax.scan(body, (p, st), jnp.arange(steps))
    xe = jax.random.uniform(jax.random.fold_in(key, 10 ** 8), (65536, 119), jnp.float32, -1, 1)
    return float(loss(p, xe))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pae-only", action="store_true", help="skip the linear AE (its result does not depend on the "
                    "network init style); used to re-check P-AE after the official-code phase's init_style change")
    ap.add_argument("--init-style", default=None, help="override NetConfig.init_style (torch | flax)")
    ap.add_argument("--out", default=os.path.join(C.REPO, "runs", "toys", "toys.json"))
    ap.add_argument("--encoders", default="bigru,mhsa")
    a = ap.parse_args()
    C.check_device()
    out = dict(floor=FLOOR)
    tm = C.Timer()
    if not a.pae_only:
        lin = [linear_ae(jax.random.PRNGKey(s)) for s in range(3)]
        out["C3_linear_mse"] = lin
        out["C3_pass"] = all(abs(v - FLOOR) / FLOOR < 0.02 for v in lin)
        C.metric("C3 linear AE", mse=lin, floor=FLOOR, passed=out["C3_pass"], sec=round(tm.lap(), 1))
    cfg = Config()
    if a.init_style:
        cfg = replace(cfg, net=replace(cfg.net, init_style=a.init_style))
    out["init_style"] = cfg.net.init_style
    for enc in a.encoders.split(","):
        c = replace(cfg, net=replace(cfg.net, encoder=enc), pre=replace(cfg.pre, bc_episodes=1, bc_epochs=1))
        net, pre = TR.make_pretrain(c)
        res = []
        for seed in range(3):
            k = TR.lane_key(c, enc, seed)
            from src.model import TrackEncoder, dummy_obs
            p0 = net.init(jax.random.fold_in(k, 0), dummy_obs(c.env, (1,)))
            ae = jax.jit(lambda kk, pp: _ae_only(c, kk, pp))
            final, curve = ae(jax.random.fold_in(k, 1), {"params": p0["params"]["track_enc"]})
            res.append(dict(final=float(final), curve=[float(x) for x in curve]))
        out[f"PAE_{enc}"] = res
        ok = all(abs(r["final"] - FLOOR) / FLOOR < 0.10 for r in res)
        out[f"PAE_{enc}_pass"] = ok
        C.metric("P-AE", encoder=enc, finals=[r["final"] for r in res], floor=FLOOR, within10pct=ok,
                 sec=round(tm.lap(), 1))
    C.write_json(a.out, out)


def _ae_only(cfg, k_ae, enc_params):
    """Same AE routine as train.make_pretrain (re-entered here to return its curve and held-out loss)."""
    net, pretrain_lane = TR.make_pretrain(cfg)
    import src.train as T
    # call the inner `ae` through a lane with use_ae = 1 would also run BC; replicate its body instead
    from src.model import TrackEncoder, AEDecoder
    from src import algorithm as alg
    K = cfg.env.k_max
    enc, dec = TrackEncoder(cfg.net), AEDecoder(cfg.pre.ae_dec_hidden, K * 7, init_style=cfg.net.init_style)
    dparams = dec.init(jax.random.fold_in(k_ae, 0), jnp.zeros((1, 64), jnp.float32))
    p = dict(enc=enc_params, dec=dparams)
    opt = optax.adam(cfg.pre.ae_lr)
    st = opt.init(p)

    def loss_fn(p, x):
        z = enc.apply(p["enc"], x, jnp.ones(x.shape[:-1], bool))
        return alg.ae_loss(dec.apply(p["dec"], z).reshape(x.shape), x)

    def body(c, i):
        p, st = c
        x = jax.random.uniform(jax.random.fold_in(k_ae, i + 1), (cfg.pre.ae_batch, K, 7), jnp.float32, -1.0, 1.0)
        l, g = jax.value_and_grad(loss_fn)(p, x)
        u, st = opt.update(g, st, p)
        return (optax.apply_updates(p, u), st), l
    (p, _), ls = jax.lax.scan(body, (p, st), jnp.arange(cfg.pre.ae_steps))
    xe = jax.random.uniform(jax.random.fold_in(k_ae, 10 ** 7), (4096, K, 7), jnp.float32, -1.0, 1.0)
    return loss_fn(p, xe), ls.reshape(20, -1).mean(1)


if __name__ == "__main__":
    main()
