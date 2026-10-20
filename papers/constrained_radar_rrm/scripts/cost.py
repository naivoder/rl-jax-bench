"""E5 latency (A-43) and compute parity: per-decision wall time (median of 1000 jitted single-decision calls, CPU)
and XLA cost_analysis FLOPs per decision for every policy; FLOPs per training slot for both learners.

A-43: the paper reports CDRL 1.9 ms vs OA 5.4 ms per decision ("identical hardware"); only the RATIO is compared
(descriptive). Run with JAX_PLATFORMS=cpu.  Usage: python scripts/full_cost.py --out runs/full/cost.json
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import EnvCfg, info, load_calibration, load_yaml, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as alg, data, env as E, eval as ev, train  # noqa: E402


def flops(f, *args):
    try:
        ca = jax.jit(f).lower(*args).compile().cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        return float(ca.get("flops", float("nan")))
    except Exception as e:  # noqa: BLE001
        return f"n/a ({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/full/cost.json")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    common.check_device()
    y = load_yaml()
    cal = load_calibration()
    cfg = EnvCfg.from_yaml(y)
    lut = ev.make_lut(cfg)
    consts = dict(snr0=jnp.float64(cal["snr0"]), s_ref=jnp.float64(cal["s_ref"]), delta=jnp.float64(0.9))
    scen = data.replica_case1(cfg)
    # a 3-track decision state: run EA for 2500 slots on the replica (Targets 2, 3, 1 present)
    pol_ea = ev.make_policy("ea", cfg, y)
    es = E.reset(cfg)
    step = jax.jit(lambda es, k: E.post(cfg, consts, lut, scen, es, E.pre(cfg, scen, es),
                                        pol_ea(None, consts, E.pre(cfg, scen, es), es), k)[0])
    for t in range(2500):
        es = step(es, jax.random.fold_in(jax.random.PRNGKey(0), t))
    view = E.pre(cfg, scen, es)
    info("decision state", n_trk=int(es.act.sum()))
    actor, critic, _, _ = alg.make_ddpg(y, cfg.K, cfg.T0)
    pa = actor.init(jax.random.PRNGKey(1), jnp.zeros((1, 11), jnp.float32))
    qcfg = alg.make_dql_cfg(y, cfg.K, 64)
    from rl_jax_bench import drqn
    pq = drqn.init(jax.random.PRNGKey(2), qcfg).params
    pols = {k: (ev.make_policy(k, cfg, y), None) for k in ("ea", "da", "oa", "oa_myopic")}
    pols["ddpg"] = (ev.make_policy("ddpg", cfg, y, actor=actor), pa)
    pols["dql"] = (ev.make_policy("dql", cfg, y, dql_cfg=qcfg), pq)
    res = dict(device=str(jax.devices()[0]), n_calls=a.n, n_trk=int(es.act.sum()), policies={})
    for k, (pol, p) in pols.items():
        f = jax.jit(lambda p_, v_, e_: pol(p_, consts, v_, e_))
        jax.block_until_ready(f(p, view, es))
        ts = []
        for _ in range(a.n):
            t0 = time.perf_counter()
            jax.block_until_ready(f(p, view, es))
            ts.append(time.perf_counter() - t0)
        res["policies"][k] = dict(median_ms=1e3 * float(np.median(ts)), p90_ms=1e3 * float(np.percentile(ts, 90)),
                                  flops=flops(lambda p_, v_, e_: pol(p_, consts, v_, e_), p, view, es))
        info("latency", policy=k, **res["policies"][k])
    # training cost per slot (one lane, one update) from the jitted lane_step
    lane_c = dict(consts, alpha=jnp.float64(5000.0), lam0=jnp.float64(5000.0), rs=jnp.float64(1e-4),
                  cs=jnp.float64(1e-4), ls=jnp.float64(1e-4))
    res["train_flops_per_slot"] = {}
    for algo in ("ddpg", "dql"):
        y2 = load_yaml()
        y2["ddpg"]["learning_starts"] = y2["dql"]["learning_starts"] = 0
        init_lane, lane_step, _, _ = train.build(algo, y2, cfg, lane_c, 300000, 3000)
        ln = init_lane(jax.random.PRNGKey(3), lane_c)
        res["train_flops_per_slot"][algo] = flops(lane_step, ln, jnp.int32(5))
    r = res["policies"]
    res["ratio_oa_over_ddpg_ms"] = r["oa"]["median_ms"] / r["ddpg"]["median_ms"]
    res["paper_ratio"] = 5.4 / 1.9
    write_json(a.out, res)
    info("cost done", ratio=res["ratio_oa_over_ddpg_ms"], paper=res["paper_ratio"])


if __name__ == "__main__":
    main()
