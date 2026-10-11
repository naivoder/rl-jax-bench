"""Compute parity: steady-state throughput of the jitted train step per arm (no evaluation in the loop),
XLA's static cost estimate, and an analytic FLOP count of the dense layers (XLA's cost_analysis counts a
while/scan body once, so it under-counts the 500-step rollout and the 5-step MAML inner loop).

    python scripts/bench.py --iters 300 --out runs/full/bench.json
"""
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import check_device, metric  # noqa: E402
from src.train import _flops, batch_shape, build, load_config, make_train_step  # noqa: E402


def mlp_macs(sizes):
    return sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))


def analytic_flops(cfg, arm, spec):
    """Dense-layer FLOPs per iteration (2 FLOPs per MAC; backward = 2x forward)."""
    h = list(cfg["hidden"])
    T = int(cfg["n_envs"]) * spec.n_jobs  # decisions per iteration
    fa = 2 * mlp_macs([spec.obs_dim] + h + [spec.n_servers])
    fc = 2 * mlp_macs([spec.critic_dim] + h + [1])
    actor = T * fa * (1 + 3)  # rollout forward + loss forward/backward
    actor += T * fa * 3 / float(cfg.get("pg_var_every", 5))  # per-rollout PG vectors every k iterations
    if arm == "none":
        critic = 0.0
    elif arm in ("state", "multi", "multi_shuffled"):
        critic = T * fc * (1 + 3)  # baseline forward + loss forward/backward
    elif arm == "maml":
        J = int(cfg["meta_inner_steps"])
        critic = T * fc * (3 * J  # J inner steps on each half (each half = T/2 decisions; two halves)
                           + 1  # cross-fitted baselines
                           + 3  # first-order meta loss forward/backward at the adapted points
                           + 1)  # unadapted MSE diagnostic
    return {"actor": actor, "critic": critic, "total": actor + critic}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/configs/load_balance.yaml")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--arms", default="none,state,multi,maml")
    ap.add_argument("--out", default="runs/full/bench.json")
    a = ap.parse_args()
    check_device()
    cfg = load_config(a.config)
    out = {}
    for arm in a.arms.split(","):
        ts, consts, models = build(cfg, arm, 0)
        step = jax.jit(make_train_step(cfg, arm, models), donate_argnums=(0,))
        t0 = time.perf_counter()
        comp = step.lower(ts, consts).compile()
        tc = time.perf_counter() - t0
        ts, _ = comp(ts, consts)
        jax.block_until_ready(ts["it"])
        t0 = time.perf_counter()
        for _ in range(a.iters):
            ts, m = comp(ts, consts)
        jax.block_until_ready(ts["it"])
        dt = time.perf_counter() - t0
        an = analytic_flops(cfg, arm, models["spec"])
        out[arm] = {"compile_s": tc, "iters_per_s": a.iters / dt, "env_steps_per_s": a.iters * 8000 / dt,
                    "s_per_iter": dt / a.iters, "xla_cost_analysis_flops": _flops(comp),
                    "analytic_dense_flops_per_iter": an, "batch": batch_shape(cfg, arm)}
        metric("bench", arm=arm, iters_per_s=round(a.iters / dt, 2), compile_s=round(tc, 2),
               analytic_gflop=round(an["total"] / 1e9, 3), xla_flops=_flops(comp))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
