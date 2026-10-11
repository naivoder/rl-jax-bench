"""Full-scale EXTENSION drivers (labelled; the paper ran none of these): X1 theory-rate grids and X4 lower bound.

    uv run python -m src.sweep x1 --grid a --points 0,1 --seeds 0,1,2,3,4 --reps 1000 --out runs/x1
    uv run python -m src.sweep x4 --Ts 1000,3000 --seeds 0,1,2,3,4 --reps 1000 --out runs/x4

X1 (): K = 3 alternating instance (base (0.5, 0.3, 0.4), swapped (0.5, 0.9, 0.4)), Upsilon
evenly spaced breakpoints; D-UCB with Remark-3 gamma, SW-UCB with Remark-9 tau (both xi = 0.6), UCB-1 (xi = 1/2),
EXP3.S (S = Upsilon, alpha = 1/T). All seeds run in ONE vmapped call per policy (src.train.run_experiment_seeds;
identical numbers to sequential seeds). Points with T >= 3e5 scan in chunks of 1e5 rounds with an orbax
checkpoint at every boundary, so a killed job resumes bit-identically (see README).

X4 (hand derivation): K = 2 stationary game P (Ber 0.5, Ber 0.3) and the mixture P* in which arm 2 is
Ber(0.7) on ONE epoch j of length d, j uniform in 0..floor(T/d)-1 drawn from the environment branch
(u = uniform(fold_in(k_env_r, 0)); rounds t >= 1 use fold_in(k_env_r, t), so the draw is disjoint and shared by
every policy -> CRN).
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import time

import numpy as np
import yaml

from . import common
from .common import check_device, info, metric, write_json
from . import algorithm as alg
from . import train

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

F64 = jnp.float64
REPO = common.REPO


def load_cfg(path=None) -> dict:
    with open(path or os.path.join(REPO, "src", "configs", "extensions.yaml")) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- X1
def x1_points(cfg: dict, grid: str) -> list[tuple[int, int]]:
    g = cfg["x1"]["grids"][grid]
    if grid == "a":
        return [(int(T), int(g["upsilon"])) for T in g["T"]]
    if grid == "b":
        return [(int(g["T"]), int(u)) for u in g["upsilon"]]
    beta = float(g["beta"])  # grids c025 / c05: Upsilon_T = ceil(T^beta)
    return [(int(T), int(math.ceil(T ** beta - 1e-9))) for T in g["T"]]


def x1_tunings(T: int, U: int, x: dict) -> dict:
    """Remark 3 / Remark 9 tunings (B = 1) and the reference arms."""
    return {"ducb": {"xi": x["xi_theory"], "gamma": 1.0 - math.sqrt(U / T) / 4.0},
            "swucb": {"xi": x["xi_theory"], "tau": int(round(2.0 * math.sqrt(T * math.log(T) / U)))},
            "ucb1": {"xi": x["xi_ucb1"]},
            "exp3s": {"S": U, "alpha": 1.0 / T}}


def x1_log_every(T: int, chunk: int | None) -> int:
    le = max(1, T // 100)
    if chunk and chunk % le:
        le = math.gcd(le, chunk)
    return le


def run_x1_point(cfg: dict, grid: str, T: int, U: int, seeds, R: int, out_root: str, ckpt_root: str | None):
    x = cfg["x1"]
    chunk = int(x["chunk"]) if T >= int(x["chunk_from_T"]) else None
    le = x1_log_every(T, chunk)
    ecfg = {"name": f"x1_{grid}_T{T}_U{U}", "label": "X1 theory-rate grid (EXTENSION)", "exp_id": int(x["exp_id"]),
            "cfg_idx": T * 1000 + U, "log_every": le, "tie": "random",
            "env": {"kind": "alternating", "T": T, "upsilon": U, "base": x["base"], "swapped": x["swapped"]},
            "policies": x1_tunings(T, U, x)}
    out = os.path.join(out_root, grid, f"T{T}_U{U}")
    if all(os.path.exists(os.path.join(out, f"seed_{s}", "metrics.json")) for s in seeds):
        info("x1 point already done; skipped", grid=grid, T=T, U=U)
        return
    ck = os.path.join(ckpt_root, grid, f"T{T}_U{U}") if (ckpt_root and chunk) else None
    t0 = time.time()
    train.run_experiment_seeds(ecfg, seeds, R, out_dir=out, chunk_steps=chunk, ckpt_root=ck)
    metric("x1 point done", grid=grid, T=T, U=U, wall_s=time.time() - t0)


# ---------------------------------------------------------------- X4
@functools.lru_cache(maxsize=None)
def x4_runner(name: str, K: int, T: int, buf_len: int):
    step = alg.make_step(name, K, "random")

    def one_rep(k_env_r, k_pol_r, params, d, M, mu_base, mu_alt):
        u0 = jax.random.uniform(jax.random.fold_in(k_env_r, 0), (), dtype=F64)
        j = jnp.minimum(jnp.floor(u0 * M), jnp.maximum(M - 1, 0)).astype(jnp.int32)
        lo, hi = j * d + 1, (j + 1) * d

        def body(carry, t):
            st, ac = carry
            mu = jnp.where((M > 0) & (t >= lo) & (t <= hi), mu_alt, mu_base)
            st, ac, _, _ = step(st, ac, t, mu, jnp.asarray(False), k_env_r, k_pol_r, params)
            return (st, ac), None

        init = (alg.init_state(name, K, buf_len), alg.init_acc(K))
        (st, ac), _ = jax.lax.scan(body, init, jnp.arange(1, T + 1, dtype=jnp.int32))
        return ac

    vm = jax.vmap(one_rep, in_axes=(0, 0, None, None, None, None, None))

    def run(k_env, k_pol, params, d, M, mu_base, mu_alt):
        ac = vm(k_env, k_pol, params, d, M, mu_base, mu_alt)
        common.nan_guard({"regret": ac.regret, "bad": jnp.where(ac.bad, jnp.nan, 0.0)}, {"regret": ac.regret},
                         tag=100 + alg.POLICY_ID[name])
        return ac

    return jax.jit(run)


def x4_policies(T: int, x: dict) -> dict:
    U = int(x["upsilon_tuning"])
    return {"ucb1": {"xi": x["xi_ucb1"]},
            "ducb": {"xi": x["xi_theory"], "gamma": 1.0 - math.sqrt(U / T) / 4.0},
            "swucb": {"xi": x["xi_theory"], "tau": int(round(2.0 * math.sqrt(T * math.log(T) / U)))},
            "exp3s": {"S": U, "alpha": 1.0 / T}}


def run_x4_T(cfg: dict, T: int, seeds, R: int, out_root: str):
    x = cfg["x4"]
    K = 2
    mu_base = jnp.asarray(x["mu"], F64)
    mu_alt = jnp.asarray([x["mu"][0], x["nu"]], F64)
    alpha_kl = x["mu"][1] * math.log(x["mu"][1] / x["nu"]) + (1 - x["mu"][1]) * math.log((1 - x["mu"][1]) / (1 - x["nu"]))
    res = {"T": T, "seeds": list(seeds), "reps_per_seed": R, "alpha_kl": alpha_kl, "policies": {}}
    for name, pc in x4_policies(T, x).items():
        params, prec, buf_len = train.policy_params(name, pc, K, T)
        ks = [alg.rep_keys(s, int(x["exp_id"]), T, alg.POLICY_ID[name], R) for s in seeds]
        k_env = jnp.concatenate([k[0] for k in ks], 0)
        k_pol = jnp.concatenate([k[1] for k in ks], 0)
        run = x4_runner(name, K, T, buf_len)

        def go(d, M):
            tm = time.time()
            ac = run(k_env, k_pol, params, jnp.asarray(d, jnp.int32), jnp.asarray(M, jnp.int32), mu_base, mu_alt)
            reg = np.asarray(ac.regret).reshape(len(seeds), R)
            n2 = np.asarray(ac.ntilde)[:, 1].reshape(len(seeds), R)
            return reg, n2, time.time() - tm

        reg0, n20, w0 = go(1, 0)  # stationary game P (M = 0 -> no epoch)
        EN = float(n20.mean())
        dstar = int(min(T, max(1, round(16.0 * T / (9.0 * alpha_kl * EN)))))
        dgrid = sorted({max(1, int(round(T * 2.0 ** -k))) for k in range(1, 11)} | {dstar})
        mix = {}
        for d in dgrid:
            M = T // d
            reg, _, w = go(d, M)
            mix[str(d)] = {"M": M, "seed_means": reg.mean(1).tolist(), "mean": float(reg.mean()), "wall_s": w}
        res["policies"][name] = {"params": prec, "stationary": {"seed_means": reg0.mean(1).tolist(), "mean": float(reg0.mean()),
                                                                "EN2_seed_means": n20.mean(1).tolist(), "EN2": EN, "wall_s": w0},
                                 "dstar": dstar, "mixture": mix}
        metric("x4 policy done", T=T, policy=name, ER=float(reg0.mean()), EN2=EN, dstar=dstar,
               EstarR_dstar=mix[str(dstar)]["mean"], EstarR_max=max(v["mean"] for v in mix.values()))
    res["device"] = str(jax.devices()[0])
    res["git_sha"], res["git_dirty"] = common.git_sha(), common.git_dirty()
    write_json(os.path.join(out_root, f"T{T}.json"), res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["x1", "x4"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--grid", default="a")
    ap.add_argument("--points", default=None, help="comma list of point indices in the grid (default all)")
    ap.add_argument("--Ts", default=None)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args(argv)
    check_device()
    cfg = load_cfg(a.config)
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.what == "x1":
        pts = x1_points(cfg, a.grid)
        idx = [int(i) for i in a.points.split(",")] if a.points else range(len(pts))
        for i in idx:
            run_x1_point(cfg, a.grid, *pts[i], seeds, a.reps, a.out, a.ckpt)
    else:
        for T in [int(t) for t in a.Ts.split(",")]:
            run_x4_T(cfg, T, seeds, a.reps, a.out)


if __name__ == "__main__":
    main()
