"""Sensitivity runs requested in review (pre-declared before running). Nothing here replaces a primary result.

    uv run python -m scripts.sensitivity bins      # S-BIN: Return HCA, 5 bin layouts x 2 update orders
    uv run python -m scripts.sensitivity eps0      # S-EPS0: E6/E7 at eps = 0, PG re-tuned on seed 999
    uv run python -m scripts.sensitivity slide     # S-SLIDE: E3 with sliding-window 3-step returns
    uv run python -m scripts.sensitivity f3rv      # S-F3R-V: E2 PG with an online-learned V(x0) baseline
    uv run python -m scripts.sensitivity alpha     # D-ALPHA: oracle-h_z Return HCA on E8 eps=0 vs policy lr
    uv run python -m scripts.sensitivity x0_alias  # exact x0 update of bootstrapped State HCA vs aliasing
    uv run python scripts/sensitivity_report.py    # tables and tests (no jax)

Seed 0, 100 runs, the primary run keys (paired by run index). Outputs go to runs/sensitivity/.
"""
from __future__ import annotations

import argparse
import os

import src  # noqa: F401  (GPU bootstrap + x64)
import jax
import jax.numpy as jnp
import numpy as np

from src import data as D
from src.algorithm import mc_returns, return_hca_grads
from src.eval import start_value
from src.model import init_params, policy
from src.train import apply_update, load_config, make_env, make_fig3r_runner, method_lrs, run_keys, settings
from scripts.full import TUNING_SEED, run_arm, summarise_curve, tuned_lrs
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, write_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "sensitivity")
DE_EXPS = ("E3", "E4", "E5")


def layouts(exp: str) -> dict:
    """Bin layouts of S-BIN as runner-kwarg overrides (see the pre-declaration)."""
    out = {"A": {}, "W1": {"bin_widen": 1.0}, "W0": {"bin_widen": 0.0}, "RUN": {"bin_mode": "running"}}
    if exp in DE_EXPS:
        out["OUT"] = {"centres_override": (-1.0, 0.0, 1.0)}
    else:
        out["OUT"] = {"centres_override": (1.0, 2.0), "n_bins": 2}
    return out


def pack(c, exp):
    s = summarise_curve(c, exp)
    return {k: v for k, v in s.items() if k not in ("std_curve",)}


def cmd_bins(cfg, n_runs, seed):
    out = {"seed": seed, "n_runs": n_runs, "cells": []}
    lrs = np.asarray(method_lrs(cfg, "return_hca"))
    for exp in ("E3", "E4", "E5", "E6", "E7", "E8"):
        for si, kw in enumerate(settings(cfg["experiments"][exp])):
            for lay, rk in layouts(exp).items():
                for order in ("official", "alg"):
                    tm = Timer()
                    c = run_arm(cfg, exp, kw, "return_hca", seed, n_runs, lrs, si, order=order, **rk)
                    s = pack(c, exp)
                    out["cells"].append({"exp": exp, "si": si, "setting": kw, "layout": lay, "order": order,
                                         "runner_kwargs": {k: v for k, v in rk.items()}, **s, "wall_s": tm.lap()})
                    metric("S-BIN", exp=exp, si=si, layout=lay, order=order, final=round(float(s["final"].mean()), 4),
                           avg=round(float(s["avg"].mean()), 4))
    write_json(os.path.join(OUT, "bins.json"), out)


def tune_pg(cfg, exp, kw, si, n_runs, **rk):
    scores = {}
    for lr in cfg["pg"]["lr_grid"]:
        c = run_arm(cfg, exp, kw, "pg", TUNING_SEED, n_runs, method_lrs(cfg, "pg", lr), si, **rk)
        scores[lr] = float(c.mean())
    return max(scores, key=scores.get), scores


def cmd_eps0(cfg, n_runs, seed):
    out = {"seed": seed, "n_runs": n_runs, "rows": []}
    for exp in ("E6", "E7"):
        kw = dict(settings(cfg["experiments"][exp])[0], eps=0.0)
        best, scores = tune_pg(cfg, exp, kw, 0, n_runs)
        metric("S-EPS0 tune", exp=exp, best_lr=best, **{f"lr{k}": round(v, 4) for k, v in scores.items()})
        arms = [("state_hca", method_lrs(cfg, "state_hca")), ("return_hca", method_lrs(cfg, "return_hca"))]
        arms += [(f"pg_lr{lr}", method_lrs(cfg, "pg", lr)) for lr in cfg["pg"]["lr_grid"]]
        rec = {"exp": exp, "setting": kw, "pg_tuned_lr": best, "pg_tune_scores": scores, "arms": {}}
        for name, lrs in arms:
            m = "pg" if name.startswith("pg") else name
            rec["arms"][name] = pack(run_arm(cfg, exp, kw, m, seed, n_runs, lrs, 0), exp)
        rec["arms"]["pg"] = rec["arms"][f"pg_lr{best}"]
        for a in ("state_hca", "return_hca", "pg"):
            metric("S-EPS0", exp=exp, arm=a, final=round(float(rec["arms"][a]["final"].mean()), 4))
        out["rows"].append(rec)
    write_json(os.path.join(OUT, "eps0.json"), out)


def cmd_slide(cfg, n_runs, seed):
    exp, si = "E3", 0
    kw = settings(cfg["experiments"][exp])[si]
    best, scores = tune_pg(cfg, exp, kw, si, n_runs, seg_mode="sliding")
    metric("S-SLIDE tune", best_lr=best, **{f"lr{k}": round(v, 4) for k, v in scores.items()})
    rec = {"exp": exp, "setting": kw, "pg_tuned_lr": best, "pg_tune_scores": scores, "arms": {}}
    for order in ("official", "alg"):
        rec["arms"][f"state_hca_{order}"] = pack(run_arm(cfg, exp, kw, "state_hca", seed, n_runs,
                                                         method_lrs(cfg, "state_hca"), si, seg_mode="sliding",
                                                         order=order), exp)
    for lr in cfg["pg"]["lr_grid"]:
        rec["arms"][f"pg_lr{lr}"] = pack(run_arm(cfg, exp, kw, "pg", seed, n_runs, method_lrs(cfg, "pg", lr), si,
                                                 seg_mode="sliding"), exp)
    rec["arms"]["pg"] = rec["arms"][f"pg_lr{best}"]
    for a, v in rec["arms"].items():
        metric("S-SLIDE", arm=a, ep100=round(float(v["ep100"].mean()), 4), final=round(float(v["final"].mean()), 4))
    write_json(os.path.join(OUT, "slide.json"), {"seed": seed, "n_runs": n_runs, **rec})


def cmd_f3rv(cfg, n_runs, seed):
    ec = cfg["experiments"]["E2"]
    env = make_env(ec, ec["env_kwargs"])
    out = {"seed": seed, "n_runs": n_runs, "points": []}
    for si, p in enumerate(ec["p_long"]):
        keys = run_keys(seed, "E2", si, n_runs)
        rec = {"p_long": p}
        for vlr in (0.3, 0.1):
            run, oracle = make_fig3r_runner(env, "pg", p, rollouts=ec["rollouts"], n_bins=ec["n_bins"],
                                            bin_widen=cfg["hca"]["bin_widen"], gamma=cfg["gamma"], immediate="rhat",
                                            hca_lrs=method_lrs(cfg, "state_hca"), pg_baseline="online_v", pg_v_lr=vlr)
            rec[f"pg_online_v_lr{vlr}"] = np.asarray(jax.block_until_ready(run(keys))["online"])
            rec["oracle"] = oracle
        metric("S-F3R-V", p=round(p, 3), oracle=round(rec["oracle"], 4),
               **{k: round(float(np.mean(v)), 4) for k, v in rec.items() if k.startswith("pg")})
        out["points"].append(rec)
    write_json(os.path.join(OUT, "f3rv.json"), out)


def cmd_alpha(cfg, n_runs, seed):
    """Same construction as scripts.full.cmd_oracle_hz (exact continuous h_z, no bins), E8 eps = 0 only, with the
    policy lr varied. alpha = 0.3 must reproduce runs/full/oracle_hz.json."""
    exp, si = "E8", 0
    ec = cfg["experiments"][exp]
    kw = settings(ec)[si]
    env = make_env(ec, kw)
    ea = D.env_arrays(env)
    eps, sig = kw["eps"], kw["sigma"]

    def hz_exact(pi0, z):
        n1 = jnp.exp(-0.5 * ((z - 1.0) / sig) ** 2)
        n2 = jnp.exp(-0.5 * ((z - 2.0) / sig) ** 2)
        p = jnp.array([(1 - eps) * n1 + eps * n2, eps * n1 + (1 - eps) * n2]) * pi0
        return p / jnp.sum(p)

    def make(lr):
        lrs = jnp.asarray(method_lrs(cfg, "return_hca")).at[0].set(lr)

        def one_run(key):
            params = init_params(env.n_obs, ec["n_bins"])
            v0 = start_value(ea, params.theta, env.s0, env.L)

            def body(p, e):
                pi = policy(p.theta)
                ep = D.sample_episode(ea, env.s0, env.L, lambda o: pi[o], jax.random.fold_in(key, e))
                Z = mc_returns(ep, 1.0)
                hz = jnp.stack([hz_exact(pi[ep.obs[t]], Z[t]) for t in range(env.L)])
                g = return_hca_grads(p, ep, 1.0, jnp.zeros(ec["n_bins"]), hz_override=hz)
                p = apply_update(p, g, lrs)
                return p, start_value(ea, p.theta, env.s0, env.L)

            _, vals = jax.lax.scan(body, params, jnp.arange(ec["episodes"]))
            return jnp.concatenate([v0[None], vals])
        return jax.jit(jax.vmap(one_run))

    out = {"seed": seed, "n_runs": n_runs, "rows": []}
    for lr in (0.3, 0.6, 1.0, 2.0):
        c = np.asarray(make(lr)(run_keys(seed, exp, si, n_runs)))
        s = pack(c, exp)
        out["rows"].append({"alpha": lr, **s})
        metric("D-ALPHA", alpha=lr, avg=round(float(s["avg"].mean()), 4), final=round(float(s["final"].mean()), 4))
    write_json(os.path.join(OUT, "alpha.json"), out)


def cmd_x0_alias(cfg, n_runs, seed):
    """Exact expected x0 update of the bootstrapped State-HCA estimator on Delayed Effect n = 5 (E3), with the
    exact pair hindsight and the observation-level exact V, under both aliasing modes and both segment semantics,
    plus the MC estimator; compared with the true gradient (enumeration, no sampling; same construction as G1.5a)."""
    from scripts.exact_checks import A, state_grad_case
    from src.eval import dp_values, enumerate_trajectories
    out = {"rows": []}
    for aliasing in ("per_position", "shared"):
        env = D.delayed_effect(5, 0.0, aliasing)
        pi = np.full((env.n_obs, A), 0.5)
        pi[0] = [0.6, 0.4]
        ea = D.env_arrays(env)
        with np.errstate(divide="ignore"):
            th = jnp.asarray(np.log(pi))
        Vs = np.asarray(dp_values(ea, th, env.L))
        occ = np.zeros(env.P.shape[0])
        for p, xs, _ in enumerate_trajectories(env, pi):
            for st in xs:
                occ[st] += p
        Vo, w = np.zeros(env.n_obs), np.zeros(env.n_obs)
        for st in range(env.P.shape[0]):
            Vo[env.obs[st]] += occ[st] * Vs[st]
            w[env.obs[st]] += occ[st]
        Vo = np.where(w > 0, Vo / np.maximum(w, 1e-300), 0.0)
        for seg in (None, 3):
            for mode in (("segment",) if seg is None else ("segment", "sliding")):
                from src.eval import exact_pair_hindsight, true_policy_gradient, trajectories_to_batch, to_jnp_episode
                from src.algorithm import state_hca_grads
                hp, _ = exact_pair_hindsight(env, pi, seg, mode)
                psi = np.zeros((env.n_obs, env.n_obs, A))
                for (ox, oy), v in hp.items():
                    psi[ox, oy] = np.log(np.maximum(v, 1e-300))
                prm = init_params(env.n_obs, 2)._replace(theta=th, psi=jnp.asarray(psi), V=jnp.asarray(Vo))
                probs, batch = trajectories_to_batch(env, enumerate_trajectories(env, pi))
                g = jax.vmap(lambda e: state_hca_grads(prm, e, seg, 1.0, "official", seg_mode=mode).theta)(
                    to_jnp_episode(batch))
                g_exp = np.tensordot(probs, np.asarray(g), axes=1)[0]
                g_true = np.asarray(true_policy_gradient(ea, th, env.s0, env.L))[0]
                h_x0_chain = {k: v.tolist() for k, v in hp.items() if k[0] == 0}
                row = {"aliasing": aliasing, "seg_len": seg, "seg_mode": mode, "grad_exp_x0": g_exp.tolist(),
                       "grad_true_x0": g_true.tolist(), "h_x0_pairs": {str(k): v for k, v in h_x0_chain.items()}}
                out["rows"].append(row)
                metric("x0_alias", aliasing=aliasing, seg=seg, mode=mode, exp_x0=np.round(g_exp, 6).tolist(),
                       true_x0=np.round(g_true, 6).tolist())
    write_json(os.path.join(OUT, "x0_alias.json"), out)


def cmd_vlr(cfg, n_runs, seed):
    """Value lr (descriptive, added after the pre-declaration): E3 State HCA (3-step) with the value-table lr varied; the
    value lr (0.3, A-LR) has no paper support and V only enters State HCA through the bootstrap."""
    exp, si = "E3", 0
    kw = settings(cfg["experiments"][exp])[si]
    out = {"seed": seed, "n_runs": n_runs, "rows": []}
    for vlr in (0.1, 0.2, 0.3, 0.4):
        lrs = np.asarray(method_lrs(cfg, "state_hca")).copy()
        lrs[2] = vlr
        s = pack(run_arm(cfg, exp, kw, "state_hca", seed, n_runs, lrs, si), exp)
        out["rows"].append({"value_lr": vlr, **s})
        metric("vlr", value_lr=vlr, ep100=round(float(s["ep100"].mean()), 4), final=round(float(s["final"].mean()), 4))
    write_json(os.path.join(OUT, "vlr.json"), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bins", "eps0", "slide", "f3rv", "alpha", "x0_alias", "vlr"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_runs", type=int, default=100)
    args = ap.parse_args()
    cfg = load_config()
    check_device()
    os.makedirs(OUT, exist_ok=True)
    t = Timer()
    {"bins": cmd_bins, "eps0": cmd_eps0, "slide": cmd_slide, "f3rv": cmd_f3rv, "alpha": cmd_alpha, "x0_alias": cmd_x0_alias, "vlr": cmd_vlr}[args.cmd](
        cfg, args.n_runs, args.seed)
    info("sensitivity done", cmd=args.cmd, wall_s=t.lap())


if __name__ == "__main__":
    main()
