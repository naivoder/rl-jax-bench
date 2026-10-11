"""Training loops: one jitted, vmapped function per (experiment setting, method) that runs ``n_runs``
independent learners for ``episodes`` episodes and returns their exact-value learning curves.

    uv run python -m src.train --exp E1 [--n_runs 100] [--seed 0] [--out runs/...]

PRNG key tree:
    root      = PRNGKey(seed)
    exp_key   = fold_in(root, EXP_ID[exp])            E1..E8 -> 1..8
    set_key   = fold_in(exp_key, setting_idx)         sweep point (sigma / eps / p_long)
    run_keys  = split(set_key, n_runs)                one per run, SHARED by all methods (paired design)
    ep_key    = fold_in(run_key, episode)             (E2: fold_in(run_key, rollout))
        fold_in(ep_key, 0) env transitions, 1 reward noise, 2 action sampling (data.sample_episode)
    PG lr tuning uses root' = PRNGKey(seed + pg_tuning_seed_offset) through the same tree.
Method and learning rate never enter the key: the same run index sees the same uniforms/normals.

A run is seconds of compute, so there is no checkpointing; results are written atomically at the end of each experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from . import data as D
from .algorithm import (counterfactual_q, episode_grads, state_hca_grads, return_hca_grads, mc_returns, return_bins,
                        windows)
from .eval import dp_q, start_value
from .model import Params, init_params, policy

from rl_jax_bench.jaxlog import Timer, check_device, info, metric, nan_guard, warn, write_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "src", "configs", "default.yaml")
EXP_ID = {f"E{i}": i for i in range(1, 9)}
METHODS = ("state_hca", "return_hca", "pg")
ENV_FNS = {"shortcut": D.shortcut, "delayed_effect": D.delayed_effect, "ambiguous_bandit": D.ambiguous_bandit}


def load_config(path: str = CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def code_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "uncommitted"


def settings(exp_cfg: dict) -> list[dict]:
    """Expand a sweep into a list of env kwargs (one per sweep point)."""
    base = dict(exp_cfg.get("env_kwargs", {}))
    sweep = exp_cfg.get("sweep")
    if not sweep:
        return [base]
    (k, vals), = sweep.items()
    return [{**base, k: v} for v in vals]


def run_keys(seed: int, exp: str, setting_idx: int, n_runs: int):
    k = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), EXP_ID[exp]), setting_idx)
    return jax.random.split(k, n_runs)


# ---------------------------------------------------------------- learning runs
def make_optimizer(lr_pi, lr_h, lr_v, lr_r):
    """Plain SGD per table (ascent: the update directions are negated before optax)."""
    labels = Params(theta="pi", psi="h", phi="h", V="v", rhat="r")
    return optax.multi_transform({"pi": optax.sgd(lr_pi), "h": optax.sgd(lr_h), "v": optax.sgd(lr_v),
                                  "r": optax.sgd(lr_r)}, labels)


def apply_update(params: Params, grads, lrs) -> Params:
    opt = make_optimizer(*lrs)
    ascent = Params(*(-g for g in grads))
    upd, _ = opt.update(ascent, opt.init(params), params)
    return optax.apply_updates(params, upd)


def _custom_h_grads(method: str, p: Params, ep, H, seg, centres, lr_h, variant: str, seg_mode: str, gamma: float,
                    h_param: str, h_est: str):
    """Recheck h variants (S-HPRIOR residual h, S-HFORM counts / EMA). Returns (grads, new H table).
    The policy, V and r-hat directions are the unchanged estimator of algorithm.py evaluated with the variant's h;
    only how h is represented / estimated differs. psi/phi directions are zeroed for the table estimators."""
    o = ep.obs
    oh = jax.nn.one_hot(ep.action, p.theta.shape[-1])
    if method == "state_hca":
        if h_param == "residual":
            hp = jax.nn.softmax(p.theta[o][:, None, :] + p.psi[o[:, None], o[None, :]], -1)
        else:
            hp = (H / jnp.sum(H, -1, keepdims=True))[o[:, None], o[None, :]]
        g = state_hca_grads(p, ep, seg, gamma, variant, h_pair_override=hp, seg_mode=seg_mode)
        if h_est != "sgd":
            w = windows(ep, seg, gamma, seg_mode)
            inc = w.hpair[..., None] * (oh[:, None, :] if h_est == "counts" else lr_h * (oh[:, None, :] - hp))
            H = H.at[o[:, None], o[None, :]].add(inc)
            g = g._replace(psi=jnp.zeros_like(g.psi))
        return g, H
    b = return_bins(mc_returns(ep, gamma), centres)
    if h_param == "residual":
        hz = jax.nn.softmax(p.theta[o] + p.phi[o, b], -1)
    else:
        hz = (H / jnp.sum(H, -1, keepdims=True))[o, b]
    g = return_hca_grads(p, ep, gamma, centres, hz_override=hz)
    if h_est != "sgd":
        inc = ep.decision[:, None] * (oh if h_est == "counts" else lr_h * (oh - hz))
        H = H.at[o, b].add(inc)
        g = g._replace(phi=jnp.zeros_like(g.phi))
    return g, H


def make_runner(env: D.TabEnv, method: str, *, episodes: int, seg_len, n_bins: int, bin_widen: float,
                gamma: float, variant: str = "official", freeze_policy: bool = False, h_lr_decay_tau=None,
                order: str = "official", centres_override=None, seg_mode: str = "segment", bin_mode: str = "fixed",
                h_param: str = "logit", h_est: str = "sgd", theta0=None, h0: str = "zero"):
    """Returns jitted f(run_keys[n], lrs[4]) -> dict(curve [n, E+1], final params). lrs = (pi, h, V, rhat).
    ``h_lr_decay_tau`` (exact-check convergence test only; never in experiments): hindsight lr / (1 + e / tau).
    ``order`` (HCA arms only): 'official' = every direction from the pre-update tables (primary);
    'alg' = the literal Algorithm 1/2 order -- h (and V, r-hat) are trained on the episode first and the policy
    credit is then computed with the updated tables (full-scale divergence probe, never the primary setting).
    ``centres_override`` (full-scale bin-range probe): explicit return-bin centres instead of A-BINS.
    ``seg_mode`` (sensitivity): 'segment' (A-SEG, primary) or 'sliding' n-step windows.
    ``bin_mode`` (sensitivity): 'fixed' (A-BINS or ``centres_override``, primary) or 'running' =
    centres linspace(min, max, n_b) over the Monte-Carlo returns this run has observed so far, including the current
    episode (uses no environment statistics).
    Recheck sensitivities (2026-10-11). The defaults reproduce the pre-recheck primary bit for bit:
    ``h_param`` 'logit' (A-INIT: h = softmax(psi), psi zero-initialised; primary) or 'residual' (S-HPRIOR:
    h = softmax(theta[x] + psi[x, .]) with the current policy logits as offset, so h = pi in unvisited cells; same CE
    fixed point; the official dlogits_h step is applied to psi).
    ``h_est`` 'sgd' (one official logit-space CE step per pair; primary), 'counts' (S-HFORM: maximum-likelihood
    h = (N + 1) / sum(N + 1) over all pairs so far) or 'ema' (S-HFORM: probability-space h += lr_h (onehot - h)).
    ``theta0`` (S-E1START): initial policy logits [n_obs, A] instead of zeros; ``h0`` 'zero' (uniform h,
    A-INIT) or 'policy' (hindsight logits initialised at theta0 of the source observation, so h = pi at start)."""
    if h_param not in ("logit", "residual") or h_est not in ("sgd", "counts", "ema") or h0 not in ("zero", "policy"):
        raise ValueError((h_param, h_est, h0))
    custom_h = method in ("state_hca", "return_hca") and (h_param != "logit" or h_est != "sgd")
    if custom_h and (order != "official" or (h_param == "residual" and h_est != "sgd") or h0 != "zero"
                     or bin_mode != "fixed"):
        raise ValueError("recheck h options: one at a time, with order='official', h0='zero', bin_mode='fixed'")
    ea = D.env_arrays(env)
    centres = jnp.asarray(D.bin_centres(env, n_bins, bin_widen) if centres_override is None else centres_override)
    s0, L = env.s0, env.L
    seg = None if method == "return_hca" else seg_len  # A-RET-MC: Return HCA never bootstraps

    if bin_mode not in ("fixed", "running"):
        raise ValueError(bin_mode)

    def one_run(key, lrs):
        params = init_params(env.n_obs, n_bins)
        if theta0 is not None:
            th0 = jnp.asarray(theta0, dtype=params.theta.dtype)
            params = params._replace(theta=th0)
            if h0 == "policy":
                params = params._replace(psi=jnp.broadcast_to(th0[:, None, :], params.psi.shape) + 0.0,
                                         phi=jnp.broadcast_to(th0[:, None, :], params.phi.shape) + 0.0)
        if custom_h and h_est != "sgd":
            shp = params.psi.shape if method == "state_hca" else params.phi.shape
            H0 = jnp.ones(shp) if h_est == "counts" else jnp.full(shp, 0.5)
        else:
            H0 = jnp.zeros((1,))
        v0 = start_value(ea, params.theta, s0, L)

        def body(carry, e):
            p, zr, H = carry
            k = jax.random.fold_in(key, e)
            pi = policy(p.theta)
            ep = D.sample_episode(ea, s0, L, lambda o: pi[o], k)
            if bin_mode == "running":
                Zs = mc_returns(ep, gamma)
                zr = jnp.stack([jnp.minimum(zr[0], jnp.min(jnp.where(ep.decision, Zs, jnp.inf))),
                                jnp.maximum(zr[1], jnp.max(jnp.where(ep.decision, Zs, -jnp.inf)))])
                cen = jnp.linspace(zr[0], zr[1], n_bins)
            else:
                cen = centres
            lr_e = lrs if h_lr_decay_tau is None else lrs * jnp.array([1.0, 1.0 / (1.0 + e / h_lr_decay_tau), 1.0, 1.0])
            if custom_h:
                g, H = _custom_h_grads(method, p, ep, H, seg, cen, lr_e[1], variant, seg_mode, gamma, h_param, h_est)
            else:
                g = episode_grads(method, p, ep, seg_len=seg, gamma=gamma, centres=cen, variant=variant,
                                  seg_mode=seg_mode)
            if order == "alg" and method != "pg":
                p_aux = p._replace(psi=p.psi + lr_e[1] * g.psi, phi=p.phi + lr_e[1] * g.phi,
                                   V=p.V + lr_e[2] * g.V, rhat=p.rhat + lr_e[3] * g.rhat)
                g2 = episode_grads(method, p_aux, ep, seg_len=seg, gamma=gamma, centres=cen, variant=variant,
                                   seg_mode=seg_mode)
                g = g._replace(theta=g2.theta)
            elif order not in ("official", "alg"):
                raise ValueError(order)
            if freeze_policy:
                g = g._replace(theta=jnp.zeros_like(g.theta))
            p = apply_update(p, g, lr_e)
            return (p, zr, H), (start_value(ea, p.theta, s0, L), jnp.sum(ep.reward))

        zr0 = jnp.array([jnp.inf, -jnp.inf])
        (params, _, _), (vals, rets) = jax.lax.scan(body, (params, zr0, H0), jnp.arange(episodes))
        curve = jnp.concatenate([v0[None], vals])
        return curve, rets, params

    @jax.jit
    def run(keys, lrs):
        curve, rets, params = jax.vmap(one_run, in_axes=(0, None))(keys, lrs)
        nan_guard((curve, params), keys, tag=1, dump_dir=os.path.join(ROOT, "nan_dumps"))
        return dict(curve=curve, returns=rets, params=params)

    return run


def method_lrs(cfg: dict, method: str, pg_lr: float | None = None):
    h = cfg["hca"]
    if method == "pg":
        return jnp.array([pg_lr, 0.0, pg_lr, 0.0])  # A-PG-LR: one lr for actor and critic
    return jnp.array([h["policy_lr"], h["hindsight_lr"], h["value_lr"], h["reward_model_lr"]])


def make_env(exp_cfg: dict, kwargs: dict) -> D.TabEnv:
    return ENV_FNS[exp_cfg["env"]](**kwargs)


# ---------------------------------------------------------------- Fig. 3 right (fixed policy)
def make_fig3r_runner(env: D.TabEnv, method: str, p_long: float, *, rollouts: int, n_bins: int,
                      bin_widen: float, gamma: float, immediate: str, hca_lrs, estimate_after_update: bool = False,
                      pg_baseline: str = "batch", pg_v_lr: float = 0.3):
    """Per repetition: ``rollouts`` episodes of the fixed policy (pi(long) = p_long at every chain state),
    hindsight tables learned online (A-F3R) with the pre-update tables used for that rollout's estimate.
    Returns per-rep mean of Qhat(s0,short) - Qhat(s0,long) (online h, primary) and the final-table
    re-evaluation on the same rollouts (secondary). ``estimate_after_update`` (full-scale probe): the rollout's
    estimate uses the tables after they were trained on that rollout (Algorithm 1/2 order).
    ``pg_baseline`` (PG arm only): 'batch' = mean return over the repetition's rollouts (A-F3R primary, non-causal);
    'online_v' (sensitivity) = a tabular V(x0) learned online, V += pg_v_lr (Z - V), with each rollout's
    estimate using the V from before that rollout (zero init, as A-INIT)."""
    ea = D.env_arrays(env)
    centres = jnp.asarray(D.bin_centres(env, n_bins, bin_widen))
    s0, L = env.s0, env.L
    theta = jnp.tile(jnp.array([0.0, jnp.log(p_long / (1 - p_long))]), (env.n_obs, 1))
    lr_h, lr_r = hca_lrs[1], hca_lrs[3]

    def one_rep(key):
        params = init_params(env.n_obs, n_bins)._replace(theta=theta)
        pi = policy(theta)

        def body(p, r):
            ep = D.sample_episode(ea, s0, L, lambda o: pi[o], jax.random.fold_in(key, r))
            q = counterfactual_q(method, p, ep, gamma, centres, immediate)
            if method == "state_hca":
                g = state_hca_grads(p, ep, None, gamma, "paper")  # dpsi is variant-independent; gives drhat
                p = p._replace(psi=p.psi + lr_h * g.psi, rhat=p.rhat + lr_r * g.rhat)
            elif method == "return_hca":
                g = return_hca_grads(p, ep, gamma, centres)
                p = p._replace(phi=p.phi + lr_h * g.phi)
            if estimate_after_update:
                q = counterfactual_q(method, p, ep, gamma, centres, immediate)
            return p, (q, ep)

        p_fin, (qs, eps) = jax.lax.scan(body, params, jnp.arange(rollouts))
        if method == "pg":
            Z = jnp.sum(qs, -1)  # one_hot * Z summed = Z
            d = jnp.where(eps.action[:, 0] == 0, 1.0, -1.0)
            if pg_baseline == "batch":
                est = jnp.mean(d * (Z - jnp.mean(Z)))  # baseline = batch-mean return (A-F3R primary)
            elif pg_baseline == "online_v":
                def vstep(v, z):
                    return v + pg_v_lr * (z - v), v
                _, vs = jax.lax.scan(vstep, jnp.asarray(0.0), Z)
                est = jnp.mean(d * (Z - vs))
            else:
                raise ValueError(pg_baseline)
            return est, est
        online = jnp.mean(qs[:, 0] - qs[:, 1])
        q_fin = jax.vmap(lambda e: counterfactual_q(method, p_fin, e, gamma, centres, immediate))(eps)
        return online, jnp.mean(q_fin[:, 0] - q_fin[:, 1])

    @jax.jit
    def run(keys):
        online, final = jax.vmap(one_rep)(keys)
        nan_guard((online, final), keys, tag=2, dump_dir=os.path.join(ROOT, "nan_dumps"))
        return dict(online=online, final=final)

    oracle = dp_q(ea, theta, L, gamma)[s0]
    return run, float(oracle[0] - oracle[1])


# ---------------------------------------------------------------- experiment drivers
def run_learning_exp(cfg: dict, exp: str, n_runs: int, seed: int, pg_lrs=None, methods=METHODS,
                     variant: str | None = None) -> dict:
    """All methods (and every PG lr in the grid) on every sweep point of a learning-curve experiment."""
    ec = cfg["experiments"][exp]
    variant = variant or cfg["hca"]["state_variant"]
    pg_lrs = cfg["pg"]["lr_grid"] if pg_lrs is None else pg_lrs
    out = {"exp": exp, "settings": [], "n_runs": n_runs, "seed": seed}
    for si, kw in enumerate(settings(ec)):
        env = make_env(ec, kw)
        keys = run_keys(seed, exp, si, n_runs)
        rec = {"env_kwargs": kw, "env_sha256": D.env_sha256(env), "env": env.name, "methods": {}}
        for m in methods:
            arms = [(f"pg_lr{lr}", lr) for lr in pg_lrs] if m == "pg" else [(m, None)]
            runner = make_runner(env, m, episodes=ec["episodes"], seg_len=ec["seg_len"], n_bins=ec["n_bins"],
                                 bin_widen=cfg["hca"]["bin_widen"], gamma=cfg["gamma"], variant=variant)
            for name, lr in arms:
                lrs = method_lrs(cfg, m, lr)
                tm = Timer()
                res = jax.block_until_ready(runner(keys, lrs))
                t_first = tm.lap()
                jax.block_until_ready(runner(keys, lrs))
                t_steady = tm.lap()
                curve = np.asarray(res["curve"])
                if not np.all(np.isfinite(curve)):
                    warn("non-finite curve", exp=exp, arm=name)
                rec["methods"][name] = {"curve": curve, "first_call_s": t_first, "steady_call_s": t_steady}
                metric(f"{exp} {name}", setting=si, v0=float(curve[:, 0].mean()), final=float(curve[:, -1].mean()),
                       final_se=float(curve[:, -1].std(ddof=1) / np.sqrt(n_runs)),
                       curve_mean=float(curve.mean()), first_call_s=t_first, steady_s=t_steady,
                       eps_per_s=n_runs * ec["episodes"] / max(t_steady, 1e-9))
        out["settings"].append(rec)
    return out


def run_fig3r(cfg: dict, n_runs: int, seed: int, immediate: str | None = None) -> dict:
    ec = cfg["experiments"]["E2"]
    env = make_env(ec, ec["env_kwargs"])
    immediate = immediate or ec["immediate"]
    out = {"exp": "E2", "env_sha256": D.env_sha256(env), "immediate": immediate, "points": []}
    for si, p in enumerate(ec["p_long"]):
        keys = run_keys(seed, "E2", si, n_runs)
        rec = {"p_long": p, "methods": {}}
        for m in METHODS:
            run, oracle = make_fig3r_runner(env, m, p, rollouts=ec["rollouts"], n_bins=ec["n_bins"],
                                            bin_widen=cfg["hca"]["bin_widen"], gamma=cfg["gamma"],
                                            immediate=immediate, hca_lrs=method_lrs(cfg, "state_hca"))
            tm = Timer()
            res = jax.block_until_ready(run(keys))
            rec["oracle"] = oracle
            rec["methods"][m] = {k: np.asarray(v) for k, v in res.items()} | {"first_call_s": tm.lap()}
            metric(f"E2 {m}", p_long=round(p, 4), oracle=oracle, online=float(res["online"].mean()),
                   final_h=float(res["final"].mean()))
        out["points"].append(rec)
    return out


def provenance(cfg: dict) -> dict:
    import flax
    return {"config_sha256": config_sha256(cfg), "code_commit": code_commit(),
            "official_commit": "3a564ad6438df2b1c68ca75a91d713d87d56453a",
            "jax": jax.__version__, "optax": optax.__version__, "flax": flax.__version__,
            "device": str(jax.devices()[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="E1..E8 or 'all'")
    ap.add_argument("--n_runs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pg_lrs", type=float, nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config()
    check_device()
    n_runs = args.n_runs or cfg["n_runs"]
    seed = cfg["seed"] if args.seed is None else args.seed
    exps = list(EXP_ID) if args.exp == "all" else [args.exp]
    out_dir = args.out or os.path.join(ROOT, "runs", f"seed_{seed}")
    result = {"provenance": provenance(cfg), "config": cfg, "experiments": {}}
    t0 = time.perf_counter()
    for e in exps:
        info("experiment", exp=e, n_runs=n_runs, seed=seed)
        result["experiments"][e] = run_fig3r(cfg, n_runs, seed) if e == "E2" else \
            run_learning_exp(cfg, e, n_runs, seed, pg_lrs=args.pg_lrs)
    result["wall_s"] = time.perf_counter() - t0
    write_json(os.path.join(out_dir, "metrics.json"), result)
    info("wrote", path=os.path.join(out_dir, "metrics.json"), wall_s=result["wall_s"])


if __name__ == "__main__":
    main()
