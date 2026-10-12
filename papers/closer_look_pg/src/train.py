"""E0/E2 training: PPO or TRPO on a brax MJX proxy, vmapped over seeds in one process.

Per iteration: fresh 2048-step single-actor rollout from reset (normalisers updated), GAE, value step then
policy step (official order). Every ``value_log_every`` iterations: a held-out 2048-step rollout collected
BEFORE the update with frozen normalisers, value errors on train + held-out after the
update, and a 10-episode stochastic evaluation. Checkpoints theta_k (params after k updates, plus normaliser,
optimiser state; normalisers as after rollout k) at the instrument iterations; orbax resume checkpoints every ``ckpt_every`` iterations.

Usage: python -m src.train --env hopper --alg ppo --seeds 0 1 2 --iters 488 --out runs/e0
"""
from __future__ import annotations

import argparse
import json
import os
import time

from . import common
from .common import Timer, info, metric, warn, write_json

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from . import algorithm as A
from . import data as D
from .configs.default import INSTRUMENTS, algo_config
from .model import Nets


def build(env_name: str, alg: str, overrides: dict | None = None):
    cfg = algo_config(alg, env_name)
    cfg.update(overrides or {})
    env, spec = D.make_env(env_name, cfg["physics_backend"], bool(cfg.get("env_post_step_kinematics", False)))
    nets = Nets(spec.obs_dim, spec.act_dim, cfg)
    rollout = D.make_rollout(env, cfg, nets)
    update, init_state = A.make_train_update(nets, cfg)
    return cfg, env, spec, nets, rollout, update, init_state


def batch_from_traj(nets, cfg, v_flat, traj):
    """[T] single-actor trajectory -> A.Batch with the agent's own V and GAE."""
    values = nets.v_apply(v_flat, traj.obs)
    adv, ret = A.gae(traj.rew, values, traj.not_done, cfg["gamma"], cfg["gae_lambda"])
    vadv = None if cfg.get("adv_baseline", "gae") == "gae" else adv   # value target stays V + A^GAE in every arm
    if cfg.get("adv_baseline", "gae") == "zero_gae":  # E5 divergence test: GAE(gamma, lambda) with V == 0
        adv, _ = A.gae(traj.rew, jnp.zeros_like(values), traj.not_done, cfg["gamma"], cfg["gae_lambda"])
    if cfg.get("adv_baseline", "gae") == "zero":     # E5 arm: advantages := MC returns (no baseline)
        adv = ret
    return A.Batch(traj.obs, traj.act, traj.logp, adv, ret, values, traj.not_done, vadv)


def make_iteration(cfg, nets, rollout, update, log_value: bool):
    T = int(cfg["steps_per_iteration"])

    def iteration(ts: A.TrainState, norm: D.NormState, it, root):
        k_it = jax.random.fold_in(jax.random.fold_in(root, common.K_TRAIN), it)
        k_roll, k_upd = jax.random.split(k_it)
        traj, norm1 = rollout(ts.pi, norm, k_roll, T, True)
        b = batch_from_traj(nets, cfg, ts.v, traj)
        out = {}
        if log_value:
            k_h = common.k_instr(root, it, common.I_HELDOUT)
            traj_h, _ = rollout(ts.pi, norm1, k_h, int(INSTRUMENTS["heldout_pairs"]), False)
            bh = batch_from_traj(nets, cfg, ts.v, traj_h)
        ts1, st = update(ts, b, it, k_upd)
        common.nan_guard((ts1.pi, ts1.v), {"obs": traj.obs, "rew": traj.rew}, tag=1,
                         dump_dir=os.path.join(common.REPO, "nan_dumps"))
        mrew, nep = D.completed_episode_mean(traj)
        out.update(st, mean_reward=mrew, n_episodes=nep, mean_std=jnp.mean(jnp.exp(nets.pi_apply(ts.pi, traj.obs[:1])[1])),
                   rew_clip_rate=jnp.mean(traj.rew_clip_hit), ep_len_mean=jnp.sum(jnp.where(traj.not_done < .5,
                   traj.ep_len, 0)) / jnp.maximum(jnp.sum(traj.not_done < .5), 1))
        if log_value:
            clip = bool(cfg.get("value_clipping", False))
            for tag, bb in (("train", b), ("heldout", bh)):
                ve = A.value_errors(nets.v_apply(ts1.v, bb.obs), bb.values, bb.adv if bb.vadv is None else bb.vadv, bb.ret, bb.not_done, clip,
                                    cfg.get("clip_val_eps", 0.2))
                for k, v in ve.items():
                    out[f"{tag}_{k}"] = v
            # 10-episode stochastic evaluation of the PRE-update policy (unbiased first-episode returns)
            k_e = common.k_instr(root, it, common.I_EVAL)
            tr_e, _ = jax.vmap(lambda k: rollout(ts.pi, norm1, k, int(cfg["episode_length"]), False))(
                jax.random.split(k_e, 10))
            sel, rets = D.unbiased_episode_returns(tr_e, int(cfg["episode_length"]))
            out["eval_return"] = jnp.sum(jnp.where(sel, rets, 0.0)) / jnp.maximum(jnp.sum(sel), 1)
        return ts1, norm1, out

    return iteration


def save_ckpt(path, tree):
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(os.path.abspath(path), tree, force=True)
    ckptr.wait_until_finished()


def load_ckpt(path, like):
    ckptr = ocp.StandardCheckpointer()
    abstract = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, like)
    return ckptr.restore(os.path.abspath(path), abstract)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--alg", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--iters", type=int, default=488)
    ap.add_argument("--out", default="runs/e0")
    ap.add_argument("--ckpt_iters", type=int, nargs="*", default=list(INSTRUMENTS["checkpoints"]))
    ap.add_argument("--resume_every", type=int, default=50)
    ap.add_argument("--value_log_every", type=int, default=INSTRUMENTS["value_log_every"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                    help="training rollouts are 2048 SEQUENTIAL single-env steps; MJX is far faster on CPU "
                         "(README perf table), instruments use the GPU")
    ap.add_argument("--train_iterations_for_schedule", type=int, default=None,
                    help="LR-annealing horizon (default: config train_iterations=488)")
    ap.add_argument("--adv_baseline", default="gae", choices=["gae", "zero", "zero_gae"],
                    help="E5: 'zero' trains with A_t := G_t (no baseline); the value net is still fit but unused")
    ap.add_argument("--adv_center", default="on", choices=["on", "off"],
                    help="DIAGNOSTIC (E5 divergence test): 'off' = PPO advantage normalisation divides by the std "
                         "but does NOT subtract the minibatch mean (the mean subtraction is an implicit constant baseline)")
    ap.add_argument("--env_kinematics", default="post_step", choices=["post_step", "legacy"],
                    help="recheck: 'post_step' (default) reads x/z after the step as gym does; 'legacy' is the "
                         "pre-fix brax behaviour (one 2 ms substep behind; all full-scale runs)")
    ap.add_argument("--reward_norm", default=None, choices=["on", "off"],
                    help="DIAGNOSTIC override of the Table-1 reward normalisation (official-code check: TRPO value-MRE scale "
                         "check against the official code's norm_rewards='returns'); reward clip left as configured")
    args = ap.parse_args(argv)

    common.check_device()
    dev = jax.devices("cpu")[0] if args.device == "cpu" else jax.devices()[0]
    if args.device == "cpu":
        warn("training placed on CPU BY CHOICE (sequential single-actor MJX stepping; see README perf table)")
    overrides = {"adv_baseline": args.adv_baseline, "adv_center": args.adv_center == "on",
                 "env_post_step_kinematics": args.env_kinematics == "post_step"}
    if args.train_iterations_for_schedule:
        overrides["train_iterations"] = args.train_iterations_for_schedule
    if args.reward_norm is not None:
        overrides["reward_norm"] = args.reward_norm == "on"
    cfg, env, spec, nets, rollout, update, init_state = build(args.env, args.alg, overrides)
    tag = f"{args.env}_{args.alg}" + {"gae": "", "zero": "-nobaseline", "zero_gae": "-zerogae"}[args.adv_baseline] + \
        ("" if args.adv_center == "on" else "-nocenter") + \
        ("" if args.reward_norm is None else f"-rewnorm{args.reward_norm}")
    out_dir = os.path.join(common.REPO, args.out, tag)
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(args.seeds)
    S = len(seeds)
    info("config", env=args.env, alg=args.alg, seeds=str(seeds), iters=args.iters, device=str(dev),
         obs_dim=spec.obs_dim, act_dim=spec.act_dim, n_pi=nets.n_pi, n_v=nets.n_v)

    with jax.default_device(dev):
        fp = D.env_fingerprint(env)
        info("env fingerprint", **fp)
        roots = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
        ts = jax.vmap(lambda r: init_state(jax.random.fold_in(r, common.K_INIT)))(roots)
        norm = jax.vmap(lambda _: D.norm_init(spec.obs_dim))(roots)
        it0 = 0
        resume_path = os.path.join(out_dir, "ckpt", "resume")
        if os.path.isdir(resume_path):
            st = load_ckpt(resume_path, {"ts": ts, "norm": norm, "it": jnp.int32(0)})
            ts, norm, it0 = st["ts"], st["norm"], int(st["it"])
            info("resumed", it=it0)
        hist_path = os.path.join(out_dir, "history.json")
        history = json.load(open(hist_path)) if (it0 > 0 and os.path.exists(hist_path)) else []
        history = [h for h in history if h["it"] < it0]

        it_plain = jax.jit(jax.vmap(make_iteration(cfg, nets, rollout, update, False), in_axes=(0, 0, None, 0)))
        it_log = jax.jit(jax.vmap(make_iteration(cfg, nets, rollout, update, True), in_axes=(0, 0, None, 0)))
        timer = Timer()
        t_compile = {}
        steady_t, steady_n = 0.0, 0
        for it in range(it0, args.iters):
            log_v = (it % args.value_log_every == 0)
            fn = it_log if log_v else it_plain
            timer.lap()
            ts_prev = ts
            ts, norm, out = fn(ts, norm, jnp.int32(it), roots)
            if it in args.ckpt_iters:
                # theta_k = params after k updates; normalisers as used by update k (after rollout k), so that the
                # iteration-0 checkpoint does not carry empty statistics (n=0 -> std 0 -> obs/1e-8).
                save_ckpt(os.path.join(out_dir, "ckpt", f"it{it:04d}"), {"ts": ts_prev, "norm": norm, "it": jnp.int32(it)})
            out = jax.tree_util.tree_map(np.asarray, out)
            dt = timer.lap()
            if log_v not in t_compile:
                t_compile[log_v] = dt
                info("first call (compile + run)", logged=log_v, seconds=round(dt, 2))
            elif not log_v:
                steady_t += dt
                steady_n += 1
            rec = {"it": it}
            for k, v in out.items():
                rec[k] = [float(x) for x in np.ravel(v)]
            history.append(rec)
            if log_v or it == args.iters - 1:
                metric("iter", it=it, mean_reward=float(np.mean(out["mean_reward"])),
                       **({"eval_return": float(np.mean(out["eval_return"])),
                           "heldout_returns_mre": float(np.mean(out["heldout_returns_mre"])),
                           "heldout_gae_mre_code": float(np.mean(out["heldout_gae_mre_code"]))} if log_v else {}),
                       steps_per_s=round(S * cfg["steps_per_iteration"] / max(dt, 1e-9), 1))
            if (it + 1) % args.resume_every == 0:
                save_ckpt(resume_path, {"ts": ts, "norm": norm, "it": jnp.int32(it + 1)})
                write_json(hist_path, history)
        write_json(hist_path, history)
        perf = {"compile_plus_first_s": {str(k): v for k, v in t_compile.items()},
                "steady_iter_s": steady_t / max(steady_n, 1),
                "steady_env_steps_per_s": S * cfg["steps_per_iteration"] * steady_n / max(steady_t, 1e-9)}
        info("timing", **{k: (v if not isinstance(v, dict) else str(v)) for k, v in perf.items()})
        for i, s in enumerate(seeds):
            per = [{k: (v[i] if isinstance(v, list) else v) for k, v in h.items()} for h in history]
            write_json(os.path.join(out_dir, f"seed_{s}", "metrics.json"),
                       {"env": args.env, "alg": args.alg, "seed": s, "config": {k: v for k, v in cfg.items()},
                        "env_fingerprint": fp, "perf": perf, **common.git_info(),
                        "key_tree": "root=PRNGKey(seed); see src/common.py", "history": per})
    info("done", out=out_dir)


if __name__ == "__main__":
    main()
