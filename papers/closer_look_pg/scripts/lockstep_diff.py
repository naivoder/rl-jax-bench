"""Recheck : MULTI-ITERATION lockstep differential of our training update against the
official MadryLab/implementation-matters@5ee6ecb primitives.

One-step parity (tests/test_official_parity.py) cannot see state that persists across iterations: Adam moments of the
policy and value optimisers, the LambdaLR schedule, value clipping against the previous value net's outputs, and the
coupling of each iteration's GAE to the value net the previous iteration produced. Here our REAL training iteration
(src.train.make_iteration's rollout -> batch_from_traj -> update, the full-scale config of the arm) runs K iterations from
the full-scale init of a seed, recording every batch (raw policy inputs, actions, log-probs, training-unit rewards,
not_done) and every key-derived permutation / Fisher subset. The official code then replays the K batches with ITS OWN
persistent state (its own value net computes the values and GAE; its own Adam optimisers and schedulers persist), and
its parameters after each iteration are compared with ours. Pass = every parameter tensor within 1e-4 relative of
ours after every iteration (the one-step tolerance), with the official GAE inputs checked along the way.

Usage: JAX_PLATFORMS=cpu python scripts/lockstep_diff.py --env walker2d --alg ppo --iters 10 --seed 0
       -> runs/recheck/lockstep/<env>_<alg>_s<seed>.json (+ the recorded .npz, git-ignored)
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402  (bootstrap first)
from src.common import info, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import data as D  # noqa: E402
from src.train import batch_from_traj, build  # noqa: E402

ORACLE = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))
HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "official")
OUT = os.path.join(common.REPO, "runs", "recheck", "lockstep")


def layers(nets, flat, prefix, policy=True):
    d = (nets.pi_unravel if policy else nets.v_unravel)(flat)["params"]
    sub = d["mean"] if policy else d["v"]
    out = {}
    for i in range(3):
        out[f"{prefix}W{i}"] = np.asarray(sub[f"Dense_{i}"]["kernel"], np.float32)
        out[f"{prefix}b{i}"] = np.asarray(sub[f"Dense_{i}"]["bias"], np.float32)
    if policy:
        out[f"{prefix}log_std"] = np.asarray(d["log_std"], np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--alg", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env_kinematics", default="legacy", choices=["legacy", "post_step"])
    ap.add_argument("--start_run", default=None,
                    help="start from a saved checkpoint instead of the init: runs/<exp>/<env>_<alg> (with --start_it); "
                         "the policy, value net and normaliser come from the checkpoint, both optimisers start fresh on "
                         "BOTH sides (identity test of the update at a late training state, e.g. where TRPO's line "
                         "search accepts fractional steps)")
    ap.add_argument("--start_it", type=int, default=0)
    ap.add_argument("--control_eps", type=float, default=1e-7,
                    help="float-chaos control: OUR update replayed on the SAME recorded batches (values/GAE recomputed "
                         "with its own value net, recorded log-probs, as the official replay does) from an init "
                         "perturbed by control_eps * |theta| * N(0,1); its divergence from our trajectory is the "
                         "float-level sensitivity any second implementation would show")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg, env, spec, nets, rollout, update, init_state = build(
        args.env, args.alg, {"env_post_step_kinematics": args.env_kinematics == "post_step"})
    T = int(cfg["steps_per_iteration"])
    root = jax.random.PRNGKey(args.seed)
    ts = init_state(jax.random.fold_in(root, common.K_INIT))
    norm = D.norm_init(spec.obs_dim)
    it0 = 0
    if args.start_run:
        from src.algorithm import TrainState
        from src.train import load_ckpt
        rd = os.path.join(common.REPO, args.start_run)
        seeds = sorted(int(x.split("_")[1]) for x in os.listdir(rd) if x.startswith("seed_"))
        S = len(seeds)
        like = {"ts": jax.tree_util.tree_map(lambda x: jnp.stack([x] * S), ts),
                "norm": jax.tree_util.tree_map(lambda x: jnp.stack([x] * S), norm), "it": jnp.int32(0)}
        ck = load_ckpt(os.path.join(rd, "ckpt", f"it{args.start_it:04d}"), like)
        j = seeds.index(args.seed)
        pick = lambda t: jax.tree_util.tree_map(lambda x: x[j], t)
        pi_tx, v_tx = A.make_optimizers(cfg)
        c = pick(ck["ts"])
        ts = TrainState(c.pi, c.v, pi_tx.init(c.pi), v_tx.init(c.v))
        norm = pick(ck["norm"])
        it0 = args.start_it
    j_roll = jax.jit(rollout, static_argnums=(3, 4))
    j_upd = jax.jit(update)
    rec = {k: [] for k in ("obs", "act", "logp", "rew", "not_done", "values", "adv", "ret", "v_perms", "p_perms",
                           "subset", "pi_after", "v_after")}
    lay0 = {**layers(nets, ts.pi, "pi0_"), **layers(nets, ts.v, "v0_", False)}
    ts0_for_control = ts
    after = []
    kups = []
    for it in range(it0, it0 + args.iters):
        k_it = jax.random.fold_in(jax.random.fold_in(root, common.K_TRAIN), it)   # == train.make_iteration
        k_roll, k_upd = jax.random.split(k_it)
        traj, norm = j_roll(ts.pi, norm, k_roll, T, True)
        b = batch_from_traj(nets, cfg, ts.v, traj)
        kups.append(k_upd)
        k_v, k_p = jax.random.split(k_upd)                                          # == make_train_update
        n = b.obs.shape[0]
        rec["v_perms"].append(np.asarray(A._minibatch_perms(k_v, n, cfg["value_epochs"], cfg["value_minibatches"])
                                         ).reshape(cfg["value_epochs"], n))
        if args.alg == "ppo":
            rec["p_perms"].append(np.asarray(A._minibatch_perms(k_p, n, cfg["policy_epochs"], cfg["minibatches"])
                                             ).reshape(cfg["policy_epochs"], n))
        else:
            rec["subset"].append(np.asarray(jax.random.choice(k_p, n, (int(n * cfg["fisher_fraction"]),),
                                                              replace=False)))
        for k, x in (("obs", b.obs), ("act", b.act), ("logp", b.logp), ("rew", traj.rew), ("not_done", b.not_done),
                     ("values", b.values), ("adv", b.adv), ("ret", b.ret)):
            rec[k].append(np.asarray(x))
        ts, st = j_upd(ts, b, jnp.int32(it), k_upd)
        rec.setdefault("ls_scale", []).append(np.asarray(st.get("ls_scale", np.nan)))
        after.append({**layers(nets, ts.pi, "pi_"), **layers(nets, ts.v, "v_", False)})
        info("lockstep ours", it=it, mean_reward=float(D.completed_episode_mean(traj)[0]))
    # ---- float-chaos control (ours vs ours, perturbed init, same batches, official-replay semantics)
    kc = jax.random.PRNGKey(10_007)
    k1, k2 = jax.random.split(kc)
    from src.algorithm import TrainState
    ts_c = ts0_for_control
    ts_c = TrainState(ts_c.pi * (1 + args.control_eps * jax.random.normal(k1, ts_c.pi.shape)),
                      ts_c.v * (1 + args.control_eps * jax.random.normal(k2, ts_c.v.shape)), ts_c.pi_opt, ts_c.v_opt)
    ctrl = []
    for it in range(args.iters):
        obs, nd = jnp.asarray(rec["obs"][it]), jnp.asarray(rec["not_done"][it])
        vals = nets.v_apply(ts_c.v, obs)
        adv, ret = A.gae(jnp.asarray(rec["rew"][it]), vals, nd, cfg["gamma"], cfg["gae_lambda"])
        bc = A.Batch(obs, jnp.asarray(rec["act"][it]), jnp.asarray(rec["logp"][it]), adv, ret, vals, nd)
        ts_c, _ = j_upd(ts_c, bc, jnp.int32(it0 + it), kups[it])
        ctrl.append({**layers(nets, ts_c.pi, "pi_"), **layers(nets, ts_c.v, "v_", False)})
    inp = {k: np.stack(v) for k, v in rec.items() if v}
    inp.update(lay0)
    for key in after[0]:
        inp[f"after_{key}"] = np.stack([a[key] for a in after])
    sc = dict(alg=args.alg, iters=args.iters, gamma=cfg["gamma"], lam=cfg["gae_lambda"], value_lr=cfg["value_lr"],
              value_epochs=cfg["value_epochs"], value_minibatches=cfg["value_minibatches"],
              value_clipping=bool(cfg.get("value_clipping", False)), clip_val_eps=cfg.get("clip_val_eps", 0.2),
              anneal=bool(cfg.get("anneal_lr", False)), train_iterations=cfg["train_iterations"], start_it=it0)
    if args.alg == "ppo":
        sc.update(policy_lr=cfg["policy_lr"], policy_epochs=cfg["policy_epochs"], minibatches=cfg["minibatches"],
                  clip_eps=cfg["clip_eps"], init=cfg["init"])
    else:
        sc.update(max_kl=cfg["kl_delta"], damping=cfg["cg_damping"], cg_steps=cfg["cg_steps"],
                  fisher_fraction=cfg["fisher_fraction"], backtrack_steps=cfg["backtrack_steps"], init=cfg["init"])
    inp["config_json"] = np.asarray(json.dumps(sc))
    tag = f"{args.env}_{args.alg}_s{args.seed}" + (f"_from{it0}" if args.start_run else "")
    fin, fout = os.path.join(OUT, f"{tag}_in.npz"), os.path.join(OUT, f"{tag}_official.npz")
    np.savez(fin, **inp)
    py = os.path.join(ORACLE, ".venv", "bin", "python")
    res = subprocess.run([py, "-I", os.path.join(HARNESS, "lockstep_replay.py"), fin, fout], capture_output=True,
                         text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:], res.stderr[-4000:])
        raise SystemExit(1)
    o = dict(np.load(fout))
    rows = []
    ok = True
    for it in range(args.iters):
        r = {"it": it}
        for what in ("adv", "ret", "logp"):
            ours, th = inp[what][it].astype(np.float64), o[f"{what}"][it].astype(np.float64)
            r[f"{what}_max_rel"] = float(np.max(np.abs(ours - th)) / max(np.max(np.abs(th)), 1e-12))
        for net in ("pi", "v"):
            errs, scale = [], 0.0
            for key in after[0]:
                if not key.startswith(net + "_"):
                    continue
                ours, th = inp[f"after_{key}"][it].astype(np.float64), o[f"after_{key}"][it].astype(np.float64)
                errs.append(np.max(np.abs(ours - th)))
                scale = max(scale, float(np.max(np.abs(th))))
            r[f"{net}_max_abs"] = float(max(errs))
            r[f"{net}_max_rel"] = float(max(errs) / scale)
            # how far the parameters moved this iteration (so a match is not a match of non-moving params)
            prev = (lambda k: inp[f"after_{k}"][it - 1]) if it > 0 else (lambda k: inp[k.replace(net + "_", net + "0_")])
            r[f"{net}_step_norm"] = float(np.sqrt(sum(np.sum((inp[f"after_{k}"][it].astype(np.float64)
                                                               - prev(k).astype(np.float64)) ** 2)
                                                        for k in after[0] if k.startswith(net + "_"))))
        for net in ("pi", "v"):
            keys_ = [k for k in after[0] if k.startswith(net + "_")]
            scale = max(float(np.max(np.abs(after[it][k]))) for k in keys_)
            r[f"{net}_control_max_rel"] = float(max(np.max(np.abs(ctrl[it][k].astype(np.float64)
                                                                 - after[it][k].astype(np.float64))) for k in keys_) / scale)
        r["ls_scale_ours"] = float(inp["ls_scale"][it]) if "ls_scale" in inp else None
        r["ok"] = bool(r["pi_max_rel"] <= 1e-4 and r["v_max_rel"] <= 1e-4)
        # within float chaos: official-vs-ours no larger than 10x ours-vs-ours(perturbed) at the same iteration
        r["within_float_chaos"] = bool(r["pi_max_rel"] <= max(1e-4, 10 * r["pi_control_max_rel"]) and
                                       r["v_max_rel"] <= max(1e-4, 10 * r["v_control_max_rel"]))
        ok &= r["ok"]
        rows.append(r)
        print(json.dumps(r))
    chaos_ok = all(r["within_float_chaos"] for r in rows)
    write_json(os.path.join(OUT, f"{tag}.json"), {"pass": ok, "pass_within_float_chaos": chaos_ok, "tol_rel": 1e-4,
                                                  "control_eps": args.control_eps, "config": sc, "rows": rows,
                                                  "official_stdout": res.stdout[-500:], **common.git_info()})
    print("LOCKSTEP PASS" if ok else "LOCKSTEP FAIL (strict 1e-4)",
          "| within float chaos:", chaos_ok)
    return 0 if (ok or chaos_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
