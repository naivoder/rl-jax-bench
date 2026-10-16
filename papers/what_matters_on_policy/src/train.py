"""one-factor experiment trainer: every (arm, seed) is a lane of ONE vmapped, jitted loop per environment.

  python -m src.train --env halfcheetah --arms primary --seeds 0-7 --tag full

Key tree (README): root = PRNGKey(root_seed); per paired seed s:
  k_env = fold_in(fold_in(root,0),s)  [initial reset = fold_in(k_env,0); reset pool of iteration i = fold_in(k_env,i+1)]
  k_init = fold_in(fold_in(root,1),s) [layer l: fold_in(k_init, l)]
  k_act = fold_in(fold_in(root,2),s)  [iteration i: fold_in(k_act,i), split over the unroll steps]
  k_mb  = fold_in(fold_in(root,3),s)  [iteration i, epoch e: fold_in(fold_in(k_mb,i),e); perm = fold_in(.,0), entropy
                                        sample of minibatch j = fold_in(., j+1)]
  k_eval= fold_in(fold_in(root,4),s)  [eval point p: fold_in(k_eval,p) -> (reset, action) split]
The arm never enters a key: two arms with identical configs produce identical lanes (tests/test_smoke.py).

Evaluation (paper §2): a policy snapshot (params + frozen obs stats) is taken at it=0 and at the first iteration
boundary at or after every 100k env steps; the snapshots are evaluated after training in wide batches (100
stochastic episodes each, separate env instances). Score = mean of the 10 points after step 0.
Checkpoint/resume: orbax, after every eval point (lane state incl. env states + snapshots).
"""
from __future__ import annotations

import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rl_jax_bench.jaxlog import check_device, info, metric, warn, write_json  # noqa: E402

from . import algorithm as A  # noqa: E402
from . import configs as C  # noqa: E402
from . import data as D  # noqa: E402
from . import model as M  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOL_FIELDS = ("obs_norm", "tanh", "softplus", "value_norm", "adv_norm", "recompute", "shared", "reward_norm",
               "logp_presquash")


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def seed_keys(root, s):
    f = lambda i: jax.random.fold_in(jax.random.fold_in(root, i), s)
    return {"env": f(0), "init": f(1), "act": f(2), "mb": f(3), "eval": f(4)}


def lane_cfg(r: dict, static: dict) -> A.LaneCfg:
    vals = {}
    for f in A.LaneCfg._fields:
        if f == "lr_horizon_steps":
            vals[f] = jnp.float32(C.lr_horizon_steps(static, r.get("lr_horizon_frames")))
        elif f in BOOL_FIELDS:
            vals[f] = jnp.bool_(r[f])
        else:
            vals[f] = jnp.float32(r[f])
    return A.LaneCfg(**vals)


def _flat_nonempty(tree):
    """orbax cannot save zero-size arrays (e.g. mjx Data.act for actuator-less models): save the non-empty leaves only."""
    return {str(i): x for i, x in enumerate(jax.tree.leaves(tree)) if x.size > 0}


def _unflat(template, flat):
    leaves, treedef = jax.tree.flatten(template)
    return jax.tree.unflatten(treedef, [flat.get(str(i), x) for i, x in enumerate(leaves)])


def eval_points(static: dict, it_total: int) -> list[int]:
    """Iteration index of each eval point: 0, then the first boundary >= k*eval_every (k=1..)."""
    per = static["iteration_size"]
    pts = [0]
    k = 1
    while k * static["eval_every"] <= it_total * per:
        pts.append(math.ceil(k * static["eval_every"] / per))
        k += 1
    return pts


def build(env, spec, cfg, arms, seeds, root, st: A.Static, static):
    vmax = max(C.resolve(cfg, a)["v_width"] for a in arms)
    cfgs, params, keys, meta = [], [], [], []
    for a in arms:
        r = C.resolve(cfg, a)
        for s in seeds:
            k = seed_keys(root, s)
            params.append(M.init_params(k["init"], spec.obs_dim, spec.act_dim, pi_width=static["pi_width"],
                                        pi_depth=static["pi_depth"], v_width=r["v_width"], v_depth=static["v_depth"],
                                        v_width_max=vmax, pi_last_gain=r["pi_last_gain"]))
            cfgs.append(lane_cfg(r, static))
            keys.append(k)
            meta.append({"arm": a, "seed": s, "resolved": r})
    stack = lambda xs: jax.tree.map(lambda *a: jnp.stack(a), *xs)
    params, cfgs, keys = stack(params), stack(cfgs), stack(keys)
    ls = jax.jit(jax.vmap(lambda p, k: A.init_lane(env, spec, p, st, k)))(params, keys["env"])
    return ls, cfgs, keys, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--arms", default="primary")
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--root_seed", type=int, default=20210101)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval_episodes", type=int, default=None)
    ap.add_argument("--total_steps", type=int, default=None, help="override (smoke/dev only)")
    ap.add_argument("--num_envs", type=int, default=None, help="override (smoke/dev only)")
    ap.add_argument("--iteration_size", type=int, default=None, help="override (smoke/dev only)")
    ap.add_argument("--eval_every", type=int, default=None, help="override (smoke/dev only)")
    ap.add_argument("--eval_width", type=int, default=24576, help="max simultaneous eval envs")
    ap.add_argument("--det_final", type=int, default=1)
    ap.add_argument("--backend", default="mjx")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--lr_horizon_frames", type=int, default=None,
                    help="override static.lr_horizon_frames (default = training length; 2000000 = the earlier 2M schedule)")
    a = ap.parse_args(argv)

    plat = check_device()
    cfg = C.load()
    static = dict(cfg["static"])
    for k in ("total_steps", "num_envs", "iteration_size", "eval_every", "eval_episodes", "lr_horizon_frames"):
        v = getattr(a, k)
        if v is not None:
            static["total_env_steps" if k == "total_steps" else k] = v
    arms, seeds = C.arm_list(cfg, a.arms), parse_seeds(a.seeds)
    out = a.out or os.path.join(REPO, "runs", a.tag, a.env)
    os.makedirs(out, exist_ok=True)
    st = A.Static(num_envs=static["num_envs"], unroll=static["iteration_size"] // static["num_envs"],
                  epochs=static["epochs"], minibatch=static["minibatch"], episode_length=static["episode_length"],
                  pi_depth=static["pi_depth"], v_depth=static["v_depth"])
    it_total = math.ceil(static["total_env_steps"] / static["iteration_size"])
    pts = eval_points(static, it_total)
    E = static["eval_episodes"]
    env, spec = D.make_env(a.env, a.backend)
    root = jax.random.PRNGKey(a.root_seed)
    info("run", env=a.env, arms=",".join(arms), seeds=a.seeds, lanes=len(arms) * len(seeds), it_total=it_total,
         eval_points=str(pts), episodes=E, platform=plat, out=out)
    fp = D.env_fingerprint(env, spec)
    info("env fingerprint", **fp)

    ls, cfgs, keys, meta = build(env, spec, cfg, arms, seeds, root, st, static)
    L = len(meta)
    n_pts = len(pts)
    snaps = {"params": jax.tree.map(lambda x: jnp.zeros((n_pts,) + x.shape, x.dtype), ls.params),
             "obs_ms": jax.tree.map(lambda x: jnp.zeros((n_pts,) + x.shape, x.dtype), ls.obs_ms)}
    diag_hist = []
    n_done = 0

    ckpt_dir = os.path.join(out, "ckpt")
    mgr = None
    if not a.no_ckpt:
        import orbax.checkpoint as ocp
        mgr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=1))
        if mgr.latest_step() is not None:
            step = mgr.latest_step()
            template = {"ls": ls, "snaps": snaps}
            restored = mgr.restore(step, args=ocp.args.StandardRestore(_flat_nonempty(template)))
            restored = _unflat(template, restored)
            ls, snaps = restored["ls"], restored["snaps"]
            n_done = int(step)
            with open(os.path.join(ckpt_dir, "diag.json")) as f:
                diag_hist = json.load(f)
            info("resumed", point=n_done, it=int(ls.it[0]))

    v_iter = jax.vmap(lambda l, c, k: A.iteration(env, l, c, st, k), in_axes=(0, 0, 0))
    kk = {k: keys[k] for k in ("env", "act", "mb")}
    aux_shape = jax.eval_shape(v_iter, ls, cfgs, kk)[1]

    @partial(jax.jit, donate_argnums=(0,))
    def run_chunk(ls, cfgs, kk, n):
        acc0 = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), aux_shape)

        def body(i, c):
            l, acc = c
            l, aux = v_iter(l, cfgs, kk)
            return l, jax.tree.map(jnp.add, acc, aux)

        return jax.lax.fori_loop(0, n, body, (ls, acc0))

    t0 = time.time()
    compiled = run_chunk.lower(ls, cfgs, kk, jnp.int32(1)).compile()
    info("compile", what="train_chunk", seconds=round(time.time() - t0, 2))

    def snapshot(ls, snaps, p):
        snaps = {"params": jax.tree.map(lambda s, x: s.at[p].set(x), snaps["params"], ls.params),
                 "obs_ms": jax.tree.map(lambda s, x: s.at[p].set(x), snaps["obs_ms"], ls.obs_ms)}
        return snaps

    if n_done == 0:
        snaps = snapshot(ls, snaps, 0)
        n_done = 1
    steady = []
    for p in range(n_done, n_pts):
        n = pts[p] - pts[p - 1]
        t = time.time()
        ls, acc = compiled(ls, cfgs, kk, jnp.int32(n))
        acc = jax.device_get(acc)
        dt = time.time() - t
        sps = L * n * static["iteration_size"] / dt
        steady.append(sps)
        snaps = snapshot(ls, snaps, p)
        d = {k: (np.asarray(v) / n).tolist() for k, v in acc.items() if k not in ("ep_ret_sum", "ep_count")}
        d["train_ep_return"] = (np.asarray(acc["ep_ret_sum"]) / np.maximum(acc["ep_count"], 1)).tolist()
        d["it"] = pts[p]
        diag_hist.append(d)
        bad = np.asarray(ls.bad)
        if bad.any():
            warn("non-finite update(s) skipped; lanes frozen at last finite step", lanes=str(np.where(bad)[0].tolist()))
        by_arm = {}
        for i, m in enumerate(meta):
            by_arm.setdefault(m["arm"], []).append(d["train_ep_return"][i])
        metric("chunk", point=p, it=pts[p], env_steps=pts[p] * static["iteration_size"], sec=round(dt, 2),
               env_steps_per_s=round(sps), **{k: round(float(np.mean(v)), 1) for k, v in by_arm.items()})
        if mgr is not None:
            import orbax.checkpoint as ocp
            mgr.save(p + 1, args=ocp.args.StandardSave(_flat_nonempty({"ls": ls, "snaps": snaps})))
            mgr.wait_until_finished()
            with open(os.path.join(ckpt_dir, "diag.json"), "w") as f:
                json.dump(diag_hist, f)
    if steady:
        metric("steady_state", env_steps_per_s=round(float(np.median(steady))), lanes=L)

    # ---------------- evaluation of all snapshots in wide batches
    ev_key = jax.vmap(lambda k, p: jax.random.fold_in(k, p), in_axes=(0, None))

    def make_eval(stochastic):
        @jax.jit
        def ev(params, obs_ms, cfgs_, keys_):
            return jax.vmap(lambda pr, o, c, k: A.evaluate(env, pr, o, c, st, k, E, stochastic))(
                params, obs_ms, cfgs_, keys_)
        return ev

    ev_sto = make_eval(True)
    per_group = max(1, a.eval_width // (L * E))
    R = np.zeros((n_pts, L, E), np.float32)
    LN = np.zeros((n_pts, L, E), np.int32)
    t = time.time()
    for g0 in range(0, n_pts, per_group):
        g = list(range(g0, min(n_pts, g0 + per_group)))
        take = lambda tree: jax.tree.map(lambda x: x[jnp.array(g)].reshape((-1,) + x.shape[2:]), tree)
        pr, om = take(snaps["params"]), take(snaps["obs_ms"])
        cf = jax.tree.map(lambda x: jnp.tile(x, (len(g),) + (1,) * (x.ndim - 1)), cfgs)
        ks = jnp.concatenate([ev_key(keys["eval"], p) for p in g], 0)
        r, ln = jax.device_get(ev_sto(pr, om, cf, ks))
        R[g] = np.asarray(r).reshape(len(g), L, E)
        LN[g] = np.asarray(ln).reshape(len(g), L, E)
        info("eval group", points=str(g), seconds=round(time.time() - t, 1))
    det = None
    if a.det_final:
        ev_det = make_eval(False)
        last = jax.tree.map(lambda x: x[-1], snaps)
        r, _ = jax.device_get(ev_det(last["params"], last["obs_ms"], cfgs, ev_key(keys["eval"], 10_000)))
        det = np.asarray(r)
    metric("eval_done", seconds=round(time.time() - t, 1))

    # ---------------- per-lane metrics
    lanes = []
    for i, m in enumerate(meta):
        curve = R[:, i].mean(1)
        se = R[:, i].std(1, ddof=1) / math.sqrt(E) if E > 1 else np.zeros(n_pts)
        score_guarded = float(curve[1:].mean()) if n_pts > 1 else float("nan")
        score = float("nan") if bool(np.asarray(ls.bad)[i]) else score_guarded   #: NaN lanes flagged
        lanes.append({
            "arm": m["arm"], "seed": m["seed"], "env": a.env, "score": score, "score_guarded": score_guarded,
            "final": float(curve[-1]),
            "det_final": None if det is None else float(det[i].mean()),
            "curve_steps": [pp * static["iteration_size"] for pp in pts], "curve": curve.tolist(),
            "curve_se_over_episodes": se.tolist(), "ep_len": LN[:, i].mean(1).tolist(),
            "nonfinite": bool(np.asarray(ls.bad)[i]),
            "diag": [{k: (v[i] if isinstance(v, list) else v) for k, v in d.items()} for d in diag_hist],
            "resolved_config": m["resolved"]})
        metric("lane", arm=m["arm"], seed=m["seed"], score=round(score, 1), final=round(float(curve[-1]), 1))
    for s in seeds:
        write_json(os.path.join(out, f"seed_{s}", "metrics.json"),
                   {"env": a.env, "tag": a.tag, "root_seed": a.root_seed, "static": static,
                    "lanes": [l for l in lanes if l["seed"] == s]})
    write_json(os.path.join(out, "run.json"), {
        "env": a.env, "tag": a.tag, "arms": arms, "seeds": seeds, "root_seed": a.root_seed, "static": static,
        "eval_points_it": pts, "env_fingerprint": fp, "steady_env_steps_per_s": float(np.median(steady)) if steady else None,
        "jax": jax.__version__, "devices": str(jax.devices()), "argv": sys.argv})
    by_arm = {}
    for l in lanes:
        by_arm.setdefault(l["arm"], []).append(l["score"])
    for k, v in by_arm.items():
        v = np.asarray(v, float)
        metric("arm_score", env=a.env, arm=k, mean=round(float(np.nanmean(v)), 1),
               se=round(float(np.nanstd(v, ddof=1) / math.sqrt(len(v))), 1) if len(v) > 1 else 0.0, n=len(v))
    info("done", out=out)


if __name__ == "__main__":
    main()
