"""Training: CDRL-DDPG (Alg. 2) and CDRL-DQL (Alg. 1) as ONE jitted program per algorithm, vmapped over lanes
(training seeds), lax.scan over slots in chunks; orbax checkpoint per chunk; validation every `val_every` slots on
k_valid spawn episodes; NaN guard on learner state; [METRIC] lines per chunk; compile vs steady-state timing.

Lanes of ONE run (variants x seeds) share k_lane per seed, so variants of the same algorithm and case train on the same
scenario / exploration / replay streams for seed s. DIFFERENT algorithms (and cases) do NOT: k_lane folds in ARM_ID +
CASE_OFF, so DDPG seed s and DQL seed s see different training streams; they share only the k_test evaluation draws
.

Key tree: k_lane = fold_in(fold_in(root, arm_id + case_offset), seed);
  fold_in(k_lane, 0) network init; fold_in(k_lane, 1) training scenarios (episode j: fold_in(., j) -> scenario
  draw fold_in(., 0x5A), per-slot noise fold_in(., 1)); fold_in(k_lane, 2) exploration (slot i: fold_in(., i));
  fold_in(k_lane, 3) replay sampling (slot i: fold_in(., i)).
Training episodes: 3000 slots of the spawn model with n0 ~ U{1,2,3} (A-17); lambda carried across episode
boundaries (Alg. 2 initialises lambda once, A-53); obs reset to zeros (Alg. 2 line 2).

Usage: python -m src.train --algo ddpg --case case1 --seeds 0,1,2,3,4 --slots 300000 --out runs/<name>
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, NamedTuple

from . import common  # noqa: F401  (gpu bootstrap + x64 before jax)
from .common import EnvCfg, info, load_calibration, load_yaml, metric, nan_guard, warn, write_json

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from . import algorithm as alg  # noqa: E402
from . import data  # noqa: E402
from . import env as E  # noqa: E402
from . import eval as ev  # noqa: E402

sys_drqn = alg.drqn
ARM_ID = {"ddpg": 1, "dql": 2}
CASE_OFF = {"case1": 0, "case2": 10, "case3": 20}


def case_cfg(y: dict, case: str) -> EnvCfg:
    if case == "case1":
        return EnvCfg.from_yaml(y)
    if case == "case2":
        return EnvCfg.from_yaml(y, weights="asset")
    if case == "case3":
        return EnvCfg.from_yaml(y, lbm=True)
    raise ValueError(case)


class Lane(NamedTuple):
    consts: Any       # per-lane traced constants: snr0, s_ref, delta, alpha, lam0, rs, cs, ls (full-run variants)
    es: Any           # env state
    scen: Any         # current training scenario
    ep: jax.Array     # episode index
    learner: Any      # DDPGState or drqn.QState
    buf: Any          # Replay or drqn.TraceBuffer
    k_lane: jax.Array


def new_scenario(cfg, k_lane, ep, ep_slots, circle: bool = False, cst=None):
    """cst (optional per-lane consts): n0lo/n0hi (n0 ~ U{n0lo..n0hi-1}; full-run = 1, 4; paper-literal arm 'n0zero' = 0, 1)
    and om_fast (Case III probability of the fast omega; per-omega arms = 1 or 0). Absent keys = full-run program."""
    cst = cst or {}
    k_ep = jax.random.fold_in(jax.random.fold_in(k_lane, 1), ep)
    k_s = jax.random.fold_in(k_ep, 0x5A)
    if "n0lo" in cst:
        n0 = jax.random.randint(jax.random.fold_in(k_s, 7), (), cst["n0lo"].astype(jnp.int32),
                                cst["n0hi"].astype(jnp.int32))
    else:
        n0 = jax.random.randint(jax.random.fold_in(k_s, 7), (), 1, 4)   # U{1,2,3} (A-17)
    sc = data.spawn_scenario(k_s, cfg, ep_slots, n0)
    if circle:                                                          # Case III training (A-37)
        sc = data.add_circle(sc, jax.random.fold_in(k_s, 0xC1), cfg, ep_slots, p_fast=cst.get("om_fast"))
    return sc


def slot_key(k_lane, ep):
    return jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(k_lane, 1), ep), 1)


def build(algo: str, y: dict, cfg: EnvCfg, consts: dict, n_slots: int, ep_slots: int):
    """Returns (init_lane(k_lane), lane_step(lane, i) -> (lane, metrics), policy for eval)."""
    lut = ev.make_lut(cfg)
    tr = y["training"]
    tf = tr.get("obs_transform", "affine")
    circ = cfg.lbm                                                       # Case III training adds the arc target
    warm = tr.get("random_warmup", True)
    K = cfg.K
    if algo == "ddpg":
        d = y["ddpg"]
        actor, critic, opt_a, opt_c = alg.make_ddpg(y, K, cfg.T0)
        obs_dim = 2 * K + 1
        cap = int(min(d["buffer"], n_slots))

        def init_lane(k_lane, cst):
            learner = alg.ddpg_init(jax.random.fold_in(k_lane, 0), obs_dim, K, actor, critic, opt_a, opt_c)
            return Lane(cst, E.reset(cfg, cst["lam0"], lit="obs_lit" in cst),
                        new_scenario(cfg, k_lane, 0, ep_slots, circ, cst),
                        jnp.zeros((), jnp.int32), learner, alg.replay_init(cap, obs_dim, K), k_lane)

        def lane_step(ln: Lane, i):
            es, consts = ln.es, ln.consts
            cs, ls, rs = consts["cs"], consts["ls"], consts["rs"]
            v = E.pre(cfg, ln.scen, es)
            lit = consts.get("obs_lit")
            s = E.obs_ddpg(es, cfg, cs, ls, tf, lit)
            kx = jax.random.fold_in(jax.random.fold_in(ln.k_lane, 2), i)
            mu = actor.apply(ln.learner.actor, s)
            a_expl = jnp.clip(mu + d["expl_std"] * cfg.T0 * jax.random.normal(kx, (K,)), 0.0, cfg.T0)   # A-22
            a_rand = jax.random.uniform(jax.random.fold_in(kx, 1), (K,), maxval=cfg.T0)                # A-25
            a = jnp.where((i < d["learning_starts"]) & warm, a_rand, a_expl)
            es2, out = E.post(cfg, consts, lut, ln.scen, es, v, a.astype(jnp.float64),
                              jax.random.fold_in(slot_key(ln.k_lane, ln.ep), es.t))
            s2 = E.obs_ddpg(es2, cfg, cs, ls, tf, lit)
            buf = alg.replay_add(ln.buf, s, a, (out["r"] * rs).astype(jnp.float32), s2)   # A-23 reward scale

            def do_upd(lr):
                batch = alg.replay_sample(buf, jax.random.fold_in(jax.random.fold_in(ln.k_lane, 3), i), d["batch"])
                return alg.ddpg_update(lr, batch, actor.apply, critic.apply, opt_a, opt_c, d["gamma"], d["tau_soft"])

            zinfo = dict(critic_loss=jnp.float32(0), actor_loss=jnp.float32(0), q_mean=jnp.float32(0),
                         grad_c=jnp.float32(0), grad_a=jnp.float32(0))
            learner, linfo = jax.lax.cond(i >= d["learning_starts"], do_upd, lambda lr: (lr, zinfo), ln.learner)
            return ln._replace(es=es2, learner=learner, buf=buf), dict(out=out, **linfo)

        policy = ev.make_policy("ddpg", cfg, y, actor=actor)
        return init_lane, lane_step, policy, dict(actor=actor, critic=critic)

    if algo == "dql":
        q = y["dql"]
        cap = int(min(q["buffer"], n_slots * K))
        qcfg = alg.make_dql_cfg(y, K, cap)
        L = q["n_levels"]

        def init_lane(k_lane, cst):
            learner = sys_drqn.init(jax.random.fold_in(k_lane, 0), qcfg)
            return Lane(cst, E.reset(cfg, cst["lam0"], lit="obs_lit" in cst),
                        new_scenario(cfg, k_lane, 0, ep_slots, circ, cst),
                        jnp.zeros((), jnp.int32), learner, sys_drqn.buffer_init(qcfg), k_lane)

        def lane_step(ln: Lane, i):
            es, consts = ln.es, ln.consts
            cs, ls, rs = consts["cs"], consts["ls"], consts["rs"]
            v = E.pre(cfg, ln.scen, es)
            lit = consts.get("obs_lit")
            o = E.obs_dql(es, cfg, cs, ls, tf, lit)                               # [K, 2K+2]
            kx = jax.random.fold_in(jax.random.fold_in(ln.k_lane, 2), i)
            if "eps0" in consts:      # 'eps01' arm (Table IV 'exploration rate 0.1', 'gradually reduced'): eps0 -> eps1, whole run
                eps = sys_drqn.linear_epsilon(i, consts["eps0"], consts["eps1"], float(n_slots))
            else:
                eps = sys_drqn.linear_epsilon(i, 1.0, q["eps_final"], q["eps_frac"] * n_slots)   # A-28
            greedy = jnp.argmax(alg.dql_q(qcfg, ln.learner.params, o), -1)
            rand = jax.random.randint(jax.random.fold_in(kx, 1), (K,), 0, L)
            explore = jax.random.uniform(kx, (K,)) < eps
            lvl = jnp.where(explore, rand, greedy).astype(jnp.int32)
            tau = lvl.astype(jnp.float64) / (L - 1) * cfg.T0
            es2, out = E.post(cfg, consts, lut, ln.scen, es, v, tau,
                              jax.random.fold_in(slot_key(ln.k_lane, ln.ep), es.t))
            o2 = E.obs_dql(es2, cfg, cs, ls, tf, lit)
            r = (out["r"] * rs).astype(jnp.float32)
            buf = ln.buf
            occ = es.act                                                      # the agents that acted (Alg. 1)
            for n in range(K):
                trace = dict(obs=jnp.stack([o[n], o2[n]]), act=lvl[n:n + 1], rew=r[None],
                             disc=jnp.full((1,), q["gamma"], jnp.float32), resets=jnp.zeros(2, bool), carry0=())
                buf = sys_drqn.buffer_add(buf, trace, occ[n])
            n_occ = occ.sum()

            def do_upd(st):
                infos = []
                for n in range(K):                                            # one step per occupied agent (A-30)
                    kb = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(ln.k_lane, 3), i), n)
                    batch = sys_drqn.buffer_sample(buf, kb, q["batch"])
                    st2, inf = sys_drqn.update(st, batch, qcfg)
                    apply = n < n_occ
                    st = jax.tree.map(lambda a_, b_: jnp.where(apply, a_, b_), st2, st)
                    st = sys_drqn.sync(st, apply & sys_drqn.sync_due(st, q["target_period"]))
                    infos.append(jnp.where(apply, inf["loss"], 0.0))
                return st, dict(loss=jnp.stack(infos).sum() / jnp.maximum(n_occ, 1))

            learner, linfo = jax.lax.cond(i >= q["learning_starts"], do_upd,
                                          lambda st: (st, dict(loss=jnp.float32(0))), ln.learner)
            return ln._replace(es=es2, learner=learner, buf=buf), dict(out=out, **linfo)

        policy = ev.make_policy("dql", cfg, y, dql_cfg=qcfg)
        return init_lane, lane_step, policy, dict(qcfg=qcfg)
    raise ValueError(algo)


def episode_boundary(cfg, ln: Lane, ep_slots, lam_carry: bool = True):
    """After the last slot of an episode: new scenario, env reset (obs zero), lambda carried (A-53) or reset."""
    ep = ln.ep + 1
    lit = "obs_lit" in ln.consts
    if lam_carry:
        es = E.reset(cfg, lit=lit)._replace(lam=ln.es.lam)
        if lit:
            es = es._replace(lam_prev=ln.es.lam_prev)
    else:
        es = E.reset(cfg, ln.consts["lam0"], lit=lit)
    return ln._replace(es=es, scen=new_scenario(cfg, ln.k_lane, ep, ep_slots, cfg.lbm, ln.consts), ep=ep)


CONST_KEYS = ("snr0", "s_ref", "delta", "alpha", "lam0", "rs", "cs", "ls")
# Optional per-lane constants (the re-evaluation arms). A key is carried by a run only if one of its variants sets
# it (others get the full-run-equivalent default), so runs without them compile the full-run program.
#   g_req   1 = budget term g_t on the agent's requested (clipped) dwells, eq. 30/33 literal; 0 = executed (A-13)
#   obs_lit 1 = eq. 34 literal state: chosen dwell tau_{t-1} and lambda_{t-1}; 0 = executed tau, lambda_t (A-51)
#   eps0/eps1 DQL epsilon linear eps0 -> eps1 over the whole run; default = A-28 schedule
#   n0lo/n0hi initial targets per training episode n0 ~ U{n0lo..n0hi-1} (full-run = 1, 4)
#   om_fast Case III training: probability of the fast omega per episode (full-run = 0.5)
OPT_KEYS = dict(g_req=0.0, obs_lit=0.0, eps0=1.0, eps1=0.1, n0lo=1.0, n0hi=4.0, om_fast=0.5)
OPT_GROUPS = (("eps0", "eps1"), ("n0lo", "n0hi"))


def parse_variants(spec: str, y: dict, cal: dict) -> dict:
    """'A04:;A04a:s_ref=s_ref_table6;dual50:alpha=50' -> {name: {snr0, s_ref, delta, alpha, lam0, rs, cs, ls}}.
    Defaults: calibration.json snr0 / s_ref (A-02, A-04), delta 0.9, alpha / lam0 Table III, A-23 scales."""
    tr, ob = y["training"], y["objective"]
    out = {}
    for item in [x for x in spec.split(";") if x.strip()]:
        name, _, kv = item.partition(":")
        v = dict(snr0=cal["snr0"], s_ref=cal["s_ref"], delta=0.9, alpha=ob["alpha"], lam0=ob["lambda0"],
                 rs=tr["reward_scale"], cs=tr["obs_cost_scale"], ls=tr["obs_lambda_scale"])
        for pair in [p for p in kv.split(",") if p.strip()]:
            k, _, val = pair.partition("=")
            k = k.strip()
            if k == "s_ref":
                v["s_ref"] = cal[val.strip()]
                v["s_ref_key"] = val.strip()
            else:
                assert k in CONST_KEYS or k in OPT_KEYS, k
                v[k] = float(val)
        v.setdefault("s_ref_key", "s_ref")
        out[name.strip()] = v
    used = {k for v in out.values() for k in v if k in OPT_KEYS}
    for grp in OPT_GROUPS:
        if used & set(grp):
            used |= set(grp)
    for v in out.values():
        for k in used:
            v.setdefault(k, OPT_KEYS[k])
    return out


def lane_keys(variants: dict) -> tuple:
    first = next(iter(variants.values()))
    return CONST_KEYS + tuple(k for k in OPT_KEYS if k in first)


def learner_params(algo, ln: Lane):
    return ln.learner.actor if algo == "ddpg" else ln.learner.params


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ddpg", "dql"])
    ap.add_argument("--case", default="case1")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--slots", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--val_every", type=int, default=10000)
    ap.add_argument("--val_eps", type=int, default=4)
    ap.add_argument("--learning_starts", type=int, default=None)
    ap.add_argument("--root_seed", type=int, default=20261019)
    ap.add_argument("--obs_transform", default=None, choices=[None, "affine", "log"])
    ap.add_argument("--no_random_warmup", action="store_true")
    ap.add_argument("--lam_reset", action="store_true")
    ap.add_argument("--g_on", default=None, choices=[None, "executed", "requested"])
    ap.add_argument("--actor_out_bias", type=float, default=None)
    ap.add_argument("--variants", default="A04:",
                    help="full-run lane variants 'name:k=v,k=v;name2:...' (lanes = variants x seeds, same k_lane per seed). "
                         "keys: s_ref (calibration.json key: s_ref = A-04 declared, s_ref_table6 = A-04a), alpha "
                         "(eq. 33 step; Table III 5000), lam0 (Table III 5000), rs/cs/ls (A-23 reward/cost/lambda scales)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    common.check_device()
    y = load_yaml()
    if a.learning_starts is not None:
        y["ddpg"]["learning_starts"] = y["dql"]["learning_starts"] = a.learning_starts
    if a.obs_transform:
        y["training"]["obs_transform"] = a.obs_transform
    if a.g_on:
        y["objective"]["g_on"] = a.g_on
    if a.actor_out_bias is not None:
        y["ddpg"]["actor_out_bias"] = a.actor_out_bias
    if a.no_random_warmup:
        y["training"]["random_warmup"] = False
    cal = load_calibration()
    cfg = case_cfg(y, a.case)
    n_slots = a.slots or int(y["training"]["slots"])
    ep_slots = int(y["training"]["episode_slots"])
    assert n_slots % a.chunk == 0, "--slots must be a multiple of --chunk (whole chunks only)"
    seed_list = [int(s) for s in a.seeds.split(",")]
    variants = parse_variants(a.variants, y, cal)
    lane_ids = [(vn, sd) for vn in variants for sd in seed_list]
    seeds = [sd for _, sd in lane_ids]
    consts_l = {k: jnp.stack([jnp.float64(variants[vn][k]) for vn, _ in lane_ids]) for k in lane_keys(variants)}
    consts = {k: v[0] for k, v in consts_l.items()}
    root = jax.random.PRNGKey(a.root_seed)
    k_lanes = jnp.stack([jax.random.fold_in(jax.random.fold_in(root, ARM_ID[a.algo] + CASE_OFF[a.case]), s)
                         for s in seeds])
    init_lane, lane_step, policy, nets = build(a.algo, y, cfg, consts, n_slots, ep_slots)
    lanes = jax.tree.map(jnp.copy, jax.vmap(init_lane)(k_lanes, consts_l))   # distinct buffers (donation)
    write_json(os.path.join(a.out, "config_used.json"), dict(yaml=y, args=vars(a), calibration=cal,
                                                             variants=variants, lanes=lane_ids))
    info("train start", algo=a.algo, obs_transform=y["training"].get("obs_transform", "affine"),
         random_warmup=y["training"].get("random_warmup", True), lam_reset=a.lam_reset, case=a.case, seeds=a.seeds, slots=n_slots, ep_slots=ep_slots,
         snr0=cal["snr0"], variants=a.variants, n_lanes=len(lane_ids))

    def chunk_fn(lanes, i0):
        def body(lanes, j):
            i = i0 + j
            lanes, m = jax.vmap(lane_step, in_axes=(0, None))(lanes, i)
            lanes = jax.lax.cond((i + 1) % ep_slots == 0,
                                 lambda l: jax.vmap(lambda x: episode_boundary(cfg, x, ep_slots, not a.lam_reset))(l), lambda l: l, lanes)
            o = m["out"]
            red = dict(U=o["U"], r=o["r"], g=o["g"], lam=o["lam"], n_trk=o["n_trk"].astype(jnp.float64),
                       nmiss=o["nmiss"].astype(jnp.float64), c=o["c_sum"],
                       viol=(o["g"] > 0.01).astype(jnp.float64),
                       loss=(m["critic_loss"] if a.algo == "ddpg" else m["loss"]).astype(jnp.float64))
            return lanes, red
        lanes, red = jax.lax.scan(body, lanes, jnp.arange(a.chunk))
        nan_guard(learner_params(a.algo, lanes), dict(i0=i0), tag=1, dump_dir=os.path.join(a.out, "nan"))
        return lanes, jax.tree.map(lambda x: x.mean(0), red)

    chunk_j = jax.jit(chunk_fn, donate_argnums=0)

    # validation: k_valid spawn episodes (3000 slots, n0 ~ U{1,2,3}), deterministic policy, lambda0 restart
    k_valid = jax.random.fold_in(root, 0xA11D)
    vkeys = jax.random.split(jax.random.fold_in(k_valid, CASE_OFF[a.case]), a.val_eps)
    def vscen_of(k):
        sc = data.spawn_scenario(jax.random.fold_in(k, 0x5A), cfg, ep_slots,
                                 jax.random.randint(jax.random.fold_in(k, 7), (), 1, 4))
        return data.add_circle(sc, jax.random.fold_in(k, 0xC1), cfg, ep_slots) if cfg.lbm else sc
    vscen = jax.vmap(vscen_of)(vkeys)
    val_eval = ev.make_evaluator(cfg, ev.make_lut(cfg), policy, ep_slots)
    val_all = jax.jit(jax.vmap(lambda p, c: val_eval(p, c, vscen, vkeys)))

    os.makedirs(a.out, exist_ok=True)
    import orbax.checkpoint as ocp
    ckpt_dir = os.path.abspath(os.path.join(a.out, "ckpt"))
    mgr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=1))
    start = 0
    curve = []
    if mgr.latest_step() is not None:
        start = int(mgr.latest_step())
        lanes = mgr.restore(start, args=ocp.args.StandardRestore(lanes))
        prev = os.path.join(a.out, "curve.json")
        if os.path.exists(prev):
            import json
            curve = json.load(open(prev))
        info("resumed from checkpoint", step=start)

    t_all = time.perf_counter()
    first = True
    for i0 in range(start, n_slots, a.chunk):
        t0 = time.perf_counter()
        lanes, red = chunk_j(lanes, jnp.int32(i0))
        red = jax.device_get(red)
        dt = time.perf_counter() - t0
        if first:
            info("first chunk (includes compile)", seconds=dt)
            first = False
        else:
            metric("throughput", slots_per_s_per_lane=a.chunk / dt, seconds=dt)
        done = i0 + a.chunk
        row = dict(slot=done, **{k: [float(x) for x in np.asarray(v)] for k, v in red.items()})
        if done % a.val_every == 0 or done == n_slots:
            vs = jax.device_get(val_all(learner_params(a.algo, lanes), lanes.consts))
            row["val_U"] = [float(x) for x in np.asarray(vs["U"]).mean(1)]
            row["val_c"] = [float(x) for x in np.asarray(vs["c"]).mean(1)]
            row["val_track_share"] = [float(x) for x in np.asarray(vs["track_share"]).mean(1)]
        curve.append(row)
        metric("chunk", slot=done, U=float(np.mean(red["U"])), lam=float(np.mean(red["lam"])),
               viol=float(np.mean(red["viol"])), loss=float(np.mean(red["loss"])),
               val_U=float(np.mean(row["val_U"])) if "val_U" in row else float("nan"))
        write_json(os.path.join(a.out, "curve.json"), curve)
        mgr.save(done, args=ocp.args.StandardSave(lanes))
    mgr.wait_until_finished()
    info("train done", wall_s=time.perf_counter() - t_all)
    # final params (for eval)
    import pickle
    with open(os.path.join(a.out, "params.pkl"), "wb") as f:
        pickle.dump(dict(algo=a.algo, case=a.case, seeds=seeds, lanes=lane_ids, variants=variants,
                         consts={k: np.asarray(v) for k, v in consts_l.items()},
                         params=jax.device_get(learner_params(a.algo, lanes)),
                         root_seed=a.root_seed, slots=n_slots), f)
    return lanes


if __name__ == "__main__":
    main()
