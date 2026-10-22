"""algorithm.py -- the closed-loop experiment: filter + planner + sensing + GOSPA scoring as pure jitted, lane-vmapped functions.

A "lane" is one (MC run m, clutter rate) pair; all lanes of one (configuration, algorithm) are advanced together by
`make_chunk_runner`.  Algorithms differ ONLY in the planner (Alg: cost kind, lookahead T, budgets); the filter, the sensing and the
metric are shared code.  Per step k (official main.py order):

    predict (+birth)  ->  plan sensor moves on the predicted density  ->  move  ->  measure  ->  per-sensor PMB update (LBP + TOMB)
    ->  merge  ->  prune  ->  region  ->  extract r > 0.5  ->  GOSPA against the fixed truth (full 4-D state; position-only secondary).

PRNG key tree: root = PRNGKey(seed); k_meas = fold_in(root, 1); k_plan = fold_in(root, 2);
measurement key = fold_in(fold_in(fold_in(k_meas, m), k), s); planner key = fold_in(fold_in(fold_in(k_plan, algorithm id), m), k).
The measurement key does not depend on the algorithm or the clutter rate, so the target detection/noise draws are common
random numbers across policies and clutter columns (the clutter COUNT also shares the key but depends on the rate).
"""
from __future__ import annotations

import sys
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from src import model as M
from src import data as D
from src import filter as FL
from src import gospa_jax as G
from src import planner as PL
from src.model import Cfg, Alg


class Carry(NamedTuple):
    fs: FL.FState
    spos: jnp.ndarray       # [2, 2]
    cnt: jnp.ndarray        # int32 [9] overflow / health counters (see COUNTERS)


COUNTERS = ["birth_overflow", "meas_overflow", "est_overflow", "root_overflow", "plan_overflow", "nonfinite_filter",
            "lbp_nonconverged", "plan_nan_steps", "r_gt_1"]   # r_gt_1 (informational): steps with a Bernoulli existence > 1 after the merge (D21 uncapped merge)


def init_carry(cfg: Cfg, n_lanes: int) -> Carry:
    fs = FL.empty_state(cfg)
    spos = jnp.asarray(M.SENSOR_INIT, jnp.float64)
    one = Carry(fs, spos, jnp.zeros((len(COUNTERS),), jnp.int32))
    return jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (n_lanes,) + a.shape).copy(), one)


def build_root(fs: FL.FState, cfg: Cfg, T: int) -> PL.Belief:
    """Planner root = the predicted filter density f_{k|k-1} (paper eq. 19), positions only for the CFG-O planner (D9)."""
    NP = cfg.r_cap + T
    rc = cfg.r_cap
    r = jnp.zeros((NP,)).at[:rc].set(fs.r[:rc])
    if cfg.plan_nx == 2:
        x = fs.x[:rc][:, M.POS_IDX]
        P = fs.P[:rc][:, M.POS_IDX][:, :, M.POS_IDX]
        nx = 2
    else:
        x, P, nx = fs.x[:rc], fs.P[:rc], 4
    X = jnp.zeros((NP, nx)).at[:rc].set(x)
    PP = jnp.tile(jnp.eye(nx), (NP, 1, 1)).at[:rc].set(P)
    nb = jnp.minimum(fs.n, rc)
    return PL.Belief(r, X, PP, nb.astype(jnp.int32)), fs.n > rc


def make_plan_fn(cfg: Cfg, alg: Alg, inj_u=None):
    """plan(fs_pred, spos[2,2], key, k) -> (new spos, dict(used_joint, plan_overflow, root_overflow, plan_nan)).

    inj_u: optional (UJR[K,Bj,Tm], UJE[K,Bj], UIR[K,2,Bi,Tm], UIE[K,2,Bi]) injected uniforms (differential tests only)."""
    off = jnp.asarray(M.action_offsets())
    if inj_u is not None:
        inj_u = tuple(jnp.asarray(a) for a in inj_u)

    if alg.cost == "HOLD":
        def plan(fs, spos, key, k):
            z = jnp.asarray(False)
            return spos, dict(joint=z, plan_over=z, root_over=z, plan_nan=z)
        return plan

    if alg.cost == "RANDOM":
        def plan(fs, spos, key, k):
            fe = PL.feas_joint(spos, off)
            u = jax.random.uniform(key, ())
            a = PL.kth_true(fe, jnp.floor(u * jnp.sum(fe)).astype(jnp.int32))
            z = jnp.asarray(False)
            return PL.pos_of(spos, a, off), dict(joint=z, plan_over=z, root_over=z, plan_nan=z)
        return plan

    T = alg.T
    kind = alg.cost
    joint = PL.make_mcts(cfg, 2, T, alg.b_joint, kind)
    indiv = PL.make_mcts(cfg, 1, T, alg.b_ind, kind)
    Tm = max(T - 1, 1)

    def plan(fs, spos, key, k):
        root, root_over = build_root(fs, cfg, T)
        kj, k0, k1 = jax.random.split(key, 3)

        def us(kk, B):
            a, b = jax.random.split(kk)
            Bu = PL.n_iter(cfg, B)                          # = B unless the budget counts nodes (CFG-P2)
            return jax.random.uniform(a, (Bu, Tm)), jax.random.uniform(b, (Bu,))

        if inj_u is None:
            uj = us(kj, alg.b_joint)
            ui = [us(k0, alg.b_ind), us(k1, alg.b_ind)]
        else:
            uj = (inj_u[0][k], inj_u[1][k])
            ui = [(inj_u[2][k, 0], inj_u[3][k, 0]), (inj_u[2][k, 1], inj_u[3][k, 1])]
        oj = joint(root, spos, uj[0], uj[1])
        o0 = indiv(root, spos[0][None, :], ui[0][0], ui[0][1])
        o1 = indiv(root, spos[1][None, :], ui[1][0], ui[1][1])
        use_joint = jnp.sqrt(jnp.sum((spos[0] - spos[1]) ** 2)) < M.PSI
        pos_ind = jnp.stack([o0["pos"][0], o1["pos"][0]], axis=0)
        newpos = jnp.where(use_joint, oj["pos"], pos_ind)
        d = dict(joint=use_joint,
                 plan_over=jnp.where(use_joint, oj["overflow"], o0["overflow"] | o1["overflow"]),
                 root_over=root_over,
                 plan_nan=jnp.where(use_joint, oj["any_nan"], o0["any_nan"] | o1["any_nan"]))
        return newpos, d

    return plan


def make_lane_step(cfg: Cfg, alg: Alg, seed: int = 0, nan_mode: str = "warn", inj=None, emit_est: bool = False, truth_set=None):
    """lane_step(carry, k, m, lam) -> (carry, outputs) for ONE lane; vmapped by the runner.

    inj (differential tests only): dict with z[K,2,M,2], zv[K,2,M] injected measurement sets, optional spos[K,2,2] (forced sensor
    positions: the planner is bypassed) and optional u=(UJR, UJE, UIR, UIE) injected planner uniforms."""
    tr, al, comp, cm = D.truth_arrays() if truth_set is None else truth_set     # truth_set: S-TRUTH row, stacked [Kt, ...], run m uses truth m mod Kt
    tr_j, al_j, comp_j, cm_j = (jnp.asarray(a) for a in (tr, al, comp, cm))
    n_truth_set = None if truth_set is None else tr_j.shape[0]
    plan = make_plan_fn(cfg, alg, None if inj is None else inj.get("u"))
    root = jax.random.PRNGKey(seed)
    k_meas = jax.random.fold_in(root, 1)
    k_plan = jax.random.fold_in(root, 2)
    aid = M.ALG_ID[alg.name]
    cap = {"myopic_only": alg.myopic or not alg.planner, "always": True, "never": False}[cfg.merge_cap]

    def lane_step(c: Carry, k, m, lam):
        if n_truth_set is None:
            tr_m, al_m, comp_m, cm_m = tr_j, al_j, comp_j, cm_j
        else:
            ti = m % n_truth_set
            tr_m, al_m, comp_m, cm_m = tr_j[ti], al_j[ti], comp_j[ti], cm_j[ti]
        fs_p, bover = FL.predict(c.fs, cfg)
        kp = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(k_plan, aid), m), k)
        spos, pd = plan(fs_p, c.spos, kp, k) if not (inj is not None and "spos" in inj) else (jnp.asarray(inj["spos"])[k], dict(joint=jnp.asarray(False), plan_over=jnp.asarray(False), root_over=jnp.asarray(False), plan_nan=jnp.asarray(False)))
        lam_dens = lam / (jnp.pi * M.FOV ** 2)
        km = jax.random.fold_in(jax.random.fold_in(k_meas, m), k)
        fs = fs_p
        mover = jnp.asarray(False)
        nonconv = jnp.asarray(0, jnp.int32)
        for s in range(2):
            if inj is not None:
                z, zv, ov = jnp.asarray(inj["z"])[k, s], jnp.asarray(inj["zv"])[k, s], jnp.asarray(False)
            else:
                z, zv, ov = D.gen_measurements(jax.random.fold_in(km, s), tr_m[k], al_m[k], spos[s], lam, cfg)
            mover = mover | ov
            fs, info = FL.update_sensor(fs, z, zv, spos[s], lam_dens, cfg)
            nonconv = nonconv + (~info["lbp_conv"]).astype(jnp.int32)
            if cfg.merge_each:                              # sensitivity: merge after EACH sensor update (paper SS V-C 'after each update')
                fs = FL.merge(fs, cap)
        if not cfg.merge_each:
            fs = FL.merge(fs, cap)
        ex, em, eover = FL.estimate(fs, cfg)
        act = jnp.arange(M.NB_MAX) < fs.n
        rgt1 = jnp.any(jnp.where(act, fs.r > 1.0 + 1e-12, False))
        finite = jnp.all(jnp.where(act, jnp.isfinite(fs.r), True)) & jnp.all(jnp.where(act[:, None, None], jnp.isfinite(fs.P), True))
        idx = None if cfg.gospa_space == "full" else M.POS_IDX
        tot, loc, mis, fal = G.gospa_parts(comp_m[k], cm_m[k], ex, em, M.GOSPA_C, idx)
        totp = G.gospa_parts(comp_m[k], cm_m[k], ex, em, M.GOSPA_C, M.POS_IDX)[0]
        inc = jnp.stack([bover, mover, eover, pd["root_over"], pd["plan_over"], ~finite, nonconv > 0, pd["plan_nan"], rgt1]).astype(jnp.int32)
        if nan_mode != "off":
            jax.debug.callback(_nan_cb, (~finite).astype(jnp.int32), k, m, ordered=False)
        out = dict(gospa_sq=tot, loc=loc, missed=mis, false=fal, gospa_pos_sq=totp, n_est=jnp.sum(em).astype(jnp.int32),
                   n_bern=fs.n, spos=spos, joint=pd["joint"])
        if emit_est:                                        # E3 snapshot renderer only (default off: the experiment outputs are unchanged)
            out["est"] = ex
            out["est_m"] = em
        return Carry(fs, spos, c.cnt + inc), out

    return lane_step


_NAN_SEEN = {"n": 0}


def _nan_cb(bad, k, m):
    """Host-side NaN/Inf guard (jax.debug.callback): logs the first few non-finite filter states, never aborts a vmapped batch;
    the per-run counter `nonfinite_filter` is part of every metrics.json and a non-zero value invalidates the run."""
    if int(bad):
        _NAN_SEEN["n"] += 1
        if _NAN_SEEN["n"] <= 5:
            print(f"[WARN] non-finite filter state at step {int(k)} run {int(m)} (guard count {_NAN_SEEN['n']})", flush=True)


def make_chunk_runner(cfg: Cfg, alg: Alg, seed: int = 0, nan_mode: str = "warn", emit_est: bool = False, truth_set=None):
    """run_chunk(carry, ks[nk], ms[L], lams[L]) -> (carry, outputs stacked [nk, L, ...]); jitted, vmapped over lanes."""
    lane_step = make_lane_step(cfg, alg, seed, nan_mode, emit_est=emit_est, truth_set=truth_set)
    vstep = jax.vmap(lane_step, in_axes=(0, None, 0, 0))

    @jax.jit
    def run_chunk(carry, ks, ms, lams):
        def body(c, k):
            return vstep(c, k, ms, lams)
        return lax.scan(body, carry, ks)

    return run_chunk
