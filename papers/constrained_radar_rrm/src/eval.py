"""Evaluation: policy factory, jitted rollouts (lax.scan over slots, vmap over scenarios / seeds) and the
 metrics. Learners act deterministically (DDPG mu(s), DQL greedy); lambda starts at lambda0 and
keeps updating by eq. 33 (A-26) inside the env for every policy.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import baselines as bl
from . import env as E
from . import physics as ph
from .algorithm import dql_q

BASELINES = ("ea", "da", "oa", "oa_blind", "oa_myopic")

# Per-slot noise key of an evaluation episode with scenario key sk. "v2" (the re-evaluation, default):
# fold_in(fold_in(sk, 1), t), as in training, so slot keys can never collide with the scenario-table keys
# fold_in(sk, 0x5A / 7 / 0xC1). "legacy": fold_in(sk, t), the full-run tree (collides at t = 7, 90, 193); kept so that
# full-run numbers and the frozen calibration stay reproducible (scripts pass --key_scheme legacy).
SLOT_KEY = "v2"


def slot_key(sk, t):
    return jax.random.fold_in(jax.random.fold_in(sk, 1), t) if SLOT_KEY == "v2" else jax.random.fold_in(sk, t)


def make_policy(kind: str, cfg, y: dict, actor=None, dql_cfg=None):
    """Returns pol(params, consts, view, es) -> raw dwells [K] (float64). consts: snr0, s_ref, delta."""
    tr = y["training"]
    cs, ls, tf = tr["obs_cost_scale"], tr["obs_lambda_scale"], tr.get("obs_transform", "affine")
    if kind == "ea":
        return lambda p, c, v, es: bl.ea(cfg, v, c["delta"])
    if kind == "da":
        return lambda p, c, v, es: bl.da(cfg, v, c["delta"])
    if kind == "oa":
        return lambda p, c, v, es: bl.oa(cfg, v, c["delta"], c["snr0"], informed=True)
    if kind == "oa_blind":
        return lambda p, c, v, es: bl.oa(cfg, v, c["delta"], c["snr0"], informed=False)
    if kind == "oa_myopic":
        return lambda p, c, v, es: bl.oa_myopic(cfg, v, c["delta"], c["snr0"])
    if kind == "ddpg":
        # per-lane obs scales (consts "cs"/"ls", full-run raw-scale variant) override the yaml A-23 constants
        return lambda p, c, v, es: actor.apply(p, E.obs_ddpg(es, cfg, c.get("cs", cs), c.get("ls", ls), tf,
                                                             c.get("obs_lit"))).astype(jnp.float64)
    if kind == "dql":
        L = dql_cfg.n_actions

        def pol(p, c, v, es):
            lvl = jnp.argmax(dql_q(dql_cfg, p, E.obs_dql(es, cfg, c.get("cs", cs), c.get("ls", ls), tf,
                                                                c.get("obs_lit"))), -1)
            return lvl.astype(jnp.float64) / (L - 1) * cfg.T0
        return pol
    raise ValueError(kind)


def rollout(cfg, lut, pol, params, consts, scen, key, T: int, lam0=None):
    """One episode of T slots. Returns (per-slot outputs dict of [T, ...], final env state)."""
    es0 = E.reset(cfg, consts.get("lam0", lam0), lit="obs_lit" in consts)   # per-lane lambda0, else Table III

    def step(es, _):
        v = E.pre(cfg, scen, es)
        tau = pol(params, consts, v, es)
        es2, out = E.post(cfg, consts, lut, scen, es, v, tau, slot_key(key, es.t))
        return es2, out

    esT, outs = jax.lax.scan(step, es0, None, length=T)
    return outs, esT


def summarize(cfg, outs, esT, scen, sat_tol: float = 0.01, gamma: float = 0.9):
    """ metrics for one episode (pure jnp)."""
    T = outs["U"].shape[0]
    t = jnp.arange(T)
    ti_sum, ti_n, never = E.tinit_stats(scen, esT)
    g = outs["g_exec"]                       # physical (executed) budget; the penalised g is outs["g"]
    viol = g > sat_tol
    viol_req = outs["g_req"] > sat_tol       # eq. 30 literal (requested, clipped dwells)
    n_pre = outs["n_trk_pre"]
    trans = jnp.concatenate([jnp.zeros(1, bool), (n_pre[:-1] == 0) & (n_pre[1:] >= 1)])
    last_tr = jax.lax.cummax(jnp.where(trans, t, -10 ** 6))
    near = (t - last_tr) <= 100
    multi = outs["n_trk"] >= 2
    has_trk = n_pre >= 1
    return dict(
        U=outs["U"].mean(), c=outs["c_sum"].mean(), wc=outs["wc_sum"].mean(),
        scan_term=cfg.beta * outs["nmiss"].mean(), nmiss=outs["nmiss"].mean(),
        tinit_sum=ti_sum, tinit_n=ti_n, never=never, tinit_cens_sum=E.tinit_cens_sum(scen, esT),
        sat_rate_req=1.0 - viol_req.mean(), g_req_mean=outs["g_req"].mean(),
        g_mean=g.mean(), g_pen_mean=outs["g"].mean(), sat_rate=1.0 - viol.mean(), sat_rate_trk=1.0 - (viol & has_trk).sum() / jnp.maximum(has_trk.sum(), 1),
        g_disc=(gamma ** t * g).sum(), starve=(outs["tau_s"] <= 0).sum(),
        viol_near_frac=(viol & near).sum() / jnp.maximum(viol.sum(), 1), n_viol=viol.sum(),
        lam_multi=(outs["lam"] * multi).sum() / jnp.maximum(multi.sum(), 1), lam_mean=outs["lam"].mean(),
        lam_max=outs["lam"].max(), trk_c_mean=(outs["c_sum"] * has_trk).sum() / jnp.maximum(has_trk.sum(), 1),
        track_share=(outs["tau"].sum(-1) / cfg.T0 * has_trk).sum() / jnp.maximum(has_trk.sum(), 1),
        n_refused=outs["n_refused"].sum(), fa=outs["fa"].sum(), overflow=scen.overflow)


def make_evaluator(cfg, lut, pol, T: int, keep_trace: bool = False, lam0=None):
    """jit(vmap over scenarios): (params, consts, scens, keys) -> summary dict of [S] (+ traces if asked)."""
    def one(params, consts, scen, key):
        outs, esT = rollout(cfg, lut, pol, params, consts, scen, key, T, lam0)
        s = summarize(cfg, outs, esT, scen)
        if keep_trace:
            s["trace"] = dict(U=outs["U"], c=outs["c_sum"], nmiss=outs["nmiss"], tau=outs["tau"], lam=outs["lam"],
                              n_trk=outs["n_trk"])
        return s
    return jax.jit(jax.vmap(one, in_axes=(None, None, 0, 0)))


def pooled_tinit(s: dict):
    """Mean T_init pooled over episodes (sum / count), plus the never-confirmed count."""
    n = jnp.maximum(s["tinit_n"].sum(), 1)
    return s["tinit_sum"].sum() / n, s["never"].sum()


def make_lut(cfg):
    return ph.make_lut(cfg.pfa)


def make_final_evaluator(cfg, lut, pol, T: int, lam0=None):
    """Like make_evaluator but keeps no per-slot outputs (for wide calibration sweeps): returns running sums
    (U, c, nmiss) and the T_init statistics. vmap over (consts, scens, keys)."""
    def one(params, consts, scen, key):
        es0 = E.reset(cfg, lam0)

        def step(carry, _):
            es, acc = carry
            v = E.pre(cfg, scen, es)
            tau = pol(params, consts, v, es)
            es2, out = E.post(cfg, consts, lut, scen, es, v, tau, slot_key(key, es.t))
            acc = dict(U=acc["U"] + out["U"], c=acc["c"] + out["c_sum"], nmiss=acc["nmiss"] + out["nmiss"])
            return (es2, acc), None

        acc0 = dict(U=jnp.zeros(()), c=jnp.zeros(()), nmiss=jnp.zeros(()))
        (esT, acc), _ = jax.lax.scan(step, (es0, acc0), None, length=T)
        ti_sum, ti_n, never = E.tinit_stats(scen, esT)
        return dict(U=acc["U"] / T, c=acc["c"] / T, nmiss=acc["nmiss"] / T, tinit_sum=ti_sum, tinit_n=ti_n,
                    never=never, tinit_cens_sum=E.tinit_cens_sum(scen, esT))
    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0)))
