"""PMBM filter core (Garcia-Fernandez, Williams, Granstrom, Svensson, TAES 2018), generic over the single-target model.

Fixed-shape, float64, jit/vmap-able.  The same code is the paper's linear-Gaussian filter (model.make_lg_model) and,
with another ``PMBMModel``.  Mathematics: the paper, Sections III-VI.

State layout (one world):
  pois_logw[P], pois_d[P]            Poisson intensity = sum_i exp(logw_i) * density_i   (undetected targets)
  trk_d[T,H], trk_r[T,H], trk_alive[T,H]
                                     Bernoulli components: T track slots x H single-target hypotheses
  gidx[G,T] int32, glogW[G]          global hypotheses: gidx = index h of the single-target hypothesis used by track t
                                     in global hypothesis g, -1 = track absent; glogW = log weight (-inf = invalid row)
Single-target weights are NOT stored: for the standard model the global weight is the product of per-track factors,
so the global log weight alone carries everything (cost ratio of eq. (47)).

One filter step = predict -> update (-> estimators before pruning) -> prune/compact (inside update).
Order inside update follows Alg. 1 + the official code: estimate BEFORE pruning; prune order:
Poisson, global-hypothesis weight prune (strict >) + top N_h + renormalise, Bernoulli existence prune (set absent),
per-track cap (extension, not in the paper), merge duplicate global hypotheses (sum weights), drop unreferenced
single-target hypotheses and empty tracks.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from src.kbest import kbest_assign, pm_fdt, pm_tiny, pm_map_alive, pm_nonzero, pm_scatter


_NEG = -jnp.inf


# --------------------------------------------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------------------------------------------
class PMBMModel(NamedTuple):
    """Single-target model.  All callables act on ONE density / ONE measurement (the core vmaps them).

    predict(d, ctx) -> d'                      motion / transition (Poisson comps and Bernoullis alike)
    survive(d, ctx) -> p_s (scalar)            survival probability, evaluated on the pre-predict density
    det(d, z, ctx) -> (logm, d_post, gated)    logm = log  integral p_d(x) l(z|x) p(x) dx ;  d_post = normalised
                                               posterior density of a detection by z ;  gated = bool (ellipsoidal /
                                               predictive-mass gate: pairs that fail get weight 0)
    miss(d, ctx) -> (m0, d_miss)               m0 = integral (1 - p_d(x)) p(x) dx ; d_miss = normalised density of a miss
    merge(ds, logw) -> d                       moment-match a stack of densities with log weights (-inf = excluded)
    birth(ctx) -> (logw[B], ds[B])             Poisson birth components appended at every predict
    log_clutter(z, ctx) -> scalar              log c(z) (clutter intensity, NOT normalised)
    point(d) -> vector                         point estimate reported by the estimators (mean)
    init_density() -> d                        a VALID template density (used to fill padded slots; must be finite)
    """
    predict: Callable
    survive: Callable
    det: Callable
    miss: Callable
    merge: Callable
    birth: Callable
    log_clutter: Callable
    point: Callable
    init_density: Callable


class PMBMConfig(NamedTuple):
    """Static (hashable) filter dimensions and dials."""
    T: int = 128                 # track slots                                     ASSUMED (monitored by overflow counters)
    H: int = 64                  # single-target hypotheses kept per track         ASSUMED
    Nh: int = 200                # N_h: max global hypotheses = G                  paper §VII
    P: int = 16                  # Poisson components                              ASSUMED
    M: int = 40                  # max measurements per scan                       ASSUMED
    prune_w: float = 1e-4        # global-hypothesis weight prune (strict >)       NOT in the paper (paper path = 0, model.paper_pmbm_config); authors' HEAD code 1e-4
    prune_pois: float = 1e-5     # Poisson weight prune                            paper §VII
    prune_r: float = 1e-5        # Bernoulli existence prune                       paper §VII
    h_cap: int = 0               # per-track cap (0 -> H)                          extension, not in the paper
    backend: str = "host"        # k-best backend 'host' | 'jax'                   see kbest.py
    queue_cap: int = 256         # k-best queue capacity for backend 'jax'
    estimators: Tuple[int, ...] = (1, 2, 3)
    est_gamma: float = 0.4       # Estimator 1 existence threshold                 paper §VII
    est_E: int = 16              # max estimate set size reported
    pred_batch: int = 0          # option not used by the paper's experiments (0 = off)
    k_per_parent: int = 0        # 0 = paper rule k_g = ceil(N_h W_g); >0 = exactness mode: k_g = k_per_parent (tests)


class PMBMState(NamedTuple):
    pois_logw: Any
    pois_d: Any
    trk_d: Any
    trk_r: Any
    trk_alive: Any
    gidx: Any
    glogW: Any
    ovf: Any          # int32[6] cumulative overflow counters, see OVF_NAMES
    trk_q: Any = None  # option not used by the paper's experiments (None on the float64 paper path)


OVF_NAMES = ("pois_evict", "slot_full", "cap_rows_dropped", "kbest_ovf", "meas_ignored", "child_trunc")


PM_QFORM = None      # option not used by the paper's experiments (float32 only)


def pm_qform():
    """Option not used by the paper's experiments (float32 only)."""
    return (not jax.config.jax_enable_x64) if PM_QFORM is None else bool(PM_QFORM)


def _tmap(f, *trees):
    return jax.tree_util.tree_map(f, *trees)


def _stack_density(model: PMBMModel, *lead):
    d0 = model.init_density()
    return _tmap(lambda a: jnp.broadcast_to(a, tuple(lead) + a.shape).astype(a.dtype), d0)


def init_state(model: PMBMModel, cfg: PMBMConfig, pois=None, tracks=None) -> PMBMState:
    """Empty state, or with an initial Poisson ``pois=(logw[n], ds[n])`` and initial single-hypothesis Bernoullis
    ``tracks=[(r, d), ...]`` (all in one global hypothesis of weight 1)."""
    T, H, P, G = cfg.T, cfg.H, cfg.P, cfg.Nh
    pois_logw = jnp.full((P,), _NEG, dtype=pm_fdt())            # STRONG-typed (a weak-typed leaf makes a per-epoch jit compile twice)
    pois_d = _stack_density(model, P)
    if pois is not None:
        lw, ds = pois
        n = lw.shape[0]
        pois_logw = pois_logw.at[:n].set(lw, mode="clip")
        pois_d = _tmap(lambda a, x: a.at[:n].set(x, mode="clip"), pois_d, ds)
    trk_d = _stack_density(model, T, H)
    trk_r = jnp.zeros((T, H))
    trk_q = jnp.ones((T, H)) if pm_qform() else None
    trk_alive = jnp.zeros((T, H), bool)
    gidx = jnp.full((G, T), -1, jnp.int32)
    glogW = jnp.full((G,), _NEG, dtype=pm_fdt()).at[0].set(0.0, mode="clip")
    if tracks:
        for t, (r, d) in enumerate(tracks):
            trk_r = trk_r.at[t, 0].set(r, mode="clip")
            trk_d = _tmap(lambda a, x: a.at[t, 0].set(x, mode="clip"), trk_d, d)
            trk_alive = trk_alive.at[t, 0].set(True, mode="clip")
            gidx = gidx.at[0, t].set(0, mode="clip")
    return PMBMState(pois_logw, pois_d, trk_d, trk_r, trk_alive, gidx, glogW, jnp.zeros(len(OVF_NAMES), jnp.int32), trk_q)


# --------------------------------------------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------------------------------------------
def pmbm_predict(state: PMBMState, model: PMBMModel, cfg: PMBMConfig, ctx=None) -> PMBMState:
    """Poisson: PHD prediction + birth; each Bernoulli: MeMBer prediction; hypotheses unchanged (paper Section V-B)."""
    ps_p = jax.vmap(model.survive, (0, None))(state.pois_d, ctx)
    d_p = jax.vmap(model.predict, (0, None))(state.pois_d, ctx)
    logw = state.pois_logw + jnp.log(ps_p)
    ps_t = jax.vmap(jax.vmap(model.survive, (0, None)), (0, None))(state.trk_d, ctx)
    if cfg.pred_batch:
        T0, H0 = state.trk_r.shape
        flat = _tmap(lambda a: a.reshape((T0 * H0,) + a.shape[2:]), state.trk_d)
        trk_d = _tmap(lambda a: a.reshape((T0, H0) + a.shape[1:]), pm_map_alive(lambda d: model.predict(d, ctx), flat, state.trk_alive.reshape(-1), cfg.pred_batch))
    else:
        trk_d = jax.vmap(jax.vmap(model.predict, (0, None)), (0, None))(state.trk_d, ctx)
    trk_r = state.trk_r * ps_t
    trk_q = None if state.trk_q is None else jnp.minimum((1.0 - ps_t) + ps_t * state.trk_q, 1.0)        # 1 - r ps = (1 - ps) + ps q
    bw, bd = model.birth(ctx)
    ovf = state.ovf
    for b in range(int(bw.shape[0])):          # static, tiny loop
        slot = jnp.argmin(logw)
        evict = jnp.isfinite(logw[slot]) & jnp.isfinite(bw[b])
        ovf = ovf.at[0].add(evict.astype(jnp.int32), mode="clip")
        put = jnp.isfinite(bw[b])
        logw = logw.at[slot].set(jnp.where(put, bw[b], logw[slot]), mode="clip")
        d_p = _tmap(lambda a, x: a.at[slot].set(jnp.where(put, x[b], a[slot]), mode="clip"), d_p, bd)
    return state._replace(pois_logw=logw, pois_d=d_p, trk_d=trk_d, trk_r=trk_r, trk_q=trk_q, ovf=ovf)


# --------------------------------------------------------------------------------------------------------------
# estimators (computed on the pre-pruning children)
# --------------------------------------------------------------------------------------------------------------
def _compact(mask, x, E):
    """Pack the True entries of mask[T] (and rows of x[T,dim]) to the front, keep E."""
    order = jnp.argsort(~mask, stable=True)[:E]
    return mask[order], x[order]


def _compact_n(mask, x, E):
    """_compact plus the number of selected entries that did not fit (estimate-set overflow)."""
    return _compact(mask, x, E), jnp.maximum(jnp.sum(mask).astype(jnp.int32) - E, 0)


def _gather_t(cand, code):
    """cand leaves [T,Hc,...]; code int [T] -> leaves [T,...]"""
    return _tmap(lambda a: jax.vmap(lambda at, c: at[c])(a, code), cand)


def estimate1(codes, logW, cand_r, cand_d, model, cfg):
    j = jnp.argmax(logW)
    code = codes[j]
    present = code >= 0
    c = jnp.maximum(code, 0)
    r = jax.vmap(lambda rt, ci: rt[ci])(cand_r, c)
    mask = present & (r > cfg.est_gamma)
    pts = jax.vmap(model.point)(_gather_t(cand_d, c))
    return _compact_n(mask, pts, cfg.est_E)


def estimate3(codes, logW, cand_r, cand_d, model, cfg):
    T = codes.shape[1]
    c = jnp.maximum(codes, 0)
    r = jax.vmap(lambda rt, ci: rt[ci], (0, 1), 1)(cand_r, c)           # [C,T]
    r = jnp.where(codes >= 0, r, 0.0)
    pres = codes >= 0
    ex = r >= 0.5
    sc = jnp.sum(jnp.where(pres, jnp.where(ex, jnp.log(r), jnp.log1p(-r)), 0.0), axis=1)
    score = logW + sc
    j = jnp.argmax(score)
    cj = c[j]
    pts = jax.vmap(model.point)(_gather_t(cand_d, cj))
    return _compact_n(pres[j] & ex[j], pts, cfg.est_E)


def _poisson_binomial(r):
    """r[C,T] -> pmf[C,T+1] by sequential convolution (exact)."""
    C, T = r.shape
    pmf0 = jnp.zeros((C, T + 1)).at[:, 0].set(1.0, mode="clip")

    def body(pmf, rt):
        shifted = jnp.concatenate([jnp.zeros((C, 1)), pmf[:, :-1]], axis=1)
        return pmf * (1 - rt)[:, None] + shifted * rt[:, None], None

    pmf, _ = lax.scan(body, pmf0, r.T)
    return pmf


def estimate2(codes, logW, cand_r, cand_d, model, cfg):
    T = codes.shape[1]
    c = jnp.maximum(codes, 0)
    r = jax.vmap(lambda rt, ci: rt[ci], (0, 1), 1)(cand_r, c)
    r = jnp.where(codes >= 0, r, 0.0)
    W = jnp.exp(logW)                                                   # normalised pre-prune weights
    pmf = jnp.sum(W[:, None] * _poisson_binomial(r), axis=0)            # eq. (48), MBM only
    nstar = jnp.argmax(pmf)
    order = jnp.argsort(-r, axis=1, stable=True)
    rs = jnp.take_along_axis(r, order, axis=1)
    top = jnp.arange(T)[None, :] < nstar
    sc = jnp.sum(jnp.where(top, jnp.log(rs), jnp.log1p(-rs)), axis=1)
    score = logW + sc
    j = jnp.argmax(score)
    cj = c[j]
    pts = jax.vmap(model.point)(_gather_t(cand_d, cj))
    sel = jnp.zeros((T,), bool).at[order[j]].set(top[0], mode="clip")
    sel = sel & (codes[j] >= 0)
    return _compact_n(sel, pts, cfg.est_E), pmf


# --------------------------------------------------------------------------------------------------------------
# update
# --------------------------------------------------------------------------------------------------------------
def pmbm_update(state: PMBMState, z, zvalid, model: PMBMModel, cfg: PMBMConfig, ctx=None, prune_w=None):
    """One PMBM update with measurement stack ``z`` (leaves [M,...]) and ``zvalid`` bool[M] (valid ones need not be
    first).  ``prune_w`` optionally overrides cfg.prune_w with a traced scalar.  Returns (new_state, out) where out holds the three estimates (before pruning) and diagnostics."""
    T, H, G, P, M, Nh = cfg.T, cfg.H, cfg.Nh, cfg.P, cfg.M, cfg.Nh
    Hc = H * (M + 1)
    kmax = cfg.k_per_parent or Nh
    C = G * kmax if cfg.k_per_parent else Nh + G
    hcap = cfg.h_cap if cfg.h_cap else H
    assert hcap <= H, 'h_cap must be <= H'
    tidx = jnp.arange(T)

    # ---- 1. Poisson part: undetected update + one new Bernoulli per measurement (eq. 16-24) ------------------------
    det_pm = jax.vmap(lambda d: jax.vmap(lambda zz: model.det(d, zz, ctx))(z))(state.pois_d)
    logm_p, dpost_p, gated_p = det_pm                                      # [P,M], [P,M,..], [P,M]
    pvalid = jnp.isfinite(state.pois_logw)
    lw = jnp.where(gated_p & pvalid[:, None] & zvalid[None, :], state.pois_logw[:, None] + logm_p, _NEG)   # [P,M]
    log_e = jax.scipy.special.logsumexp(lw, axis=0)                        # log e(z)
    has_e = jnp.isfinite(log_e)
    logc = jax.vmap(lambda zz: model.log_clutter(zz, ctx))(z)
    log_rho = jnp.logaddexp(log_e, logc)                                   # log rho_p(z) = log(c + e)
    r_new = jnp.where(has_e, jnp.exp(log_e - log_rho), 0.0)
    lw_safe = jnp.where(has_e[None, :], lw, jnp.where(jnp.arange(P)[:, None] == 0, 0.0, _NEG))
    d_new = jax.vmap(lambda dp, w: model.merge(dp, w), (1, 1))(dpost_p, lw_safe)       # leaves [M,..]
    m0_p, dmiss_p = jax.vmap(lambda d: model.miss(d, ctx))(state.pois_d)
    pois_logw = jnp.where(pvalid, state.pois_logw + jnp.log(jnp.maximum(m0_p, pm_tiny())), _NEG)
    pois_d = dmiss_p

    # ---- 2. old Bernoullis: miss / detection children (eq. 28-31, 46) -------------------------------------------
    det_th = jax.vmap(jax.vmap(lambda d: jax.vmap(lambda zz: model.det(d, zz, ctx))(z)))(state.trk_d)
    logm_t, dpost_t, gated_t = det_th                                      # [T,H,M], [T,H,M,..], [T,H,M]
    m0_t, dmiss_t = jax.vmap(jax.vmap(lambda d: model.miss(d, ctx)))(state.trk_d)
    r = state.trk_r
    qf = state.trk_q is not None
    q0 = state.trk_q if qf else 1.0 - r                                    # (float64 reference: 1 - r, the same operations as before)
    rho0 = jnp.maximum(q0 + r * m0_t, pm_tiny())
    log_rho0 = jnp.log(rho0)                                               # [T,H]
    r_miss = r * m0_t / rho0
    log_rho1 = jnp.where(gated_t & state.trk_alive[:, :, None] & zvalid[None, None, :],
                         jnp.log(jnp.maximum(r, pm_tiny()))[:, :, None] + logm_t, _NEG)       # [T,H,M]

    # candidate single-target hypotheses: index = h*(M+1) + ev ; ev 0 = miss, ev 1+m = detected by z_m
    cand_r = jnp.concatenate([r_miss[:, :, None], jnp.ones((T, H, M))], axis=2).reshape(T, Hc)
    if qf:
        cand_q = jnp.concatenate([(q0 / rho0)[:, :, None], jnp.zeros((T, H, M))], axis=2).reshape(T, Hc)         # a detected target: q = 0 exactly
    cand_d = _tmap(lambda a_miss, a_det: jnp.concatenate([a_miss[:, :, None], a_det], axis=2)
                   .reshape((T, Hc) + a_miss.shape[2:]), dmiss_t, dpost_t)

    # ---- 3. slots for the new Bernoullis (one per measurement the Poisson part could have produced) --------------
    free = ~jnp.any(state.trk_alive, axis=1)
    need = zvalid & has_e
    nrank = jnp.cumsum(need) - 1
    free_slots = pm_nonzero(free, T, T)
    nfree = jnp.sum(free)
    slot = jnp.where(need & (nrank < nfree), free_slots[jnp.clip(nrank, 0, T - 1)], T)    # T = no slot
    slot_full = jnp.sum(need & (slot >= T))
    cand_r = pm_scatter(cand_r, (slot, 0), r_new)
    if qf:
        q_new = jnp.where(has_e, jnp.exp(logc - log_rho), 1.0)             # c / (c + e) without the cancellation of 1 - r_new
        cand_q = pm_scatter(cand_q, (slot, 0), q_new)
    cand_d = _tmap(lambda a, x: pm_scatter(a, (slot, 0), x), cand_d, d_new)

    # ---- 4. children of every parent global hypothesis (eq. 47 + Murty) ---------------------------------------------
    gvalid = jnp.isfinite(state.glogW)
    hg = state.gidx
    present = (hg >= 0) & gvalid[:, None]
    hc = jnp.maximum(hg, 0)
    lr0_g = log_rho0[tidx[None, :], hc]                                    # [G,T]
    lr1_g = log_rho1[tidx[None, :], hc]                                    # [G,T,M]
    cost_old = jnp.where(present[:, :, None] & jnp.isfinite(lr1_g), -(lr1_g - lr0_g[:, :, None]), jnp.inf)
    cost_old = jnp.transpose(cost_old, (0, 2, 1))                          # [G,M,T]
    cost_new = jnp.where(zvalid, -log_rho, 0.0)
    logbase = state.glogW + jnp.sum(jnp.where(present, lr0_g, 0.0), axis=1)
    Wp = jnp.where(gvalid, jnp.exp(state.glogW - jax.scipy.special.logsumexp(jnp.where(gvalid, state.glogW, _NEG))), 0.0)
    kvec = jnp.where(gvalid, jnp.minimum(jnp.ceil(Nh * Wp).astype(jnp.int32), kmax), 0)
    if cfg.k_per_parent:
        kvec = jnp.where(gvalid, kmax, 0)
    assign, kcost, kovf = kbest_assign(cost_old, cost_new, zvalid, kvec, kmax, backend=cfg.backend,
                                       **({"queue_cap": cfg.queue_cap} if cfg.backend == "jax" else {}))            # [G,Nh,M], [G,Nh], [G]
    child_lw = jnp.where(jnp.isfinite(kcost), logbase[:, None] - kcost, _NEG).reshape(G * kmax)
    n_valid_children = jnp.sum(jnp.isfinite(child_lw))
    top_lw, top_i = lax.top_k(child_lw, C)                                 # sorted descending
    par = top_i // kmax
    asg = assign.reshape(G * kmax, M)[top_i]                                 # [C,M]
    cvalid = jnp.isfinite(top_lw)
    logW = top_lw - jax.scipy.special.logsumexp(jnp.where(cvalid, top_lw, _NEG))
    logW = jnp.where(cvalid, logW, _NEG)

    # codes [C,T]: -1 absent, else candidate index into [T,Hc]
    mt = jnp.full((C, T), -1, jnp.int32)
    mt = pm_scatter(mt, (jnp.arange(C)[:, None], jnp.where(asg >= 0, asg, T)), jnp.where(asg >= 0, jnp.arange(M)[None, :], -1).astype(jnp.int32), "max")
    hpar = hg[par]                                                         # [C,T]
    codes = jnp.where(hpar >= 0, hpar * (M + 1) + mt + 1, -1).astype(jnp.int32)
    use_new = (asg == -1) & (slot[None, :] < T)                            # [C,M]
    codes = pm_scatter(codes, (jnp.arange(C)[:, None], jnp.where(use_new, slot[None, :], T)), 0)
    codes = jnp.where(cvalid[:, None], codes, -1)

    # ---- 5. estimators on the pre-prune hypotheses ------------------------------------------------------------------
    pw = cfg.prune_w if prune_w is None else prune_w      # may be a traced scalar
    out = {}
    est_over = jnp.int32(0)          # selected tracks that did not fit the E-slot estimate set (must read 0)
    for e in cfg.estimators:
        if e == 1:
            out["est1"], o_ = estimate1(codes, logW, cand_r, cand_d, model, cfg)
        elif e == 3:
            out["est3"], o_ = estimate3(codes, logW, cand_r, cand_d, model, cfg)
        elif e == 2:
            (out["est2"], o_), out["card_pmf"] = estimate2(codes, logW, cand_r, cand_d, model, cfg)
        est_over = est_over + o_
    out["est_over"] = est_over

    # ---- 6. pruning / compaction -----------------------------------------------------------------------------------
    W = jnp.exp(logW)
    ok = cvalid & (W > pw)
    # rows are sorted by weight, so ok is a prefix; keep the first G (= N_h)
    ok = ok & (jnp.cumsum(ok) <= G)
    # guarantee at least one hypothesis survives (the best one) even if all fail the threshold
    ok = ok.at[0].set(ok[0] | cvalid[0], mode="clip")
    keep_n = jnp.sum(ok)
    sel = pm_nonzero(ok, G, C - 1)
    rowv = jnp.arange(G) < keep_n
    pcodes = jnp.where(rowv[:, None], codes[sel], -1)
    plw = jnp.where(rowv, logW[sel], _NEG)
    plw = plw - jax.scipy.special.logsumexp(plw)
    # Bernoulli existence prune -> absent
    rr = jax.vmap(lambda rt, ci: rt[ci], (0, 1), 1)(cand_r, jnp.maximum(pcodes, 0))
    pcodes = jnp.where((pcodes >= 0) & (rr < cfg.prune_r), -1, pcodes)
    # per-track cap by marginal weight (extension; paper path: never binds, counted)
    Wp2 = jnp.exp(plw)
    mw = jnp.zeros((T, Hc)).at[jnp.broadcast_to(tidx[None, :], pcodes.shape), jnp.maximum(pcodes, 0)].add(
        jnp.where(pcodes >= 0, Wp2[:, None], 0.0), mode="clip")
    kk = min(hcap, Hc)
    topv, topi = lax.top_k(mw, kk)
    keepc = jnp.zeros((T, Hc), bool).at[tidx[:, None], topi].set(topv > 0, mode="clip")
    row_ok = jnp.all((pcodes < 0) | keepc[tidx[None, :], jnp.maximum(pcodes, 0)], axis=1)
    cap_drop = jnp.sum(rowv & ~row_ok)
    plw = jnp.where(row_ok, plw, _NEG)
    plw = plw - jax.scipy.special.logsumexp(plw)
    pcodes = jnp.where(row_ok[:, None], pcodes, -1)
    # merge duplicates (sum weights, keep first occurrence)
    live = jnp.isfinite(plw)
    eq = jnp.all(pcodes[:, None, :] == pcodes[None, :, :], axis=2) & live[:, None] & live[None, :]
    first = live & ~jnp.any(jnp.tril(eq, -1), axis=1)
    mlw = jax.scipy.special.logsumexp(jnp.where(eq, plw[None, :], _NEG), axis=1)
    mlw = jnp.where(first, mlw, _NEG)
    order = jnp.argsort(-mlw, stable=True)
    new_lw = mlw[order]
    new_codes = pcodes[order]
    new_lw = new_lw - jax.scipy.special.logsumexp(new_lw)
    live2 = jnp.isfinite(new_lw)
    new_codes = jnp.where(live2[:, None], new_codes, -1)
    # relabel single-target hypotheses per track
    referenced = jnp.zeros((T, Hc), jnp.int32).at[jnp.broadcast_to(tidx[None, :], new_codes.shape),
                                                  jnp.maximum(new_codes, 0)].max((new_codes >= 0).astype(jnp.int32), mode="clip") > 0
    rank = jnp.cumsum(referenced, axis=1) - 1
    new_h = jnp.where(new_codes >= 0, rank[tidx[None, :], jnp.maximum(new_codes, 0)], -1).astype(jnp.int32)
    src = jax.vmap(lambda m: pm_nonzero(m, H, 0))(referenced)          # [T,H]
    cnt = jnp.sum(referenced, axis=1)
    trk_alive = jnp.arange(H)[None, :] < cnt[:, None]
    trk_d = _tmap(lambda a: jax.vmap(lambda at, s: at[s])(a, src), cand_d)
    trk_r = jax.vmap(lambda rt, s: rt[s])(cand_r, src) * trk_alive
    trk_q = jnp.where(trk_alive, jax.vmap(lambda qt, s: qt[s])(cand_q, src), 1.0) if qf else None
    # Poisson pruning
    pk = pois_logw >= jnp.log(cfg.prune_pois)
    pois_logw = jnp.where(pk, pois_logw, _NEG)

    ovf = state.ovf + jnp.array([0, slot_full, cap_drop, jnp.sum(kovf & gvalid), 0,
                                 jnp.maximum(n_valid_children - C, 0)], jnp.int32)
    ns = PMBMState(pois_logw, pois_d, trk_d, trk_r, trk_alive, new_h, new_lw, ovf, trk_q)
    out["n_children"] = n_valid_children
    out["n_hyp"] = jnp.sum(live2)
    out["n_tracks"] = jnp.sum(cnt > 0)
    out["max_h"] = jnp.max(cnt)
    out["ovf"] = ovf
    return ns, out


# --------------------------------------------------------------------------------------------------------------
# helpers: marginals, scan driver
# --------------------------------------------------------------------------------------------------------------
def track_marginals(state: PMBMState, cfg: PMBMConfig):
    """Per track slot: marginal hypothesis weights mw[T,H] (sum over global hypotheses of the weight of those that
    use single-target hypothesis h, i.e. P(track exists in hypothesis AND uses h)), marginal existence
    r_marg[T] = sum_h mw[T,h] r[T,h] (probability the Bernoulli target exists), and used[T]."""
    T, H = cfg.T, cfg.H
    W = jnp.where(jnp.isfinite(state.glogW), jnp.exp(state.glogW), 0.0)
    idx = state.gidx
    mw = jnp.zeros((T, H)).at[jnp.broadcast_to(jnp.arange(T)[None, :], idx.shape), jnp.maximum(idx, 0)].add(
        jnp.where(idx >= 0, W[:, None], 0.0), mode="clip")
    r_marg = jnp.sum(mw * state.trk_r, axis=1)
    return mw, r_marg, jnp.any(state.trk_alive, axis=1)


def _pmbm_nan_cb(bad):
    if bool(np.any(np.asarray(bad))):
        print("[ERROR] pmbm: NaN in hypothesis weights / existence probabilities", flush=True)


def run_filter(model: PMBMModel, cfg: PMBMConfig, z, zvalid, ctx=None, state0=None, predict_first=True,
               ctx_seq=None, nan_guard=True):
    """lax.scan over K steps for one world.  z leaves [K,M,..]; zvalid [K,M].  predict_first=False skips the predict
    at k=1 (official-code initial prior, sensitivity row).  ctx_seq (leaves [K,..]) overrides ctx per step.
    Returns (final_state, outs) with outs dict stacked over steps."""
    if state0 is None:
        state0 = init_state(model, cfg)
    K = zvalid.shape[0]

    def step(carry, inp):
        st, k = carry
        zk, zvk, ck = inp
        c = ck if ctx_seq is not None else ctx
        st = lax.cond(jnp.logical_or(predict_first, k > 0), lambda s: pmbm_predict(s, model, cfg, c), lambda s: s, st)
        st, out = pmbm_update(st, zk, zvk, model, cfg, c)
        if nan_guard:   # NaN guard (valid log-weights are finite or -inf, never NaN)
            jax.debug.callback(_pmbm_nan_cb, jnp.any(jnp.isnan(st.glogW)) | jnp.any(jnp.isnan(st.trk_r)))
        return (st, k + 1), out

    cs = ctx_seq if ctx_seq is not None else jnp.zeros((K,))
    (st, _), outs = lax.scan(step, (state0, jnp.int32(0)), (z, zvalid, cs))
    return st, outs
