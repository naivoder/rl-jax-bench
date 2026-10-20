"""Track-while-scan environment, pure functions in float64.

One slot = pre() (presence, EKF prediction, weights: everything a policy may look at) -> a policy picks dwells ->
post() (execute dwells, track updates, scan, 3-of-4 initiation, score, dual update eq. 33, exits for t+1).

Randomness per slot is keyed by (scenario key, slot) only and indexed by TARGET, never by policy-dependent
quantities, so every policy sees the same measurement-noise / detection / false-alarm draws (common random numbers,
key tree in src/train.py).

Declared readings used here (assumption labels, see the README): A-01 (eq. 6 division), A-03, A-06 (tracked targets ignored by
initiation), A-08 (eq. 33 '+'), A-10, A-11, A-12, A-13, A-14, A-15, A-44..A-47, and the pilot amendments A-49..A-53.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import physics as ph
from .common import EnvCfg
from .data import Scenario, truth


class EnvState(NamedTuple):
    t: jax.Array          # [] int32 slot index
    act: jax.Array        # [K] bool track slot occupied
    tgt: jax.Array        # [K] int32 true target index held by the slot (perfect association, A-15)
    x: jax.Array          # [K, 4] posterior state
    P: jax.Array          # [K, 4, 4] posterior covariance
    prev_wc: jax.Array    # [K] w^n c^n of the last slot (observation, eq. 34), 0 for empty slots
    prev_tau: jax.Array   # [K] executed dwell of the last slot (s), 0 for empty slots
    lam: jax.Array        # [] dual variable lambda_t (applied to this slot's reward)
    buf: jax.Array        # [M] int32 initiation-chain state per target (0 E, 1 V, 2 VV, 3 VX, 4 VVX, 5 VXV)
    b_last: jax.Array     # [M, 2] last valid scan measurement (Cartesian)
    b_prev: jax.Array     # [M, 2] the one before
    k_last: jax.Array     # [M] int32 slot of b_last
    k_prev: jax.Array     # [M] int32 slot of b_prev
    conf_slot: jax.Array  # [M] int32 first confirmation slot (-1 = never)
    seen: jax.Array       # [M] bool target was present at least once
    lam_prev: jax.Array = None  # [] lambda_{t-1} (only for the eq. 34-literal observation arm; None =
                                # absent from the pytree, so full-run programs are unchanged)


class View(NamedTuple):
    """What a decision may use at slot t (baselines read the filter's prior; learners read only the obs)."""
    present: jax.Array    # [M] (ground truth: used by the environment, never by a policy)
    pos: jax.Array        # [M, 2] truth (environment only)
    act: jax.Array        # [K]
    xp: jax.Array         # [K, 4] prior estimates
    Pp: jax.Array         # [K, 4, 4] prior covariances
    w: jax.Array          # [K] weights from the prior (eq. 35 in Case II; 1 otherwise)
    r_hat: jax.Array      # [K] estimated range |xp|
    factor_hat: jax.Array  # [K] Case IV noise factor at the ESTIMATED position (OA-informed, A-39)


def reset(cfg: EnvCfg, lam0=None, lit: bool = False) -> EnvState:
    """lit=True adds lambda_{t-1} to the state (eq. 34-literal observation arm); lambda_{-1} := lambda_0."""
    K, M = cfg.K, cfg.M
    lam0 = cfg.lambda0 if lam0 is None else lam0
    lp = jnp.asarray(lam0, jnp.float64) if lit else None
    return EnvState(lam_prev=lp, t=jnp.zeros((), jnp.int32), act=jnp.zeros(K, bool), tgt=jnp.zeros(K, jnp.int32),
                    x=jnp.zeros((K, 4)), P=jnp.tile(jnp.eye(4), (K, 1, 1)), prev_wc=jnp.zeros(K),
                    prev_tau=jnp.zeros(K), lam=jnp.asarray(lam0, jnp.float64), buf=jnp.zeros(M, jnp.int32),
                    b_last=jnp.zeros((M, 2)), b_prev=jnp.zeros((M, 2)), k_last=jnp.zeros(M, jnp.int32),
                    k_prev=jnp.zeros(M, jnp.int32), conf_slot=-jnp.ones(M, jnp.int32), seen=jnp.zeros(M, bool))


def weights(cfg: EnvCfg, pos):
    """Eq. 35 (Case II) on estimated positions; 1 otherwise."""
    if cfg.weights == "equal":
        return jnp.ones(pos.shape[:-1])
    a = jnp.asarray(cfg.asset_xy)
    d = jnp.minimum(jnp.hypot(pos[..., 0], pos[..., 1]), jnp.hypot(pos[..., 0] - a[0], pos[..., 1] - a[1]))
    return jnp.where(d <= 1.0e4, jnp.minimum(10.0, 4.0 + 1.0e4 / jnp.maximum(d, 1e-9)), 1.0)


def interf_factor(cfg: EnvCfg, pos):
    """Case IV variance factor at a position (A-38); 1 outside the regions or when Case IV is off."""
    f = jnp.ones(pos.shape[:-1])
    if not cfg.interf:
        return f
    for (x0, x1, y0, y1, k) in cfg.interf_regions:
        inside = (pos[..., 0] >= x0) & (pos[..., 0] <= x1) & (pos[..., 1] >= y0) & (pos[..., 1] <= y1)
        f = jnp.where(inside, k, f)
    return f


def lbm_factor_hat(cfg: EnvCfg, xp, Pp):
    """Case III knowledge for OA-informed: expected beam-misalignment loss (eqs. 36-37, i = 2) under the
    filter's own prior bearing uncertainty, dtheta ~ N(0, s2) with s2 = h Pp_pos h^T: E[cos^2 dtheta] =
    (1 + exp(-2 s2)) / 2. Returned as a measurement-variance factor 1 / E[L_bm] (certainty equivalent). Uses the prior
    only (no truth), like the Case IV factor at the estimated position (A-39)."""
    r2 = jnp.maximum(xp[:, 0] ** 2 + xp[:, 1] ** 2, 1e-6)
    h = jnp.stack([-xp[:, 1] / r2, xp[:, 0] / r2], -1)                     # d theta / d (x, y)
    s2 = jnp.einsum("ki,kij,kj->k", h, Pp[:, :2, :2], h)
    if cfg.lbm_i == 2:
        el = 0.5 * (1.0 + jnp.exp(-2.0 * s2))
    else:                                                                    # small-angle Gaussian approximation
        el = (1.0 + cfg.lbm_i * s2) ** -0.5
    return 1.0 / jnp.maximum(el, 1e-6)


def pre(cfg: EnvCfg, scen: Scenario, s: EnvState) -> View:
    pos, present = truth(scen, s.t, cfg)
    xp, Pp = jax.vmap(lambda x, P: ph.predict(x, P, cfg.T0, cfg.sigma_w_sq))(s.x, s.P)
    w = weights(cfg, xp[:, :2]) * s.act
    r_hat = jnp.hypot(xp[:, 0], xp[:, 1])
    fac = interf_factor(cfg, xp[:, :2])
    if cfg.lbm:                                                              # Re-evaluation: OA knows L_bm too
        fac = fac * lbm_factor_hat(cfg, xp, Pp)
    return View(present, pos, s.act, xp, Pp, w, r_hat, fac)


def execute_dwells(cfg: EnvCfg, act, tau_raw):
    """A-12 (empty slots -> 0) and A-13 (sum > T0 -> scaled to T0, scan 0). Returns executed tau [K], tau_s."""
    tau = jnp.where(act, jnp.clip(tau_raw, 0.0, cfg.T0), 0.0)
    tot = tau.sum()
    tau = tau * jnp.where(tot > cfg.T0, cfg.T0 / jnp.maximum(tot, 1e-300), 1.0)
    tau_s = jnp.maximum(cfg.T0 - tau.sum(), 0.0)
    return tau, tau_s


def dual_update(cfg: EnvCfg, lam, g, alpha=None):
    """Eq. 33 with '+' (A-08): lambda_{t+1} = max(0, lambda_t + alpha g_t). alpha defaults to cfg.alpha (Table III);
    a per-lane traced alpha (consts["alpha"]) is used only by the labelled full-run variants (dual50, fixed-lambda)."""
    return jnp.maximum(0.0, lam + (cfg.alpha if alpha is None else alpha) * g)


def post(cfg: EnvCfg, consts: dict, lut, scen: Scenario, s: EnvState, v: View, tau_raw, key):
    """Execute one slot. consts = {"snr0", "s_ref"} (traced). key = fold_in(scenario key, t)."""
    K, M, t = cfg.K, cfg.M, s.t
    tau, tau_s = execute_dwells(cfg, s.act, tau_raw)
    k_trk, k_det, k_sn, k_fa = (jax.random.fold_in(key, i) for i in range(4))
    eps_trk = jax.random.normal(k_trk, (M, 2))
    u_det = jax.random.uniform(k_det, (M,))
    eps_scan = jax.random.normal(k_sn, (M, 2))
    u_fa = jax.random.uniform(k_fa, ())

    # ---------------- tracking dwells (eqs. 5-13); P_d = 1 for track beams (A-14)
    pos_t = v.pos[s.tgt]                                   # [K, 2] truth of the held targets
    r_t = jnp.hypot(pos_t[:, 0], pos_t[:, 1])
    th_t = jnp.arctan2(pos_t[:, 1], pos_t[:, 0])
    snr = ph.track_snr(tau, r_t, consts["snr0"], cfg.r0, cfg.tau0, cfg.r_min_snr)
    if cfg.lbm:                                             # eqs. 36-37 against the PRIOR bearing (A-37)
        dth = jnp.abs(ph.wrap(th_t - jnp.arctan2(v.xp[:, 1], v.xp[:, 0])))
        snr = snr * jnp.where(dth < jnp.pi / 2, jnp.cos(dth) ** cfg.lbm_i, 0.0)
    fac = interf_factor(cfg, pos_t)                         # Case IV, at the TRUE position (A-38)
    do_upd = s.act & (tau > 0) & (snr > 0)
    snr_safe = jnp.where(do_upd, snr, 1.0)
    var_r, var_th = ph.meas_var(snr_safe, cfg.sigma_r0_sq, cfg.sigma_th0_sq)
    var_r, var_th = var_r * fac, var_th * fac
    e = eps_trk[s.tgt]
    z = jnp.stack([r_t + jnp.sqrt(var_r) * e[:, 0], th_t + jnp.sqrt(var_th) * e[:, 1]], -1)
    xu, Pu = jax.vmap(ph.ekf_update)(v.xp, v.Pp, z, var_r, var_th)
    x_new = jnp.where(do_upd[:, None], xu, v.xp)
    P_new = jnp.where(do_upd[:, None, None], Pu, v.Pp)

    # ---------------- scan (eqs. 17-18) of present, untracked targets (A-06)
    tracked = jnp.zeros(M, jnp.int32).at[s.tgt].add(s.act.astype(jnp.int32)) > 0
    untracked = v.present & ~tracked
    r_all = jnp.hypot(v.pos[:, 0], v.pos[:, 1])
    th_all = jnp.arctan2(v.pos[:, 1], v.pos[:, 0])
    snr_s = ph.scan_snr(tau_s, r_all, consts["s_ref"], cfg.r_min_snr)
    if cfg.interf and cfg.interf_scan == "snr":             # A-38b sensitivity: interference also on scan SNR
        snr_s = snr_s / interf_factor(cfg, v.pos)
    pd = ph.pd_lookup(snr_s, lut)
    det = (u_det < pd) & untracked
    sn = jnp.maximum(snr_s, cfg.snr_floor_noise)            # A-52 (noise only)
    if cfg.interf and cfg.interf_scan == "noise":           # A-38b sensitivity: scan measurement variance x factor
        sn = sn / interf_factor(cfg, v.pos)
    zr = r_all + jnp.sqrt(cfg.sigma_r0_sq / sn) * eps_scan[:, 0]
    zt = th_all + jnp.sqrt(cfg.sigma_th0_sq / sn) * eps_scan[:, 1]
    zc = jnp.stack([zr * jnp.cos(zt), zr * jnp.sin(zt)], -1)
    fa = u_fa < cfg.pfa                                     # A-49: counted, never initiates (prob ~1e-12)

    # ---------------- 3-of-4 initiation per target, gate T_d on the buffer's last measurement (A-44, A-49)
    st = jnp.where(untracked, s.buf, 0)
    gate_fail = det & (st != 0) & (jnp.hypot(*(zc - s.b_last).T) > cfg.gate_Td)
    st = jnp.where(gate_fail, 0, st)
    nxt = jnp.where(det, ph.NEXT_DET[st], ph.NEXT_MISS[st])
    nxt = jnp.where(untracked, nxt, 0)
    b_prev = jnp.where(det[:, None], s.b_last, s.b_prev)
    k_prev = jnp.where(det, s.k_last, s.k_prev)
    b_last = jnp.where(det[:, None], zc, s.b_last)
    k_last = jnp.where(det, t, s.k_last)
    conf = nxt == 6
    # FCFS slot assignment in target-index order (A-46, A-47)
    free = ~s.act
    n_free = free.sum()
    free_order = jnp.argsort(~free, stable=True)            # free slot indices first, ascending
    rank = jnp.cumsum(conf) - 1
    ok = conf & (rank < n_free)
    slot_of = free_order[jnp.clip(rank, 0, K - 1)]
    A = ok[:, None] & (slot_of[:, None] == jnp.arange(K)[None, :])    # [M, K]
    newk = A.any(0)
    j_new = jnp.argmax(A, axis=0)
    vel = (b_last - b_prev) / (jnp.maximum(k_last - k_prev, 1)[:, None] * cfg.T0)    # A-45
    x_init = jnp.concatenate([b_last, vel], -1)[j_new]
    P0 = jnp.diag(jnp.asarray(cfg.P0_diag))
    x_new = jnp.where(newk[:, None], x_init, x_new)
    P_new = jnp.where(newk[:, None, None], P0, P_new)
    act_new = s.act | newk
    tgt_new = jnp.where(newk, j_new, s.tgt)
    buf = jnp.where(conf, 0, nxt)                           # confirmed (or refused for lack of a slot) -> E
    conf_slot = jnp.where(ok & (s.conf_slot < 0), t, s.conf_slot)

    # ---------------- score (eq. 19), reward (eq. 30), dual (eq. 33)
    c = jnp.where(act_new, ph.pos_trace(P_new), 0.0)
    w = jnp.where(newk, weights(cfg, x_init[:, :2]), v.w) * act_new
    n_trk = act_new.sum()
    n_pres = v.present.sum()
    nmiss = jnp.maximum(n_pres - n_trk, 0)                  # A-11
    U = -(w * c).sum() - cfg.beta * nmiss
    g_exec = tau.sum() / cfg.T0 - cfg.theta_max             # executed dwells (A-13)
    g_req = jnp.where(s.act, jnp.clip(tau_raw, 0.0, cfg.T0), 0.0).sum() / cfg.T0 - cfg.theta_max
    g = g_exec if cfg.g_on == "executed" else g_req          # A-13a: penalty on the agent's requested dwells
    if "g_req" in consts:                                    # per-lane (the re-evaluation): 1 = eq. 30 literal
        g = g_exec + consts["g_req"] * (g_req - g_exec)
    r = U - s.lam * g
    lam_next = dual_update(cfg, s.lam, g, consts.get("alpha"))

    # ---------------- exits for slot t+1 (A-15): drop tracks whose target is gone, zero their observation
    _, pres_next = truth(scen, t + 1, cfg)
    keep = act_new & pres_next[tgt_new]
    tau_obs = tau
    if "obs_lit" in consts:                                  # eq. 34 literal: the CHOSEN (clipped) dwell
        tau_req = jnp.where(s.act, jnp.clip(tau_raw, 0.0, cfg.T0), 0.0)
        tau_obs = jnp.where(consts["obs_lit"] > 0.5, tau_req, tau)
    lam_prev = None if s.lam_prev is None else s.lam
    s2 = EnvState(t=t + 1, lam_prev=lam_prev, act=keep, tgt=tgt_new.astype(jnp.int32), x=x_new, P=P_new,
                  prev_wc=jnp.where(keep, w * c, 0.0), prev_tau=jnp.where(keep, tau_obs, 0.0), lam=lam_next,
                  buf=jnp.where(pres_next, buf, 0).astype(jnp.int32), b_last=b_last, b_prev=b_prev, k_last=k_last.astype(jnp.int32), k_prev=k_prev.astype(jnp.int32),
                  conf_slot=conf_slot.astype(jnp.int32), seen=s.seen | v.present)
    out = dict(U=U, r=r, g=g, g_exec=g_exec, g_req=g_req, tau=tau, tau_s=tau_s, c_sum=c.sum(), wc_sum=(w * c).sum(), nmiss=nmiss,
               n_trk=n_trk, n_pres=n_pres, lam=s.lam, n_conf=ok.sum(), n_refused=(conf & ~ok).sum(), fa=fa,
               n_det=det.sum(), n_trk_pre=s.act.sum(),
               # oracle-replay outputs (reference-code check: filterpy re-filters these measurements; jit drops them
               # when unused)
               dbg_z=z, dbg_var_r=var_r, dbg_var_th=var_th, dbg_upd=do_upd, dbg_newk=newk, dbg_x_init=x_init,
               dbg_act_pre=s.act, dbg_act_new=act_new, dbg_c=c, dbg_w=w)
    return s2, out


def tinit_stats(scen: Scenario, s: EnvState):
    """T_init per target = first confirmation slot - spawn slot (A-35, 'gaps' convention: VVV from the birth slot
    gives 2), over targets seen in the episode. Returns (sum over confirmed, n confirmed, n seen but never confirmed).
    The 'scans' convention adds 1 per confirmed target; tinit_cens_sum (censored at episode end) is separate."""
    seen = s.seen & scen.valid
    confd = seen & (s.conf_slot >= 0)
    ti = jnp.where(confd, s.conf_slot - scen.born, 0)
    return ti.sum(), confd.sum(), (seen & ~confd).sum()


def tinit_cens_sum(scen: Scenario, s: EnvState):
    """Sum over seen-but-never-confirmed targets of (episode end - spawn slot), i.e. their T_init
    censored at the episode end (gaps convention), with the end = min(final slot, removal)."""
    seen = s.seen & scen.valid
    never = seen & (s.conf_slot < 0)
    end = jnp.minimum(s.t, scen.death)
    return jnp.where(never, end - scen.born, 0).sum()


# ----------------------------------------------------------------------------- observations (eqs. 28 and 34)


def _enc(x, scale, transform):
    """Fixed (data-independent) encoding of a nonnegative cost / multiplier: 'affine' x*scale (A-23) or
    'log' log1p(x*scale) (A-23a amendment: costs span 1e2..1e12 m^2 once a track is starved)."""
    return x * scale if transform == "affine" else jnp.log1p(jnp.maximum(x, 0.0) * scale)


def obs_ddpg(s: EnvState, cfg: EnvCfg, cost_scale: float, lam_scale: float, transform: str = "affine", lit=None):
    """Eq. 34: [{w c_{t-1}}, {tau_{t-1}}, lambda] (dim 2K+1), fixed encoding (A-23/A-23a), float32.
    lambda position carries lambda_t (A-51); with lit (traced 0/1, 'lit' arm) > 0.5 it carries lambda_{t-1} (eq. 34
    literal; needs reset(..., lit=True))."""
    lam = s.lam if (lit is None or s.lam_prev is None) else jnp.where(lit > 0.5, s.lam_prev, s.lam)
    return jnp.concatenate([_enc(s.prev_wc, cost_scale, transform), s.prev_tau / cfg.T0,
                            _enc(lam, lam_scale, transform)[None]]).astype(jnp.float32)


def obs_dql(s: EnvState, cfg: EnvCfg, cost_scale: float, lam_scale: float, transform: str = "affine", lit=None):
    """Eq. 28 per agent n: [w^n c^n, {w c}, {tau}, lambda] (dim 2K+2) -> [K, 2K+2], float32."""
    base = obs_ddpg(s, cfg, cost_scale, lam_scale, transform, lit)
    own = _enc(s.prev_wc, cost_scale, transform).astype(jnp.float32)
    return jnp.concatenate([own[:, None], jnp.tile(base[None], (cfg.K, 1))], -1)
