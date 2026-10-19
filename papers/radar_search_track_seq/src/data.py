"""Environment: AESA beam pointing over a fixed field of regard, CV targets, a Stone-Soup-equivalent multi-target
tracker (KF predict, UKF update, Mahalanobis distance hypothesiser + gater, greedy NearestNeighbour or GNN associator,
SimpleMeasurementInitiator, trace CovarianceBasedDeleter (or the paper-text Frobenius deleter, del_stat)), the scan-history value (paper Eq. 1), the reward (paper §2.5)
and the per-step metrics (SumofCovarianceNorms, GOSPA).

There is no dataset in this paper: "data" is this simulator. Everything is a pure function of one environment; vmap
outside. float64 throughout (see common.py dtype policy); observations are emitted in float32.

Step order (one step = dt):
  1. truth moves (CV + white-acceleration noise, keyed by the scenario key and t -> common random numbers across arms)
  2. the beam at the chosen cell illuminates a 4.5 deg cone; each target inside it is detected with prob. pd and measured
     in (el, az, range) with Gaussian noise
  3. tracker: predict all tracks -> measurement prediction (UKF) -> Mahalanobis distances -> gate (< gate) -> associate
     -> UKF update associated tracks -> delete (trace P > del_thr) -> initiate one track per unassociated detection
  4. search reward r_SV = -SSV(look) evaluated BEFORE the look is pushed onto the scan FIFO; then push
  5. track reward r_TL (flipped sign = covariance reduction rewarded; the literal sign is its negation)
  6. metrics on the post-step track set; observation for the next decision.
"""
from __future__ import annotations

import itertools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from src.configs.config import EnvConfig

F64 = jnp.float64
D2R = math.pi / 180.0
LIFE_BINS = 64


# ----------------------------------------------------------------------------- state containers
class Tracks(NamedTuple):
    x: jax.Array         # [K, 6]  state [x, vx, y, vy, z, vz]
    P: jax.Array         # [K, 6, 6]
    valid: jax.Array     # [K] bool
    tid: jax.Array       # [K] int32 creation id
    last_upd: jax.Array  # [K] int32 step of the last update (initiation counts)
    n_upd: jax.Array     # [K] int32 number of updates after initiation (= revisits that hit)
    det_prev: jax.Array  # [K] bool  associated with a detection at the previous step (incl. its initiating detection)
    pn_prev: jax.Array   # [K] Frobenius norm of P at the end of the previous step


class EnvState(NamedTuple):
    t: jax.Array          # int32 steps taken
    scen: jax.Array       # scenario PRNG key
    truth: jax.Array      # [n_tgt, 6]
    tr: Tracks
    next_tid: jax.Array   # int32
    sv_az: jax.Array      # [N_SV] look azimuth (deg), index 0 = most recent
    sv_el: jax.Array      # [N_SV] look elevation (deg)
    sv_t: jax.Array       # [N_SV] int32 step of the look
    sv_valid: jax.Array   # [N_SV] bool
    last_look: jax.Array  # [n_tgt] int32 last step each target was inside the beam (ceiling tie-break)
    n_init: jax.Array     # int32 counters
    n_del: jax.Array
    overflow: jax.Array
    life_hist: jax.Array  # [LIFE_BINS] int32 histogram of n_upd of deleted tracks


class Obs(NamedTuple):
    tl: jax.Array         # [K, 7] float32 scaled track rows, valid rows first in creation order
    tl_mask: jax.Array    # [K] bool
    sv_az: jax.Array      # [N_SV] float32 deg
    sv_el: jax.Array      # [N_SV] float32 deg
    sv_age: jax.Array     # [N_SV] float32 age at the NEXT look (>= 1)
    sv_valid: jax.Array   # [N_SV] bool


# ----------------------------------------------------------------------------- constants
def action_centres_deg(cfg: EnvConfig) -> jax.Array:
    """companion eq. (2): linspace(-60, 60, 19) deg."""
    return jnp.linspace(-cfg.for_half_deg, cfg.for_half_deg, cfg.n_a, dtype=F64)


def cv_FQ(dt: float, q: float):
    """Per-axis CV model (README), block-diagonal over (x, y, z) with state order [x, vx, y, vy, z, vz]."""
    F1 = np.array([[1.0, dt], [0.0, 1.0]])
    Q1 = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    return jnp.asarray(np.kron(np.eye(3), F1)), jnp.asarray(np.kron(np.eye(3), Q1))


def meas_R(cfg: EnvConfig) -> jax.Array:
    return jnp.diag(jnp.array([(cfg.sig_el_deg * D2R) ** 2, (cfg.sig_az_deg * D2R) ** 2, cfg.sig_rng_m ** 2], F64))


def ssv_pmax(cfg: EnvConfig) -> float:
    """p_max,T = sum_{k=0}^{T} phi_k(0) (README). T is undefined in the paper: ssv_norm="fifo" (primary) uses
    T = N_SV-1; "single" uses T = 0, i.e. phi_0(0) = 1/(2 pi sigma0^2) (recheck)."""
    s0 = cfg.sigma0_deg
    if cfg.ssv_norm == "single":
        return float(1.0 / (2 * math.pi * s0 ** 2))
    if cfg.ssv_norm != "fifo":
        raise ValueError(cfg.ssv_norm)
    return float(sum(1.0 / (2 * math.pi * (s0 * (1 + cfg.gamma_sv) ** k) ** 2) for k in range(cfg.n_sv)))


def wrap_pi(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


# ----------------------------------------------------------------------------- measurement model
def h_meas(x):
    """(el, az, range) of state x [..., 6] (Stone Soup CartesianToElevationBearingRange: el = asin(z/rho))."""
    px, py, pz = x[..., 0], x[..., 2], x[..., 4]
    rho = jnp.sqrt(px ** 2 + py ** 2 + pz ** 2)
    el = jnp.arcsin(pz / rho)
    az = jnp.arctan2(py, px)
    return jnp.stack([el, az, rho], -1)


def h_inverse(z):
    """sphere2cart: state with position from (el, az, range), velocity 0."""
    el, az, rho = z[0], z[1], z[2]
    px = rho * jnp.cos(el) * jnp.cos(az)
    py = rho * jnp.cos(el) * jnp.sin(az)
    pz = rho * jnp.sin(el)
    return jnp.array([px, 0.0, py, 0.0, pz, 0.0], F64)


def fd_jacobian(x):
    """Forward-difference Jacobian exactly as stonesoup.functions.jacobian: delta = 1e8 * np.spacing(x), then
    delta[delta < 1e-8] = 1e-8. np.spacing is NEGATIVE for negative x, so every negative coordinate gets the 1e-8
    floor (a library quirk reproduced on purpose: it perturbs C0 at the 1e-4 relative level; analytic-Jacobian
    agreement is checked separately in tests)."""
    ax = jnp.abs(x)
    sp = jnp.nextafter(ax, jnp.inf) - ax
    delta = jnp.where(x > 0, jnp.maximum(1e8 * sp, 1e-8), 1e-8)
    X = x[None, :] + jnp.eye(6, dtype=F64) * delta[:, None]
    Fx = h_meas(X)                       # [6, 3]
    f0 = h_meas(x)
    d = Fx - f0[None, :]
    d = d.at[:, 0].set(wrap_pi(d[:, 0])).at[:, 1].set(wrap_pi(d[:, 1]))
    return (d / delta[:, None]).T        # [3, 6]


def ukf_weights(cfg: EnvConfig, n: int = 6):
    """Scaled-UT weights (Stone Soup gauss2sigma): lambda = alpha^2 (n + kappa) - n, c = n + lambda."""
    a2 = cfg.ukf_alpha ** 2
    lam = a2 * (n + cfg.ukf_kappa) - n
    c = n + lam
    wm = np.full(2 * n + 1, 0.5 / c)
    wm[0] = lam / c
    wc = wm.copy()
    wc[0] = lam / c + (1 - a2 + cfg.ukf_beta)
    return c, jnp.asarray(wm), jnp.asarray(wc)


def ukf_meas_pred(x, P, R, cfg: EnvConfig, h=None, angle_idx=(0, 1)):
    """Stone Soup UnscentedKalmanUpdater.predict_measurement: sigma points (Cholesky of P), angle components averaged
    with the weighted circular mean (Angle.average) and their differences wrapped, other components averaged linearly;
    S = sum Wc dz dz^T + R, C = sum Wc (chi - chi_0) dz^T. h defaults to the (el, az, range) radar model."""
    h = h_meas if h is None else h
    c, wm, wc = ukf_weights(cfg, x.shape[0])
    L = jnp.linalg.cholesky(P)
    sq = jnp.sqrt(c) * L
    chi = jnp.concatenate([x[None], x[None] + sq.T, x[None] - sq.T], 0)   # [2n+1, n]
    Z = h(chi)                                                           # [2n+1, m]
    wsum = jnp.sum(wm)
    lin = Z.T @ wm / wsum
    is_ang = np.zeros(Z.shape[1], bool)
    is_ang[list(angle_idx)] = True
    circ = jnp.arctan2(jnp.sin(Z).T @ wm / wsum, jnp.cos(Z).T @ wm / wsum)
    zhat = jnp.where(jnp.asarray(is_ang), circ, lin)
    dz = Z - zhat[None]
    dz = jnp.where(jnp.asarray(is_ang)[None], wrap_pi(dz), dz)
    S = (dz * wc[:, None]).T @ dz + R
    dx = chi - chi[0:1]
    C = (dx * wc[:, None]).T @ dz
    return zhat, S, C


def innovation(z, zhat):
    nu = z - zhat
    return nu.at[0].set(wrap_pi(nu[0])).at[1].set(wrap_pi(nu[1]))


# ----------------------------------------------------------------------------- association
def _combo_table(n_rows: int, n_cols: int):
    """All maps rows -> {0..n_cols-1, n_cols = none} that are injective on the assigned part. [C, n_rows] int32."""
    combos = [c for c in itertools.product(range(n_cols + 1), repeat=n_rows)
              if len([v for v in c if v < n_cols]) == len({v for v in c if v < n_cols})]
    return np.asarray(combos, np.int32)


def associate_nn(D, eligible):
    """Greedy NearestNeighbour (Stone Soup): repeatedly take the globally smallest-distance (track, detection) pair among
    unassigned tracks / unused detections. Gated pairs (eligible) always beat the missed hypothesis because the gate
    threshold equals missed_distance (both 3). Returns assign [K] int32 (detection index or -1)."""
    K, M = D.shape

    def body(i, carry):
        assign, tfree, dfree = carry
        Dm = jnp.where(eligible & tfree[:, None] & dfree[None, :], D, jnp.inf)
        flat = jnp.argmin(Dm)
        k, m = (flat // M).astype(jnp.int32), (flat % M).astype(jnp.int32)
        ok = jnp.isfinite(Dm.reshape(-1)[flat])
        assign = jnp.where(ok, assign.at[k].set(m), assign)
        tfree = jnp.where(ok, tfree.at[k].set(False), tfree)
        dfree = jnp.where(ok, dfree.at[m].set(False), dfree)
        return assign, tfree, dfree

    init = (jnp.full((K,), -1, jnp.int32), jnp.ones((K,), bool), jnp.ones((M,), bool))
    assign, _, _ = jax.lax.fori_loop(0, M, body, init)
    return assign


def associate_gnn(D, eligible, missed: float):
    """Global NN (paper §2.1 '2D Munkres'): exact minimum of sum_{assigned} d + missed * #(unassigned valid tracks),
    by enumerating detection -> track maps (M <= 3 detections, so the table is small)."""
    K, M = D.shape
    combos = jnp.asarray(_combo_table(M, K))            # [C, M] detection m -> track (K = none)
    Dp = jnp.concatenate([D, jnp.zeros((1, M))], 0)      # row K = "not assigned"
    elp = jnp.concatenate([eligible, jnp.ones((1, M), bool)], 0)
    ms = jnp.arange(M)
    dsel = Dp[combos, ms[None, :]]                       # [C, M]
    ok = jnp.all(elp[combos, ms[None, :]], axis=1)
    # assigning a detection replaces that track's missed cost by d: cost = sum(d - missed) over assigned
    cost = jnp.sum(jnp.where(combos < K, dsel - missed, 0.0), axis=1)
    cost = jnp.where(ok, cost, jnp.inf)
    best = combos[jnp.argmin(cost)]
    assign = jnp.full((K + 1,), -1, jnp.int32).at[best].set(ms.astype(jnp.int32))
    return assign[:K]


# ----------------------------------------------------------------------------- tracker step (detections given)
def empty_tracks(cfg: EnvConfig) -> Tracks:
    K = cfg.k_max
    x0 = jnp.zeros((K, 6), F64).at[:, 0].set(1e4)
    P0 = jnp.tile(jnp.eye(6, dtype=F64) * 100.0, (K, 1, 1))
    z = jnp.zeros((K,), jnp.int32)
    return Tracks(x0, P0, jnp.zeros((K,), bool), z - 1, z, z, jnp.zeros((K,), bool), jnp.zeros((K,), F64))


def initiate_state(z, cfg: EnvConfig):
    """SimpleMeasurementInitiator: x = inverse_function(z) (+ zero prior mean); C0 = J+ R J+^T + prior covar with the
    mapped (position) rows/cols zeroed (velocity variance init_vel_sigma^2 remains)."""
    x = h_inverse(z)
    J = jax.jacfwd(h_meas)(x) if cfg.init_jacobian == "analytic" else fd_jacobian(x)
    Jp = jnp.linalg.pinv(J)
    prior = jnp.diag(jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], F64)) * cfg.init_vel_sigma ** 2
    C0 = Jp @ meas_R(cfg) @ Jp.T + prior
    return x, C0


def rtl_flip_terms(valid, hit, det_prev, pn_prev, pn_now):
    """Per-track flipped-sign r_TL term: ||P_{t-1}||_F - ||P_t||_F for tracks associated with a detection at t
    or at t-1 (D_t u D_{t-1}); 0 otherwise. The literal printed sign is the negation."""
    return jnp.where(valid & (hit | det_prev), pn_prev - pn_now, 0.0)


def sylvester_min(S):
    """Smallest leading principal minor of each 3x3 S (all > 0 <=> S positive definite): the cheap S-PSD check."""
    m1 = S[..., 0, 0]
    m2 = S[..., 0, 0] * S[..., 1, 1] - S[..., 0, 1] * S[..., 1, 0]
    m3 = jnp.linalg.det(S)
    return jnp.minimum(jnp.minimum(m1, m2), m3)


def tracker_step(tr: Tracks, next_tid, z, zvalid, t, cfg: EnvConfig):
    """One Stone Soup MultiTargetTracker step at time index t given detections z [M, 3] (el, az, range) with validity.

    Returns (tracks', next_tid', aux). aux carries what the reward / metrics / diagnostics need."""
    F, Q = cv_FQ(cfg.dt, cfg.q_filt)
    R = meas_R(cfg)
    K = cfg.k_max
    M = z.shape[0]
    # predict (KalmanPredictor)
    xp = tr.x @ F.T
    Pp = F[None] @ tr.P @ F.T[None] + Q[None]
    # measurement prediction for every slot (invalid slots hold a harmless dummy state)
    zhat, S, C = jax.vmap(lambda x, P: ukf_meas_pred(x, P, R, cfg))(xp, Pp)
    Sinv = jnp.linalg.inv(S)
    nu = jax.vmap(lambda zh: jax.vmap(lambda zm: innovation(zm, zh))(z))(zhat)          # [K, M, 3]
    D = jnp.sqrt(jnp.maximum(jnp.einsum("kmi,kij,kmj->km", nu, Sinv, nu), 0.0))          # Mahalanobis distance
    eligible = tr.valid[:, None] & zvalid[None, :] & (D < cfg.gate)
    if cfg.associator == "gnn":
        assign = associate_gnn(D, eligible, cfg.missed)
    else:
        assign = associate_nn(D, eligible)
    hit = assign >= 0
    # UKF update (Stone Soup KalmanUpdater: K = C S^-1, P+ = P - K S K^T)
    Kg = C @ Sinv
    nu_a = jnp.take_along_axis(nu, jnp.maximum(assign, 0)[:, None, None], axis=1)[:, 0]
    xu = xp + jnp.einsum("kij,kj->ki", Kg, nu_a)
    Pu = Pp - Kg @ S @ jnp.swapaxes(Kg, 1, 2)
    x1 = jnp.where(hit[:, None], xu, xp)
    P1 = jnp.where(hit[:, None, None], Pu, Pp)
    pn_now = jnp.sqrt(jnp.sum(P1 ** 2, axis=(1, 2)))                                     # Frobenius (pre-deletion)
    # track reward bookkeeping (pre-deletion set, README; new tracks contribute 0)
    rtl_flip = jnp.sum(rtl_flip_terms(tr.valid, hit, tr.det_prev, tr.pn_prev, pn_now))
    # delete (CovarianceBasedDeleter: trace > threshold; del_stat="frob" = this paper's text, Frobenius norm)
    if cfg.del_stat == "frob":
        del_stat = pn_now
    elif cfg.del_stat == "trace":
        del_stat = jnp.trace(P1, axis1=1, axis2=2)
    else:
        raise ValueError(f"unknown del_stat {cfg.del_stat!r}")
    kill = tr.valid & (del_stat > cfg.del_thr)
    n_upd1 = tr.n_upd + hit.astype(jnp.int32)
    life_add = jnp.zeros((LIFE_BINS,), jnp.int32).at[jnp.minimum(n_upd1, LIFE_BINS - 1)].add(kill.astype(jnp.int32))
    valid1 = tr.valid & ~kill
    tr1 = Tracks(x=jnp.where(valid1[:, None], x1, empty_tracks(cfg).x),
                 P=jnp.where(valid1[:, None, None], P1, empty_tracks(cfg).P),
                 valid=valid1,
                 tid=jnp.where(valid1, tr.tid, -1),
                 last_upd=jnp.where(hit, t, tr.last_upd),
                 n_upd=jnp.where(valid1, n_upd1, 0),
                 det_prev=valid1 & hit,
                 pn_prev=jnp.where(valid1, pn_now, 0.0))
    # initiate one track per unassociated detection (SimpleMeasurementInitiator), into the first free slot
    used = jnp.zeros((M,), bool).at[jnp.where(hit, assign, M)].set(True, mode="drop")
    unassoc = zvalid & ~used
    x0s, C0s = jax.vmap(lambda zz: initiate_state(zz, cfg))(z)

    def init_one(m, carry):
        trc, ntid, n_new, ovf = carry
        free = ~trc.valid
        slot = jnp.argmax(free)
        can = unassoc[m] & jnp.any(free)
        ovf = ovf + (unassoc[m] & ~jnp.any(free)).astype(jnp.int32)
        pn0 = jnp.sqrt(jnp.sum(C0s[m] ** 2))

        def put(arr, val):
            return jnp.where(can, arr.at[slot].set(val), arr)

        trc = Tracks(put(trc.x, x0s[m]), put(trc.P, C0s[m]), put(trc.valid, True), put(trc.tid, ntid),
                     put(trc.last_upd, t), put(trc.n_upd, 0), put(trc.det_prev, True), put(trc.pn_prev, pn0))
        return trc, ntid + can.astype(jnp.int32), n_new + can.astype(jnp.int32), ovf

    tr2, next_tid2, n_new, ovf = jax.lax.fori_loop(0, M, init_one, (tr1, next_tid, jnp.int32(0), jnp.int32(0)))
    aux = dict(rtl_flip=rtl_flip, n_hit=jnp.sum(hit).astype(jnp.int32), n_new=n_new, n_kill=jnp.sum(kill).astype(jnp.int32), overflow=ovf,
               cov_killed=jnp.sum(jnp.where(kill, pn_now, 0.0)),   # last-state ||P||_F of tracks deleted this step
               kill=kill, x_last=x1,                               # deleted tracks' last state (library GOSPA)
               life_add=life_add, assign=assign, min_S_minor=jnp.min(jnp.where(tr.valid, sylvester_min(S), jnp.inf)))
    return tr2, next_tid2, aux


# ----------------------------------------------------------------------------- scan value (paper Eq. 1)
def ssv_at(look_az, look_el, sv_az, sv_el, ages, valid, cfg: EnvConfig):
    """SSV = sum_i phi_{age_i}(look - mu_i) / p_max, phi_k isotropic 2-D Gaussian with sigma_k = sigma0 (1+gamma)^k."""
    sig = cfg.sigma0_deg * (1.0 + cfg.gamma_sv) ** ages
    d2 = (look_az - sv_az) ** 2 + (look_el - sv_el) ** 2
    phi = jnp.exp(-d2 / (2 * sig ** 2)) / (2 * jnp.pi * sig ** 2)
    return jnp.sum(jnp.where(valid, phi, 0.0)) / ssv_pmax(cfg)


def render_raster(obs_sv_az, obs_sv_el, obs_age, obs_valid, cfg: EnvConfig):
    """48 x 48 SSV image over linspace(-60, 60)^2 (rows = elevation, cols = azimuth), float32. Separable Gaussian:
    image = Ey^T diag(w) Ex. Works on a single observation; vmap for batches."""
    g = jnp.linspace(-cfg.for_half_deg, cfg.for_half_deg, cfg.raster, dtype=jnp.float32)
    sig = jnp.float32(cfg.sigma0_deg) * jnp.float32(1.0 + cfg.gamma_sv) ** obs_age
    w = jnp.where(obs_valid, 1.0 / (2 * jnp.pi * sig ** 2), 0.0) / jnp.float32(ssv_pmax(cfg))
    ex = jnp.exp(-((g[None, :] - obs_sv_az[:, None]) ** 2) / (2 * sig[:, None] ** 2))   # [N, 48]
    ey = jnp.exp(-((g[None, :] - obs_sv_el[:, None]) ** 2) / (2 * sig[:, None] ** 2))
    return (ey * w[:, None]).T @ ex                                                     # [48(el), 48(az)]


# ----------------------------------------------------------------------------- GOSPA (Rahmathullah et al. 2017)
_GOSPA_CACHE = {}


def gospa(truth, tx, tvalid, c: float, p: int):
    """GOSPA with alpha = 2 over the full 6-D state (Stone Soup GOSPAMetric default measure). Exact minimum by
    enumeration of truth -> track maps. Returns (distance, localisation, missed, false) for one time step."""
    n, K = truth.shape[0], tx.shape[0]
    key = (n, K)
    if key not in _GOSPA_CACHE:
        _GOSPA_CACHE[key] = _combo_table(n, K)
    combos = jnp.asarray(_GOSPA_CACHE[key])                                      # [C, n]
    d = jnp.sqrt(jnp.sum((truth[:, None, :] - tx[None, :, :]) ** 2, -1))        # [n, K]
    dp = jnp.concatenate([d, jnp.full((n, 1), jnp.inf)], 1)
    vp = jnp.concatenate([tvalid, jnp.array([True])])
    ii = jnp.arange(n)[None, :]
    dsel = dp[ii, combos]                                                        # [C, n]
    asg = combos < K
    ok = jnp.all(~asg | (vp[combos] & (dsel < c)), axis=1)
    n_asg = jnp.sum(asg, 1)
    ny = jnp.sum(tvalid)
    loc = jnp.sum(jnp.where(asg, dsel, 0.0) ** p, 1)
    cost = loc + (c ** p / 2.0) * (n + ny - 2 * n_asg)
    cost = jnp.where(ok, cost, jnp.inf)
    b = jnp.argmin(cost)
    na = n_asg[b]
    return (cost[b] ** (1.0 / p), loc[b], (c ** p / 2.0) * (n - na), (c ** p / 2.0) * (ny - na))


# ----------------------------------------------------------------------------- scenario
def sample_truth(scen_key, cfg: EnvConfig):
    k = jax.random.split(jax.random.fold_in(scen_key, 0x5CE), 5)
    n = cfg.n_tgt
    rng = jax.random.uniform(k[0], (n,), F64, cfg.rng_lo_m, cfg.rng_hi_m)
    az = jax.random.uniform(k[1], (n,), F64, -cfg.az_lim_deg, cfg.az_lim_deg) * D2R
    el = jax.random.uniform(k[2], (n,), F64, -cfg.el_lim_deg, cfg.el_lim_deg) * D2R
    spd = jax.random.uniform(k[3], (n,), F64, cfg.spd_lo, cfg.spd_hi)
    hdg = jax.random.uniform(k[4], (n,), F64, 0.0, 2 * jnp.pi)
    px, py, pz = rng * jnp.cos(el) * jnp.cos(az), rng * jnp.cos(el) * jnp.sin(az), rng * jnp.sin(el)
    if cfg.truth_vel == "uniform_speed":                    # original reading (kept for the scene audit)
        vx, vy = spd * jnp.cos(hdg), spd * jnp.sin(hdg)
    else:                                                   #: velocity drawn from the initiator's prior
        sd = cfg.init_vel_sigma if cfg.truth_vel_sigma is None else cfg.truth_vel_sigma   #: None = tied
        v = jax.random.normal(jax.random.fold_in(scen_key, 0x7E1), (n, 2), F64) * sd
        vx, vy = v[:, 0], v[:, 1]
    return jnp.stack([px, vx, py, vy, pz, jnp.zeros_like(px)], -1)


def reset(scen_key, cfg: EnvConfig) -> EnvState:
    N = cfg.n_sv
    zi = jnp.int32(0)
    return EnvState(t=zi, scen=scen_key, truth=sample_truth(scen_key, cfg), tr=empty_tracks(cfg), next_tid=zi,
                    sv_az=jnp.zeros((N,), F64), sv_el=jnp.zeros((N,), F64), sv_t=jnp.zeros((N,), jnp.int32),
                    sv_valid=jnp.zeros((N,), bool), last_look=jnp.full((cfg.n_tgt,), -1, jnp.int32),
                    n_init=zi, n_del=zi, overflow=zi, life_hist=jnp.zeros((LIFE_BINS,), jnp.int32))


def track_rows(tr: Tracks, cfg: EnvConfig):
    """Track list rows (companion eq. 3): (|p|, az, el, vx, vy, vz, ||P||_F), scaled, valid rows first in
    creation order. Returns (rows [K, 7] float32, mask [K] bool, order [K])."""
    order = jnp.argsort(jnp.where(tr.valid, tr.tid, jnp.iinfo(jnp.int32).max))
    x, P, v = tr.x[order], tr.P[order], tr.valid[order]
    px, py, pz = x[:, 0], x[:, 2], x[:, 4]
    rng = jnp.sqrt(px ** 2 + py ** 2 + pz ** 2)
    az = jnp.arctan2(py, px) / D2R
    el = jnp.arctan2(pz, jnp.sqrt(px ** 2 + py ** 2)) / D2R
    pn = jnp.sqrt(jnp.sum(P ** 2, axis=(1, 2)))
    rows = jnp.stack([rng / cfg.scale_range, az / cfg.scale_angle_deg, el / cfg.scale_angle_deg,
                      x[:, 1] / cfg.scale_vel, x[:, 3] / cfg.scale_vel, x[:, 5] / cfg.scale_vel,
                      pn / cfg.scale_cov], -1)
    rows = jnp.where(v[:, None], rows, 0.0)
    return rows.astype(jnp.float32), v, order


def observe(s: EnvState, cfg: EnvConfig) -> Obs:
    rows, mask, _ = track_rows(s.tr, cfg)
    age = (s.t + 1 - s.sv_t).astype(jnp.float32)
    return Obs(rows, mask, s.sv_az.astype(jnp.float32), s.sv_el.astype(jnp.float32),
               jnp.where(s.sv_valid, age, 1.0), s.sv_valid)


def detect(truth, look_az_deg, look_el_deg, key, cfg: EnvConfig):
    """Beam detections: targets within beam_r (great-circle) of the beam centre, detected w.p. pd; (el, az, range)
    measurements with Gaussian noise. Returns (z [n, 3], valid [n], in_beam [n])."""
    taz, tel = look_az_deg * D2R, look_el_deg * D2R
    u = jnp.array([jnp.cos(tel) * jnp.cos(taz), jnp.cos(tel) * jnp.sin(taz), jnp.sin(tel)])
    p = truth[:, [0, 2, 4]]
    rho = jnp.linalg.norm(p, axis=-1)
    cosang = (p @ u) / rho
    in_beam = cosang >= jnp.cos(cfg.beam_r_deg * D2R)
    k1, k2 = jax.random.split(key)
    hit = in_beam & (jax.random.uniform(k2, (truth.shape[0],), F64) < cfg.pd)
    noise = jax.random.normal(k1, (truth.shape[0], 3), F64) * jnp.sqrt(jnp.diag(meas_R(cfg)))[None]
    z = h_meas(truth) + noise
    return z, hit, in_beam


def clutter(look_az_deg, look_el_deg, key, cfg: EnvConfig):
    """False alarms for one look ( sensitivity; off in the primary scene): n_fa_slots Bernoulli slots with
    p = clutter_rate / n_fa_slots (Binomial ~ Poisson(clutter_rate) for many slots), each uniform in the beam disc
    (flat (az, el) offsets) and uniform in [rng_lo, rng_hi]. Returns (z [n_fa, 3] (el, az, range), valid [n_fa])."""
    n = cfg.n_fa_slots
    k = jax.random.split(key, 4)
    on = jax.random.uniform(k[0], (n,), F64) < (cfg.clutter_rate / n)
    r = cfg.beam_r_deg * jnp.sqrt(jax.random.uniform(k[1], (n,), F64))
    th = jax.random.uniform(k[2], (n,), F64, 0.0, 2 * jnp.pi)
    el = jnp.clip(look_el_deg + r * jnp.sin(th), -89.0, 89.0) * D2R
    az = (look_az_deg + r * jnp.cos(th)) * D2R
    rng = jax.random.uniform(k[3], (n,), F64, cfg.rng_lo_m, cfg.rng_hi_m)
    return jnp.stack([el, wrap_pi(az), rng], -1), on


LIB_KILL_SLOTS = 3   # deleted tracks per step carried into the library GOSPA (overflow counted in info["lib_kill_ovf"])


def step(s: EnvState, action, cfg: EnvConfig):
    """action = (i_az, i_el) int. Returns (state', obs', info). info['r_sv'], info['rtl_flip'] are the two reward terms
    (the literal-sign track term is -rtl_flip, README)."""
    t1 = s.t + 1
    kt = jax.random.fold_in(s.scen, t1)
    cen = action_centres_deg(cfg)
    look_az, look_el = cen[action[0]], cen[action[1]]
    # 1. truth
    F, _ = cv_FQ(cfg.dt, cfg.q_truth)
    dt, q = cfg.dt, cfg.q_truth
    L1 = jnp.linalg.cholesky(jnp.asarray(q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])))
    w = jax.random.normal(jax.random.fold_in(kt, 0), (cfg.n_tgt, 3, 2), F64) @ L1.T
    truth = s.truth @ F.T + w.reshape(cfg.n_tgt, 6)
    # 2. detections (common random numbers: keyed by scenario and t only)
    z, zvalid, in_beam = detect(truth, look_az, look_el, jax.random.fold_in(kt, 1), cfg)
    n_fa = jnp.int32(0)
    if cfg.n_fa_slots > 0:                    # clutter variant (own key branch: target draws are unchanged)
        assert cfg.associator == "nn", "the GNN enumeration is sized for the target detections only"
        zf, vf = clutter(look_az, look_el, jax.random.fold_in(kt, 2), cfg)
        z, zvalid = jnp.concatenate([z, zf], 0), jnp.concatenate([zvalid, vf], 0)
        n_fa = jnp.sum(vf).astype(jnp.int32)
    # 3. tracker
    tr, next_tid, aux = tracker_step(s.tr, s.next_tid, z, zvalid, t1, cfg)
    # 4. search reward before the push, then push
    ages = (t1 - s.sv_t).astype(F64)
    sv_az = jnp.roll(s.sv_az, 1).at[0].set(look_az)
    sv_el = jnp.roll(s.sv_el, 1).at[0].set(look_el)
    sv_t = jnp.roll(s.sv_t, 1).at[0].set(t1)
    sv_valid = jnp.roll(s.sv_valid, 1).at[0].set(True)
    if cfg.ssv_age0:     # sensitivity: after the push, ages 0..N_SV-1 (the look itself contributes phi_0(0))
        ssv = ssv_at(look_az, look_el, sv_az, sv_el, (t1 - sv_t).astype(F64), sv_valid, cfg)
    else:                # primary: before the push
        ssv = ssv_at(look_az, look_el, s.sv_az, s.sv_el, ages, s.sv_valid, cfg)
    s2 = EnvState(t=t1, scen=s.scen, truth=truth, tr=tr, next_tid=next_tid, sv_az=sv_az, sv_el=sv_el, sv_t=sv_t,
                  sv_valid=sv_valid, last_look=jnp.where(in_beam, t1, s.last_look),
                  n_init=s.n_init + aux["n_new"], n_del=s.n_del + aux["n_kill"], overflow=s.overflow + aux["overflow"],
                  life_hist=s.life_hist + aux["life_add"])
    # 6. metrics (post-step live tracks)
    pn = jnp.sqrt(jnp.sum(tr.P ** 2, axis=(1, 2)))
    n_tr = jnp.sum(tr.valid)
    info = dict(r_sv=-ssv, rtl_flip=aux["rtl_flip"], cov_sum=jnp.sum(jnp.where(tr.valid, pn, 0.0)), n_tracks=n_tr.astype(jnp.int32),
                n_det=jnp.sum(zvalid).astype(jnp.int32), n_hit=aux["n_hit"], min_S_minor=aux["min_S_minor"],
                z=z, zvalid=zvalid, cov_killed=aux["cov_killed"], n_killed=aux["n_kill"])
    if cfg.n_fa_slots > 0:
        info.update(n_fa=n_fa)
    if cfg.gospa_metrics:
        g = gospa(truth, tr.x, tr.valid, cfg.gospa_c, cfg.gospa_p)
        info.update(gospa_d=g[0], gospa_loc=g[1], gospa_miss=g[2], gospa_false=g[3])
        if cfg.gospa_lib:
            # library reading (Stone Soup tutorial pipeline, ss_metric_ref.py): live tracks plus the tracks deleted at
            # this step with their last (predicted) state; the killed slots are taken in slot order, at most 3
            kill = aux["kill"]
            idx = jnp.argsort(~kill, stable=True)[:LIB_KILL_SLOTS]
            tx = jnp.concatenate([tr.x, aux["x_last"][idx]], 0)
            tv = jnp.concatenate([tr.valid, kill[idx]], 0)
            gl = gospa(truth, tx, tv, cfg.gospa_c, cfg.gospa_p)
            info.update(gospa_lib_d=gl[0], gospa_lib_loc=gl[1], gospa_lib_miss=gl[2], gospa_lib_false=gl[3],
                        lib_kill_ovf=jnp.maximum(jnp.sum(kill) - LIB_KILL_SLOTS, 0).astype(jnp.int32))
    return s2, observe(s2, cfg), info


def final_life_hist(s: EnvState) -> jax.Array:
    """Lifetime histogram including tracks still alive at the episode end."""
    add = jnp.zeros((LIFE_BINS,), jnp.int32).at[jnp.minimum(s.tr.n_upd, LIFE_BINS - 1)].add(s.tr.valid.astype(jnp.int32))
    return s.life_hist + add


def env_version(cfg: EnvConfig) -> dict:
    import hashlib
    import json
    from dataclasses import asdict
    with open(__file__, "rb") as f:
        h_src = hashlib.sha256(f.read()).hexdigest()
    h_cfg = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()
    return dict(data_py_sha256=h_src, env_cfg_sha256=h_cfg)
