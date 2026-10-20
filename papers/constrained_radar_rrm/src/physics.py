"""Radar physics in float64: CV motion model, polar EKF, measurement-noise model, Kalata steady state, Shnidman P_d.

Pure jnp functions (jit/vmap at the call site). Equation numbers are the paper's.
State order x = [x, y, xdot, ydot] (eq. 14's P0 = diag(1e4, 1e4, 100, 100)).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# ----------------------------------------------------------------------------- motion model (eqs. 1-3)


def F_mat(T: float) -> jax.Array:
    return jnp.array([[1.0, 0, T, 0], [0, 1.0, 0, T], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])


def Q_mat(T: float, sigma_w_sq: float) -> jax.Array:
    """DWNA process noise sigma_w^2 Gamma Gamma^T per axis, Gamma = [T^2/2, T] (eq. 3)."""
    q = sigma_w_sq * jnp.array([[T ** 4 / 4, T ** 3 / 2], [T ** 3 / 2, T ** 2]])
    Z = jnp.zeros((4, 4))
    Z = Z.at[jnp.ix_(jnp.array([0, 2]), jnp.array([0, 2]))].set(q)
    Z = Z.at[jnp.ix_(jnp.array([1, 3]), jnp.array([1, 3]))].set(q)
    return Z


def predict(x, P, T, sigma_w_sq):
    F = F_mat(T)
    return F @ x, F @ P @ F.T + Q_mat(T, sigma_w_sq)


# ----------------------------------------------------------------------------- measurement model (eqs. 4-8)


def h_polar(x) -> jax.Array:
    """h(x) = [range, bearing]; atan2 for the full circle (A-10)."""
    return jnp.array([jnp.hypot(x[0], x[1]), jnp.arctan2(x[1], x[0])])


def H_jac(x) -> jax.Array:
    """Eq. 7 evaluated at the prior estimate."""
    r2 = x[0] ** 2 + x[1] ** 2
    r = jnp.sqrt(r2)
    return jnp.array([[x[0] / r, x[1] / r, 0.0, 0.0], [-x[1] / r2, x[0] / r2, 0.0, 0.0]])


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return jnp.pi - jnp.mod(jnp.pi - a, 2 * jnp.pi)


def track_snr(tau, r, snr0, r0, tau0, r_min):
    """Eq. 5: SNR = SNR0 (tau/tau0) (r/r0)^-4 (range floored at r_min, A-52)."""
    rr = jnp.maximum(r, r_min)
    return snr0 * (tau / tau0) * (rr / r0) ** (-4)


def meas_var(snr, sigma_r0_sq, sigma_th0_sq):
    """Eq. 6 read as a DIVISION (A-01): sigma^2 = sigma_0^2 / SNR. snr must be > 0."""
    return sigma_r0_sq / snr, sigma_th0_sq / snr


def ekf_update(xp, Pp, z, var_r, var_th):
    """Eqs. 11-13 (covariance form as printed), angle innovation wrapped (A-10)."""
    H = H_jac(xp)
    R = jnp.diag(jnp.array([var_r, var_th]))
    S = H @ Pp @ H.T + R
    K = Pp @ H.T @ jnp.linalg.inv(S)
    nu = z - h_polar(xp)
    nu = nu.at[1].set(wrap(nu[1]))
    x = xp + K @ nu
    P = (jnp.eye(4) - K @ H) @ Pp
    P = 0.5 * (P + P.T)
    return x, P


def pos_trace(P):
    """Eqs. 15-16: c = trace(E P E^T), E selects the position block."""
    return P[..., 0, 0] + P[..., 1, 1]


# ----------------------------------------------------------------------------- Kalata steady state


def kalata_alpha(Lam):
    """Steady-state alpha of the alpha-beta filter for DWNA with tracking index Lam (Kalata 1984)."""
    s = jnp.sqrt(Lam ** 2 + 8 * Lam)
    return -(Lam ** 2 + 8 * Lam - (Lam + 4) * s) / 8


def c_ss(tau, r, snr0, cfg, factor=1.0):
    """ steady-state POSTERIOR position-trace of a filter linearised at range r with dwell tau (A-32).
    factor multiplies both measurement variances (Case IV, A-38)."""
    snr = track_snr(tau, r, snr0, cfg.r0, cfg.tau0, cfg.r_min_snr)
    var_r, var_th = meas_var(snr, cfg.sigma_r0_sq, cfg.sigma_th0_sq)
    var_r, var_th = var_r * factor, var_th * factor
    sw = jnp.sqrt(cfg.sigma_w_sq)
    T = cfg.T0
    rr = jnp.maximum(r, cfg.r_min_snr)
    s_rad = jnp.sqrt(var_r)
    s_cr = rr * jnp.sqrt(var_th)
    return kalata_alpha(sw * T ** 2 / s_rad) * s_rad ** 2 + kalata_alpha(sw * T ** 2 / s_cr) * s_cr ** 2


# ----------------------------------------------------------------------------- Shnidman P_d LUT


def shnidman_snr(pd, pfa, n_pulses: int = 1, swerling: int = 0):
    """Shnidman (2002) closed form; Swerling 0 (C = 1), a = 0 for N < 40 (A-05). Returns linear SNR per pulse."""
    assert swerling == 0
    pd = np.asarray(pd, dtype=np.float64)
    eta = np.sqrt(-0.8 * np.log(4 * pfa * (1 - pfa))) + np.sign(pd - 0.5) * np.sqrt(-0.8 * np.log(4 * pd * (1 - pd)))
    X = eta * (eta + 2 * np.sqrt(n_pulses / 2 + (0.0 - 0.25)))
    return X / n_pulses


PD_GRID = np.round(np.arange(1, 100) * 0.01, 2)   # paper §III-B: 0.01..0.99


def make_lut(pfa: float):
    """LUT (P_d grid, SNR in dB) built once on the host."""
    return jnp.asarray(PD_GRID), jnp.asarray(10 * np.log10(shnidman_snr(PD_GRID, pfa)))


def pd_lookup(snr, lut):
    """Nearest LUT entry in dB (A-05); P_d quantised to 0.01 and clamped to [0.01, 0.99]. snr <= 0 -> 0.01."""
    pd_grid, snr_db = lut
    q = 10 * jnp.log10(jnp.maximum(snr, 1e-300))
    idx = jnp.argmin(jnp.abs(snr_db[None, :] - jnp.atleast_1d(q)[:, None]), axis=-1)
    return pd_grid[idx].reshape(jnp.shape(snr))


def scan_snr(tau_s, r, s_ref, r_min):
    """: SNR_scan = K_s tau_s r^-4, parameterised by S_ref = SNR at tau_s = 0.25 s, r = 10 km (A-04)."""
    rr = jnp.maximum(r, r_min)
    return s_ref * (tau_s / 0.25) * (rr / 1.0e4) ** (-4)


# ----------------------------------------------------------------------------- 3-of-4 initiation chain
# states: 0 E, 1 V, 2 VV, 3 VX, 4 VVX, 5 VXV; 6 = CONFIRM
NEXT_DET = jnp.array([1, 2, 6, 5, 6, 6])
NEXT_MISS = jnp.array([0, 3, 4, 0, 0, 3])


def expected_scans(p: float) -> float:
    """Exact E[scans to confirmation] from E for constant per-scan P_d p (absorbing chain)."""
    q = 1 - p
    A = np.zeros((6, 6))
    b = np.ones(6)
    nd, nm = np.asarray(NEXT_DET), np.asarray(NEXT_MISS)
    for s in range(6):
        A[s, s] += 1
        if nd[s] != 6:
            A[s, nd[s]] -= p
        A[s, nm[s]] -= q
    return float(np.linalg.solve(A, b)[0])
