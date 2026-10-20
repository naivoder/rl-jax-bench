"""Oracle harness (reference-code check, second-best references; the authors released no code).

Inputs: fixtures/env_rollout_*.npz written by OUR repo (scripts/oracle_dump_env.py): the measurement stream of our
track-while-scan environment (z, R, update flags, track births) on 1e4-slot Case I replica rollouts.
Reference code run here (own venv: filterpy 1.4.5, scipy):
  (a) filterpy ExtendedKalmanFilter re-filters every track from the recorded births and measurements
      -> per-slot position traces c_n, tracking term and U = -sum w c - beta N_miss (paper eqs. 15-16, 19).
  (b) scipy.linalg.solve_discrete_are steady-state POSTERIOR position trace of the polar EKF linearised at range r
      (the OA cost, A-32) on a (tau, r) grid -- independent of our Kalata closed form.
  (c) scipy.optimize.minimize (SLSQP) solves the OA problem min sum w c_ss_DARE(tau_n, r_hat_n) s.t. sum tau =
      delta T0, tau >= 0 on 200 recorded OA decision states (>= 2 tracks) -- independent of our Newton solver.
  (d) scipy.stats.ncx2 exact single-pulse Swerling-0 square-law P_d(SNR; P_fa) on an SNR grid (Shnidman check).
Labelled harness parts (not reference code): reading our npz, the loop that feeds filterpy, the DARE linearisation
(H at bearing 0, which is rotation-invariant for the isotropic DWNA Q), the SLSQP problem statement.
Writes fixtures/oracle_env_outputs.npz.
"""
import os
import sys

import numpy as np
import filterpy
import scipy
from filterpy.kalman import ExtendedKalmanFilter
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from scipy.stats import ncx2

FIX = os.path.join(os.environ.get("OFFICIAL_ROOT", "external/official"), "fixtures")


def FQ(T, sw2):
    F = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    q = sw2 * np.array([[T**4 / 4, T**3 / 2], [T**3 / 2, T**2]])
    Q = np.zeros((4, 4)); Q[np.ix_([0, 2], [0, 2])] = q; Q[np.ix_([1, 3], [1, 3])] = q
    return F, Q


def Hx(x):
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])


def HJ(x):
    r2 = x[0]**2 + x[1]**2; r = np.sqrt(r2)
    return np.array([[x[0] / r, x[1] / r, 0, 0], [-x[1] / r2, x[0] / r2, 0, 0]])


def resid(a, b):
    d = a - b
    d[1] = np.pi - np.mod(np.pi - d[1], 2 * np.pi)
    return d


def replay_filterpy(f):
    m = {k[5:]: f[k] for k in f.files if k.startswith("meta_")}
    F, Q = FQ(float(m["T0"]), float(m["sigma_w_sq"]))
    T, K = f["act_pre"].shape
    filt = [None] * K
    c = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            if f["act_pre"][t, k]:
                e = filt[k]
                e.predict()
                if f["upd"][t, k]:
                    e.update(f["z"][t, k], HJ, Hx, R=np.diag([f["var_r"][t, k], f["var_th"][t, k]]),
                             residual=resid)
            if f["newk"][t, k]:
                e = ExtendedKalmanFilter(dim_x=4, dim_z=2)
                e.F, e.Q = F, Q
                e.x = f["x_init"][t, k].copy()
                e.P = np.diag(m["P0_diag"]).astype(float)
                filt[k] = e
            if f["act_new"][t, k]:
                c[t, k] = filt[k].P[0, 0] + filt[k].P[1, 1]
    U = -(f["w"] * c).sum(-1) - float(m["beta"]) * f["nmiss"]
    return c, U


def _dare_axis(T, sw2, var):
    """2-state [position, velocity] DARE along one axis, normalised by the measurement variance (P scales
    linearly when Q and R are scaled together); returns the steady-state POSTERIOR position variance."""
    F = np.array([[1.0, T], [0.0, 1.0]])
    Q = sw2 / var * np.array([[T**4 / 4, T**3 / 2], [T**3 / 2, T**2]])
    H = np.array([[1.0, 0.0]])
    P = solve_discrete_are(F.T, H.T, Q, np.array([[1.0]]))         # prior, units of var
    Pp = P - P @ H.T @ H @ P / (H @ P @ H.T + 1.0)
    return var * Pp[0, 0]


def c_ss_dare(tau, r, m):
    """Steady-state POSTERIOR position trace of the polar EKF linearised at range r (A-32). In the frame rotated to
    the bearing the 4-state problem with isotropic DWNA Q decouples exactly into a radial axis (variance var_r) and a
    cross-range axis (variance r^2 var_th), each solved with scipy's DARE (the full 4-state DARE is too
    ill-conditioned at R/Q ratios of 1e8 for scipy's QZ reordering)."""
    T, sw2 = float(m["T0"]), float(m["sigma_w_sq"])
    rr = max(r, float(m["r_min_snr"]))
    snr = float(m["snr0"]) * (tau / float(m["tau0"])) * (rr / float(m["r0"])) ** (-4)
    var_r, var_th = float(m["sigma_r0_sq"]) / snr, float(m["sigma_th0_sq"]) / snr
    return _dare_axis(T, sw2, var_r) + _dare_axis(T, sw2, rr**2 * var_th)


def c_ss_dare4(tau, r, m):
    """Full 4-state DARE at bearing 0 (cross-check of the decoupling where scipy can solve it)."""
    T, sw2 = float(m["T0"]), float(m["sigma_w_sq"])
    F, Q = FQ(T, sw2)
    rr = max(r, float(m["r_min_snr"]))
    snr = float(m["snr0"]) * (tau / float(m["tau0"])) * (rr / float(m["r0"])) ** (-4)
    R = np.diag([float(m["sigma_r0_sq"]) / snr, float(m["sigma_th0_sq"]) / snr])
    H = np.array([[1.0, 0, 0, 0], [0, 1.0 / rr, 0, 0]])
    P = solve_discrete_are(F.T, H.T, Q, R)
    Pp = P - P @ H.T @ np.linalg.solve(H @ P @ H.T + R, H @ P)
    return Pp[0, 0] + Pp[1, 1]


def oa_slsqp(r_hat, w, m):
    D = float(m["delta"]) * float(m["T0"])
    n = len(r_hat)
    f0 = sum(wi * c_ss_dare(D / n, ri, m) for wi, ri in zip(w, r_hat))
    obj = lambda tau: sum(wi * c_ss_dare(max(ti, 1e-12), ri, m) for wi, ri, ti in zip(w, r_hat, tau)) / f0
    res = minimize(obj, np.full(n, D / n), method="SLSQP", bounds=[(1e-6, D)] * n,
                   constraints=[{"type": "eq", "fun": lambda tau: tau.sum() - D}],
                   options=dict(ftol=1e-14, maxiter=500))
    return res.x, res.fun * f0, res.success


def main():
    out = dict(filterpy_version=np.array(filterpy.__version__), scipy_version=np.array(scipy.__version__))
    for tag in ("ea_s_ref", "oa_s_ref_table6"):
        f = np.load(f"{FIX}/env_rollout_{tag}.npz")
        c, U = replay_filterpy(f)
        out[f"{tag}_c"], out[f"{tag}_U"] = c, U
        print(tag, "filterpy mean U", U.mean(), "ours", f["U"].mean(),
              "max rel c diff", np.max(np.abs(c - f["c"]) / np.maximum(np.abs(f["c"]), 1.0)))
    # (b) DARE grid
    m = {k[5:]: f[k] for k in f.files if k.startswith("meta_")}
    taus = np.array([0.02, 0.1, 0.3, 0.75, 1.5, 2.25])
    rs = np.array([500.0, 3000.0, 7000.0, 12000.0, 18000.0])
    out["dare_tau"], out["dare_r"] = taus, rs
    out["dare_c"] = np.array([[c_ss_dare(t, r, m) for r in rs] for t in taus])
    d4 = []
    for t in taus:
        for r in rs:
            try:
                d4.append(abs(c_ss_dare4(t, r, m) / c_ss_dare(t, r, m) - 1))
            except Exception:
                pass
    print("4-state DARE vs decoupled: solved", len(d4), "of", taus.size * rs.size, "max rel", max(d4))
    out["dare4_maxrel"], out["dare4_n"] = np.array(max(d4)), np.array(len(d4))
    # (c) OA by SLSQP on recorded OA decision states
    f = np.load(f"{FIX}/env_rollout_oa_s_ref_table6.npz")
    multi = np.where(f["act_view"].sum(-1) >= 2)[0]
    pick = multi[np.linspace(0, len(multi) - 1, 200).astype(int)]
    tau_ref, obj_ref, ok = [], [], []
    for t in pick:
        a = f["act_view"][t]
        x, fv, s = oa_slsqp(f["r_hat"][t][a], f["w_pre"][t][a], m)
        full = np.zeros(a.shape); full[a] = x
        tau_ref.append(full); obj_ref.append(fv); ok.append(s)
    out["oa_slots"], out["oa_tau"], out["oa_obj"], out["oa_ok"] = pick, np.array(tau_ref), np.array(obj_ref), np.array(ok)
    out["oa_obj_ours"] = np.array([sum(f["w_pre"][t][k] * c_ss_dare(f["tau"][t][k], f["r_hat"][t][k], m)
                                       for k in range(f["tau"].shape[1]) if f["act_view"][t][k]) for t in pick])
    print("OA slsqp success", np.mean(ok), "max |tau ours - slsqp| / D",
          np.max(np.abs(f["tau"][pick] - out["oa_tau"])) / (0.9 * 2.5),
          "max rel obj (ours - slsqp)/slsqp", np.max((out["oa_obj_ours"] - out["oa_obj"]) / out["oa_obj"]))
    # (d) exact P_d
    snr_db = np.linspace(-5.0, 20.0, 101)
    pfa = 1e-3
    out["pd_snr_db"] = snr_db
    out["pd_exact"] = ncx2.sf(-2 * np.log(pfa), 2, 2 * 10 ** (snr_db / 10))
    np.savez(f"{FIX}/oracle_env_outputs.npz", **out)
    print("wrote", f"{FIX}/oracle_env_outputs.npz", "filterpy", filterpy.__version__, "scipy", scipy.__version__)


if __name__ == "__main__":
    sys.exit(main())
