"""Oracle harness (reference-code check): filterpy 1.4.5 ExtendedKalmanFilter on 3 polar-measurement sequences.

Labelled harness parts (not filterpy code): the CV truth, the measurement draws and the per-step R from the paper's
model (eq. 5-6 as division). filterpy does predict (x = F x, P = F P F^T + Q) and update (Joseph form) with
HJacobian/Hx callables and an angle-wrapping residual. Writes fixtures/filterpy_ekf.npz.
"""
import os

import numpy as np
import filterpy
from filterpy.kalman import ExtendedKalmanFilter

T, sw2 = 2.5, 16.0
F = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1.0]])
q = sw2 * np.array([[T**4 / 4, T**3 / 2], [T**3 / 2, T**2]])
Q = np.zeros((4, 4)); Q[np.ix_([0, 2], [0, 2])] = q; Q[np.ix_([1, 3], [1, 3])] = q


def Hx(x):
    return np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])])


def HJ(x):
    r2 = x[0]**2 + x[1]**2; r = np.sqrt(r2)
    return np.array([[x[0] / r, x[1] / r, 0, 0], [-x[1] / r2, x[0] / r2, 0, 0]])


def resid(a, b):
    d = a - b
    d[1] = np.pi - np.mod(np.pi - d[1], 2 * np.pi)
    return d


rng = np.random.default_rng(20261019)
out = {}
for s in range(3):
    n = 200
    p0 = rng.uniform(-15000, 15000, 2); v = rng.uniform(-2, 2, 2)
    truth = np.array([np.r_[p0 + v * T * k, v] for k in range(1, n + 1)])
    snr0 = [1.0, 2.8, 10.0][s]
    taus = rng.uniform(0.05, 2.25, n)
    r = np.hypot(truth[:, 0], truth[:, 1])
    snr = snr0 * taus * (r / 3000.0) ** -4
    var_r, var_th = 16.0 / snr, 1e-6 / snr
    z = np.stack([r + np.sqrt(var_r) * rng.standard_normal(n),
                  np.arctan2(truth[:, 1], truth[:, 0]) + np.sqrt(var_th) * rng.standard_normal(n)], -1)
    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
    ekf.x = np.r_[p0 + rng.normal(0, 50, 2), v + rng.normal(0, 1, 2)]
    ekf.P = np.diag([1e4, 1e4, 100, 100.0]); ekf.F = F; ekf.Q = Q
    x0, P0 = ekf.x.copy(), ekf.P.copy()
    xs, Ps = [], []
    for k in range(n):
        ekf.predict()
        ekf.update(z[k], HJ, Hx, R=np.diag([var_r[k], var_th[k]]), residual=resid)
        xs.append(ekf.x.copy()); Ps.append(ekf.P.copy())
    out.update({f"s{s}_x0": x0, f"s{s}_P0": P0, f"s{s}_z": z, f"s{s}_var_r": var_r, f"s{s}_var_th": var_th,
                f"s{s}_x": np.array(xs), f"s{s}_P": np.array(Ps)})
out["filterpy_version"] = np.array(filterpy.__version__)
FIX = os.path.join(os.environ.get("OFFICIAL_ROOT", "external/official"), "fixtures")
os.makedirs(FIX, exist_ok=True)
np.savez(os.path.join(FIX, "filterpy_ekf.npz"), **out)
print("wrote", os.path.join(FIX, "filterpy_ekf.npz"), "filterpy", filterpy.__version__)
