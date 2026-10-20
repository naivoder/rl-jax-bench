"""SENSITIVITY calibration of S_ref (NOT the declared A-04 rule; recorded as amendment A-04a, pilot):
anchor = Table VI's pooled EA/DA/OA track-initiation time on the Case I replica, (10.50 + 14.67 + 20.33) / 3 = 15.17
slots (equal in expectation under). EA at delta = 0.9 on the replica, 20 noise draws from k_calib
(fold_in(k_calib, 5000 + i)), pooled mean T_init; log-grid then bisection on log S_ref (T_init is decreasing in S_ref).
Adds s_ref_table6 (+ its grid) to calibration.json without touching the frozen s_ref.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import EnvCfg, info, load_calibration, load_yaml, metric, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import data, eval as ev  # noqa: E402

# Re-evaluation: the frozen calibration was computed on the v1 replica with the legacy slot-key tree; keep both so that
# this script reproduces calibration.json. The refit sensitivity is
# scripts/calib_t6_sens.py.
data.REPLICA = "v1"
ev.SLOT_KEY = "legacy"

TARGET = (10.50 + 14.67 + 20.33) / 3


def main():
    common.check_device()
    y = load_yaml()
    cfg = EnvCfg.from_yaml(y)
    cal = load_calibration()
    lut = ev.make_lut(cfg)
    root = jax.random.PRNGKey(20261019)
    k_cal = jax.random.fold_in(root, 0xCA1B)
    R = 20
    keys = jnp.stack([jax.random.fold_in(k_cal, 5000 + i) for i in range(R)])
    rep = data.replica_case1(cfg)
    sc = jax.tree.map(lambda x: jnp.broadcast_to(x, (R,) + x.shape), rep)
    fe = ev.make_final_evaluator(cfg, lut, ev.make_policy("ea", cfg, y), 10000)

    def sim(s):
        consts = dict(snr0=jnp.full((R,), cal["snr0"]), s_ref=jnp.full((R,), s), delta=jnp.full((R,), 0.9))
        o = jax.device_get(fe(None, consts, sc, keys))
        return float(o["tinit_sum"].sum() / max(o["tinit_n"].sum(), 1)), int(o["never"].sum())

    grid = []
    lo, hi = np.log(1.0), np.log(1e4)
    for s in np.exp(np.linspace(lo, hi, 13)):
        ti, nv = sim(float(s))
        grid.append(dict(s_ref=float(s), tinit=ti, never=nv))
        metric("table6 grid", s_ref=float(s), tinit=ti, never=nv)
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        ti, nv = sim(float(np.exp(mid)))
        grid.append(dict(s_ref=float(np.exp(mid)), tinit=ti, never=nv))
        lo, hi = (mid, hi) if ti > TARGET else (lo, mid)
    s6 = float(np.exp(0.5 * (lo + hi)))
    ti, nv = sim(s6)
    info("S_ref (Table VI anchor)", s_ref_table6=s6, tinit=ti, never=nv, target=TARGET)
    cal.update(s_ref_table6=s6, s_ref_table6_tinit=ti, s_ref_table6_rule=__doc__.strip().splitlines()[0],
               s_ref_table6_grid=grid)
    write_json(os.path.join(common.CFG_DIR, "calibration.json"), cal)


if __name__ == "__main__":
    main()
