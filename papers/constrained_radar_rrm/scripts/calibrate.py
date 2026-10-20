"""Calibration of the two constants the paper never gives (frozen BEFORE any learner training).

SNR0 (A-02): solves mean_{t in [500,1500]} c_ss(0.9 T0, r_T2(t); SNR0) = Fig. 8's single-target
  baseline curve averaged over the same slots (digitised here from the page render at 700 dpi; EA = DA = OA for
  one track, so the visible purple/green line is the EA curve).
S_ref (A-04): scan SNR at tau_s = 0.25 s, r = 10 km. Least squares on log T_init over the nine Fig. 9c scan shares
  of the POOLED EA/DA/OA mean, simulated with EA at tracking share 1 - delta_s on 50 spawn-model episodes
  (10000 slots, n0 = 1) from k_calib = fold_in(root, 0xCA1B). Coarse log grid, then a fine grid.
Writes src/configs/calibration.json. Usage: python scripts/calibrate.py [--episodes 50]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import EnvCfg, info, load_yaml, metric, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from src import data, eval as ev, physics as ph  # noqa: E402

# Re-evaluation: the frozen calibration was computed on the v1 replica with the legacy slot-key tree; keep both so that
# this script reproduces calibration.json. The refit sensitivity is
# scripts/calib_t6_sens.py.
data.REPLICA = "v1"
ev.SLOT_KEY = "legacy"

# Fig. 8 baseline curve (single target), digitised 2026-10-19 from the 700-dpi render of page 12:
# column-median of the purple (DA) / green (OA) dashed-line pixels; axis 201 px per 1e4 m^2, 155 px per 1000 slots.
FIG8_T = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
FIG8_C = [5075, 4353, 3706, 3247, 2736, 2413, 2114, 2214, 1617, 1418, 1219]
# Fig. 9c pooled EA/DA/OA mean T_init (paper Fig. 9c, read +-0.5) at scan shares 0.1..0.9
FIG9C_DS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FIG9C = {"EA": [29.5, 10.0, 6.5, 5.0, 6.3, 4.3, 4.3, 4.7, 3.8], "DA": [18.7, 8.0, 6.3, 8.6, 8.4, 4.8, 4.8, 3.5, 4.2],
         "OA": [13.7, 16.8, 11.2, 6.6, 7.1, 5.6, 3.9, 6.3, 4.8]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--root_seed", type=int, default=20261019)
    a = ap.parse_args()
    common.check_device()
    y = load_yaml()
    cfg = EnvCfg.from_yaml(y)
    lut = ev.make_lut(cfg)

    # ---------------- SNR0
    scen = data.replica_case1(cfg)
    pos = np.array([np.asarray(data.truth(scen, t, cfg)[0][0]) for t in FIG8_T])
    r = np.hypot(pos[:, 0], pos[:, 1])
    target = float(np.mean(FIG8_C))
    f = lambda s: float(np.mean([ph.c_ss(0.9 * cfg.T0, rr, s, cfg) for rr in r])) - target
    snr0 = brentq(f, 0.01, 1000.0, xtol=1e-10)
    info("SNR0 calibrated", snr0=snr0, fig8_mean=target, ranges_km=f"{r.min()/1e3:.2f}-{r.max()/1e3:.2f}")

    # ---------------- S_ref
    pooled = np.mean([FIG9C[k] for k in FIG9C], 0)
    root = jax.random.PRNGKey(a.root_seed)
    k_cal = jax.random.fold_in(root, 0xCA1B)
    E_ = a.episodes
    ekeys = jax.vmap(lambda i: jax.random.fold_in(k_cal, i))(jnp.arange(E_))
    scens = jax.vmap(lambda k: data.spawn_scenario(jax.random.fold_in(k, 0x5A), cfg, 10000, 1))(ekeys)
    pol = ev.make_policy("ea", cfg, y)
    fe = ev.make_final_evaluator(cfg, lut, pol, 10000)

    def sim(s_ref):
        """pooled T_init per scan share for one S_ref value: lanes = 9 shares x E episodes."""
        D = len(FIG9C_DS)
        consts = dict(snr0=jnp.full((D * E_,), snr0), s_ref=jnp.full((D * E_,), s_ref),
                      delta=jnp.repeat(1.0 - jnp.asarray(FIG9C_DS), E_))
        sc = jax.tree.map(lambda x: jnp.tile(x, (D,) + (1,) * (x.ndim - 1)), scens)
        ks = jnp.tile(ekeys, (D, 1))
        out = jax.device_get(fe(None, consts, sc, ks))
        ts = out["tinit_sum"].reshape(D, E_).sum(1)
        tn = out["tinit_n"].reshape(D, E_).sum(1)
        return ts / np.maximum(tn, 1), out["never"].reshape(D, E_).sum(1)

    def loss(ti):
        return float(np.sum((np.log(ti) - np.log(pooled)) ** 2))

    t0 = time.perf_counter()
    rows = []
    for s in np.logspace(0, 3, 13):
        ti, nv = sim(float(s))
        rows.append((float(s), loss(ti), ti.tolist(), nv.tolist()))
        metric("S_ref coarse", s_ref=float(s), loss=rows[-1][1], tinit=" ".join(f"{x:.2f}" for x in ti),
               never=int(nv.sum()), wall=time.perf_counter() - t0)
    best = min(rows, key=lambda r_: r_[1])[0]
    for s in np.logspace(np.log10(best) - 0.25, np.log10(best) + 0.25, 11):
        ti, nv = sim(float(s))
        rows.append((float(s), loss(ti), ti.tolist(), nv.tolist()))
        metric("S_ref fine", s_ref=float(s), loss=rows[-1][1], tinit=" ".join(f"{x:.2f}" for x in ti),
               never=int(nv.sum()))
    s_ref, l_best, ti_best, nv_best = min(rows, key=lambda r_: r_[1])
    info("S_ref calibrated", s_ref=s_ref, loss=l_best)
    out = dict(snr0=snr0, s_ref=s_ref,
               snr0_rule="A-02: mean c_ss(0.9 T0, r_T2(t)) over Fig. 8 slots = digitised Fig. 8 mean",
               fig8_t=FIG8_T, fig8_c=FIG8_C, fig8_mean=target, fig8_ranges_m=r.tolist(),
               s_ref_rule="A-04: least squares on log T_init vs pooled Fig. 9c (EA sim, 50 spawn episodes, k_calib)",
               fig9c_pooled=pooled.tolist(), sim_tinit=ti_best, sim_never=nv_best, loss=l_best,
               grid=[dict(s_ref=r_[0], loss=r_[1], tinit=r_[2], never=r_[3]) for r_ in rows],
               episodes=E_, root_seed=a.root_seed, date="2026-10-19")
    write_json(os.path.join(common.CFG_DIR, "calibration.json"), out)
    info("wrote calibration.json", wall_s=time.perf_counter() - t0)


if __name__ == "__main__":
    main()
