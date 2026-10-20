"""The re-evaluation sensitivity of the A-04a anchor: S_ref_table6 = the S_ref at which EA's pooled Case I
replica T_init equals Table VI's pooled 15.17, re-solved under
  (a) the v1 replica + legacy keys + 'gaps' convention  -> must reproduce calibration.json's 32.278 (check);
  (b) the re-digitised replica + v2 keys, 'gaps' (conf_slot - born; VVV from birth = 2);
  (c) as (b), 'scans' convention (gaps + 1 per confirmed target);
  (d) as (c), never-confirmed targets included, censored at the episode end.
Method: EA at delta 0.9 on 20 k_calib draws (as scripts/calibrate_table6.py) on a 49-point log grid of S_ref in
[3, 300], then log-linear interpolation of the (monotone-smoothed) curve at 15.17. Writes runs/full_v2/calib_t6_sens.json.
The frozen calibration is NOT changed (learners were trained under it)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import EnvCfg, info, load_calibration, load_yaml, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import data, eval as ev  # noqa: E402

TARGET = (10.50 + 14.67 + 20.33) / 3
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def curve(y, cal, replica, scheme, grid):
    data.REPLICA, ev.SLOT_KEY = replica, scheme
    cfg = EnvCfg.from_yaml(y)
    lut = ev.make_lut(cfg)
    root = jax.random.PRNGKey(20261019)
    k_cal = jax.random.fold_in(root, 0xCA1B)
    R = 20
    keys = jnp.stack([jax.random.fold_in(k_cal, 5000 + i) for i in range(R)])
    rep = data.replica_case1(cfg)
    S = len(grid)
    sc = jax.tree.map(lambda x: jnp.broadcast_to(x, (S * R,) + x.shape), rep)
    fe = ev.make_final_evaluator(cfg, lut, ev.make_policy("ea", cfg, y), 10000)
    consts = dict(snr0=jnp.full((S * R,), cal["snr0"]), s_ref=jnp.repeat(jnp.asarray(grid), R),
                  delta=jnp.full((S * R,), 0.9))
    o = jax.device_get(fe(None, consts, sc, jnp.tile(keys, (S, 1))))
    rs = lambda k: np.asarray(o[k], float).reshape(S, R).sum(1)
    gaps_s, n, never, cens = rs("tinit_sum"), rs("tinit_n"), rs("never"), rs("tinit_cens_sum")
    return dict(gaps=gaps_s / np.maximum(n, 1), scans=(gaps_s + n) / np.maximum(n, 1),
                scans_cens=(gaps_s + n + cens + never) / np.maximum(n + never, 1), never=never.tolist())


def solve(grid, t):
    lg = np.log(grid)
    tm = np.minimum.accumulate(np.asarray(t))          # T_init decreases in S_ref; smooth the noise monotone
    i = np.where(tm <= TARGET)[0]
    if len(i) == 0 or i[0] == 0:
        return float("nan")
    j = i[0]
    f = (tm[j - 1] - TARGET) / max(tm[j - 1] - tm[j], 1e-12)
    return float(np.exp(lg[j - 1] + f * (lg[j] - lg[j - 1])))


def main():
    common.check_device()
    y, cal = load_yaml(), load_calibration()
    grid = np.exp(np.linspace(np.log(3.0), np.log(300.0), 49))
    out = dict(target=TARGET, frozen_s_ref_table6=cal["s_ref_table6"], grid=grid.tolist())
    a = curve(y, cal, "v1", "legacy", grid)
    b = curve(y, cal, "v2", "v2", grid)
    out["a_v1_legacy_gaps"] = dict(curve=a["gaps"].tolist(), s_ref=solve(grid, a["gaps"]))
    out["b_v2_gaps"] = dict(curve=b["gaps"].tolist(), s_ref=solve(grid, b["gaps"]))
    out["c_v2_scans"] = dict(curve=b["scans"].tolist(), s_ref=solve(grid, b["scans"]))
    out["d_v2_scans_censored"] = dict(curve=b["scans_cens"].tolist(), s_ref=solve(grid, b["scans_cens"]),
                                      never=b["never"])
    for k in ("a_v1_legacy_gaps", "b_v2_gaps", "c_v2_scans", "d_v2_scans_censored"):
        info("table6 refit", variant=k, s_ref=out[k]["s_ref"], ratio_to_frozen=out[k]["s_ref"] / cal["s_ref_table6"])
    write_json(os.path.join(REPO, "runs", "full_v2", "calib_t6_sens.json"), out)


if __name__ == "__main__":
    main()
