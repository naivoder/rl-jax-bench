"""The re-evaluation sensitivity: the tracking channel (SNR0, A-02, fitted to Fig. 8) and the scan channel
(S_ref, A-04, fitted to Fig. 9c) are two independent fits of what eqs. 5 and 18 describe as ONE radar. This sweeps each
constant over a decade and re-reads the scan-vs-track verdicts on the baselines:
  - the best fixed tracking share for EA / DA / OA over the Fig. 9 grid (scan share 0.1..0.9),
  - the scan term's share of |U| at delta = 0.9 (P2),
  - T_init at delta = 0.9 and the never-confirmed count,
  - the ordering EA < DA < OA at delta = 0.9.
Configs: S_ref x {0.1, 0.316, 1, 3.16, 10} at the frozen SNR0; SNR0 x {0.316, 3.16} at the frozen S_ref; S_ref_tied =
the scan SNR implied by the TRACKING constant at the S_ref reference point (SNR0 (0.25 s / 1 s) (10 km / 3 km)^-4,
i.e. a whole-scan beam with phi/360 = 1, the most favourable tie). Replica (v2) k_test draws 0..5, v2 keys.
Writes runs/full_v2/sens_sref_snr0.json."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import common  # noqa: E402
from src.common import EnvCfg, info, load_calibration, load_yaml, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import data, eval as ev  # noqa: E402
import full_eval as G  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ND = 6


def main():
    common.check_device()
    y, cal = load_yaml(), load_calibration()
    cfg = EnvCfg.from_yaml(y)
    lut = ev.make_lut(cfg)
    rk, _ = G.keys(jax.random.PRNGKey(20261019))
    keys = rk[:ND]
    scen = G.bcast(data.replica_case1(cfg), ND)
    s0, r0 = cal["snr0"], cal["s_ref"]
    tied = s0 * (0.25 / 1.0) * (1.0e4 / 3.0e3) ** -4
    cfgs = {f"sref_x{m}": (s0, r0 * m) for m in (0.1, 0.316, 1.0, 3.16, 10.0)}
    cfgs.update({f"snr0_x{m}": (s0 * m, r0) for m in (0.316, 3.16)})
    cfgs["sref_tied"] = (s0, tied)
    cfgs["sref_A04a"] = (s0, cal["s_ref_table6"])
    shares = np.round(np.arange(1, 10) * 0.1, 1)
    deltas = jnp.asarray(1.0 - shares)
    out = dict(scan_share=shares.tolist(), configs={k: dict(snr0=v[0], s_ref=v[1]) for k, v in cfgs.items()},
               s_ref_tied=tied, draws=ND)
    for kind in ("ea", "da", "oa"):
        one = ev.make_evaluator(cfg, lut, ev.make_policy(kind, cfg, y), G.T)
        f = jax.jit(jax.vmap(jax.vmap(lambda c_, d: one(None, dict(c_, delta=d), scen, keys), in_axes=(None, 0)),
                             in_axes=(0, None)))
        names = list(cfgs)
        C = dict(snr0=jnp.asarray([cfgs[n][0] for n in names]), s_ref=jnp.asarray([cfgs[n][1] for n in names]))
        r = jax.device_get(f(C, deltas))                        # [n_cfg, 9 shares, ND]
        for i, n in enumerate(names):
            U = np.asarray(r["U"][i]).mean(-1)
            sc = np.asarray(r["scan_term"][i]).mean(-1)
            ti = np.asarray(r["tinit_sum"][i]).sum(-1) / np.maximum(np.asarray(r["tinit_n"][i]).sum(-1), 1)
            out.setdefault(n, {})[kind] = dict(U=U.tolist(), scan_frac=(sc / np.abs(U)).tolist(), tinit=ti.tolist(),
                                               never=np.asarray(r["never"][i]).sum(-1).tolist(),
                                               best_scan_share=float(shares[int(np.argmax(U))]))
        info("sens grid", kind=kind)
    for n in cfgs:
        u9 = {k: out[n][k]["U"][0] for k in ("ea", "da", "oa")}
        out[n]["order_ea_da_oa_at_0.9"] = bool(u9["ea"] <= u9["da"] <= u9["oa"])
        info("sens", cfg=n, best=[out[n][k]["best_scan_share"] for k in ("ea", "da", "oa")],
             scan_frac_oa=round(out[n]["oa"]["scan_frac"][0], 3), tinit_oa=round(out[n]["oa"]["tinit"][0], 1),
             U_oa=round(u9["oa"]), order=out[n]["order_ea_da_oa_at_0.9"])
    write_json(os.path.join(REPO, "runs", "full_v2", "sens_sref_snr0.json"), out)


if __name__ == "__main__":
    main()
