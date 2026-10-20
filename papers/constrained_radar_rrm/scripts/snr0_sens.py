"""A-02 declared sensitivity (SNR0 x 0.5 and x 2), flown in the full run as a
verification of the baseline LEVEL offset (Fig. 9a/b replica: ours / paper = 1.23-1.40 at every scan share under
A-04a). EA/DA/OA on the Case I replica, k_test draws 0..19 (pairing 0), both S_ref anchors, SNR0 x {0.5, 1, 2}.
Usage: python scripts/snr0_sens.py   -> runs/full/eval/snr0_sens.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402,F401
from src.common import EnvCfg, info, load_calibration, load_yaml, write_json  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import data, eval as ev  # noqa: E402
data.REPLICA, ev.SLOT_KEY = "v1", "legacy"   # Re-evaluation: reproduce the recorded (pre-re-digitisation, legacy-key) numbers
from full_eval import CALIBS, NR, T, bcast, keys  # noqa: E402


def main():
    common.check_device()
    y = load_yaml()
    cal = load_calibration()
    cfg = EnvCfg.from_yaml(y)
    lut = ev.make_lut(cfg)
    rk, _ = keys(jax.random.PRNGKey(20261019))
    rscen = bcast(data.replica_case1(cfg), NR)
    res = {}
    for kind in ("ea", "da", "oa"):
        one = ev.make_evaluator(cfg, lut, ev.make_policy(kind, cfg, y), T)
        for ck in ("A04", "A04a"):
            for f in (0.5, 1.0, 2.0):
                c = dict(snr0=jnp.float64(cal["snr0"] * f), s_ref=jnp.float64(cal[CALIBS[ck]]), delta=jnp.float64(0.9))
                r = jax.device_get(one(None, c, rscen, rk[:NR]))
                res[f"{kind}|{ck}|x{f}"] = dict(U=float(np.mean(r["U"])), c=float(np.mean(r["c"])),
                                                scan=float(np.mean(r["scan_term"])))
                info("snr0 sens", kind=kind, calib=ck, factor=f, **res[f"{kind}|{ck}|x{f}"])
    write_json("runs/full/eval/snr0_sens.json", res)


if __name__ == "__main__":
    main()
