"""Reference-code check: record OUR environment's measurement stream on fixed rollouts so that independent reference code
(filterpy 1.4.5 EKF, scipy DARE + SLSQP) can recompute the tracking cost and the OA allocation from the same inputs.

Writes <out>/env_rollout_<tag>.npz with, per slot: act_pre, act_new, upd, newk, z, var_r, var_th, x_init, c (ours),
w, nmiss, U (ours), and the OA view (r_hat, act, w) + our OA dwells. Scenario: the Case I replica (A-40), one
noise draw from a DEDICATED oracle key (fold_in(root, 0x0AC1E)); never k_test.
The same function is called by tests/test_official_parity.py (shorter T) so the test re-derives our side live.

Usage: python scripts/oracle_dump_env.py --T 2000 --out external/official/fixtures
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402,F401
from src.common import EnvCfg, info, load_calibration, load_yaml  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src import data  # noqa: E402
from src import env as E  # noqa: E402
from src import eval as ev  # noqa: E402

ROOT_SEED = 20261019
ORACLE_TAG = 0x0AC1E


def record(policy: str, s_ref_key: str, T: int, draw: int = 0):
    y = load_yaml()
    cfg = EnvCfg.from_yaml(y)
    cal = load_calibration()
    consts = dict(snr0=jnp.float64(cal["snr0"]), s_ref=jnp.float64(cal[s_ref_key]), delta=jnp.float64(0.9))
    lut = ev.make_lut(cfg)
    pol = ev.make_policy(policy, cfg, y)
    scen = data.replica_case1(cfg, "v1")   # the committed oracle fixture was recorded on the v1 replica
    key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(ROOT_SEED), ORACLE_TAG), draw)

    def step(es, _):
        v = E.pre(cfg, scen, es)
        tau = pol(None, consts, v, es)
        es2, out = E.post(cfg, consts, lut, scen, es, v, tau, jax.random.fold_in(key, es.t))
        keep = {k[4:]: out[k] for k in out if k.startswith("dbg_")}
        keep.update(U=out["U"], nmiss=out["nmiss"], tau=out["tau"], r_hat=v.r_hat, w_pre=v.w, act_view=v.act)
        return es2, keep

    _, rec = jax.jit(lambda: jax.lax.scan(step, E.reset(cfg), None, length=T))()
    rec = {k: np.asarray(v) for k, v in rec.items()}
    meta = dict(T0=cfg.T0, sigma_w_sq=cfg.sigma_w_sq, P0_diag=np.asarray(cfg.P0_diag), beta=cfg.beta,
                snr0=float(cal["snr0"]), r0=cfg.r0, tau0=cfg.tau0, sigma_r0_sq=cfg.sigma_r0_sq,
                sigma_th0_sq=cfg.sigma_th0_sq, r_min_snr=cfg.r_min_snr, delta=0.9, s_ref=float(cal[s_ref_key]))
    return rec, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), "fixtures"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for policy, sk in (("ea", "s_ref"), ("oa", "s_ref_table6")):
        rec, meta = record(policy, sk, a.T)
        p = os.path.join(a.out, f"env_rollout_{policy}_{sk}.npz")
        np.savez(p, **rec, **{f"meta_{k}": v for k, v in meta.items()})
        info("wrote", path=p, n_updates=int(rec["upd"].sum()), n_newk=int(rec["newk"].sum()),
             mean_tracks=float(rec["act_new"].sum(-1).mean()))


if __name__ == "__main__":
    main()
