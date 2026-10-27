"""Smoke run of the GM-PHD / GM-CPHD baselines: N runs x 81 steps at (p_d, lambda_c) = (0.9, 10) (Table II row 4).

Paper Table II expects roughly PHD 7.05 / CPHD 3.39 (100 runs). Nothing here is tuned to match; run with --lambda0 3 --q 0.02 for the
official-code settings (sensitivity rows), without for the paper settings.

usage: python scripts/run_phd_smoke.py [--runs 10] [--pd 0.9] [--lambda-c 10] [--lambda0 3] [--q 0.02] [--config-id 3] [--seed 18]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # GPU only when explicitly told (preamble)
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from src import data  # noqa: E402
from src import eval as ev  # noqa: E402
from src import phd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--pd", type=float, default=0.9)
    ap.add_argument("--lambda-c", type=float, default=10.0)
    ap.add_argument("--lambda0", type=float, default=None)
    ap.add_argument("--q", type=float, default=None)
    ap.add_argument("--config-id", type=int, default=3)
    ap.add_argument("--seed", type=int, default=data.DATA_TOP_SEED)
    ap.add_argument("--M-max", type=int, default=48)
    ap.add_argument("--phd-estimator", choices=["est2", "est1"], default="est2", help="est2 = GMPHD_estimation2 (official default); est1 = original Vo-Ma GMPHD_estimation")
    a = ap.parse_args()
    ov = {k: v for k, v in dict(lambda0=a.lambda0, q=a.q).items() if v is not None}
    cfg = data.load_scenario_cfg(**ov)
    cfg["phd_estimator"] = a.phd_estimator
    print(f"[INFO] runs={a.runs} pd={a.pd} lambda_c={a.lambda_c} overrides={ov} phd_estimator={a.phd_estimator} seed={a.seed} config_id={a.config_id}")
    root = jax.random.PRNGKey(a.seed)
    truth = data.make_truth(jax.random.fold_in(root, 0), cfg)
    z, zv, nd = data.make_measurements_runs(root, truth, a.config_id, a.pd, a.lambda_c, a.runs, cfg, a.M_max)
    print(f"[INFO] dropped clutter: {int(np.asarray(nd).sum())}")
    tr_np = dict(X=np.asarray(truth["X"]), alive=np.asarray(truth["alive"]))
    for kind in ("phd", "cphd"):
        f = phd.make_phd_batch(cfg, kind, E=16)
        t0 = time.time()
        em, ex, info = f(z, zv, a.pd, a.lambda_c)
        jax.block_until_ready(ex)
        t1 = time.time()
        em, ex = np.asarray(em), np.asarray(ex)
        res = ev.evaluate_runs(em, ex, tr_np)
        cell = ev.table_cell(res["ospa2"], n_boot=2000)
        g = ev.joint_rms(res["gospa2"])
        parts = {k: float(np.sqrt(res[k].mean())) for k in ("g_loc", "g_mis", "g_fal")}
        ncap = int((np.asarray(info["n_hat"]) > 16).sum())
        print(f"[METRIC] {kind.upper():5s} RMS OSPA {cell['value']:.3f} (SE delta {cell['se_delta']:.3f}, boot {cell['se_boot']:.3f}) | "
              f"RMS GOSPA(extra) {g:.3f} | loc/mis/fal sqrt-mean {parts['g_loc']:.2f}/{parts['g_mis']:.2f}/{parts['g_fal']:.2f} | "
              f"n_hat>E steps {ncap} | wall {t1 - t0:.1f}s")
        rms = ev.rms_per_step(res["ospa2"])
        print(f"[METRIC] {kind.upper():5s} per-step RMS OSPA k=1,2,3,10,20,40,41,42,50,60,81: "
              + " ".join(f"{rms[k - 1]:.2f}" for k in (1, 2, 3, 10, 20, 40, 41, 42, 50, 60, 81)))
        print(f"[METRIC] {kind.upper():5s} mean |est| / |truth|: {res['n_est'].mean():.2f} / {res['n_true'].mean():.2f}")


if __name__ == "__main__":
    main()
