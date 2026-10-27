"""Table II sweep: export the 15 Table II measurement sets (Fig.-3 truth, measurement root PRNGKey(3003), config_id from (p_d, lambda_c), runs 0..99)
for the Octave Williams oracle, with sha256 of every set.  Same call as src/train.py --truth fig3 --meas-seed 3003, so run r is byte-identical to
runs/table2_*/seed_r.  Refuses (raises) when any clutter measurement is dropped at M = 48.
  python scripts/export_table2_meas.py OUTDIR [--runs 100] [--meas-seed 3003]
Writes OUTDIR/pd<p>_l<l>/{meas.npz,in_williams.mat} and OUTDIR/table2_meas_sha256.json (copied to runs/ by scripts/table2_all.sh)."""
import argparse, hashlib, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "official")]
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import numpy as np, jax
from src import data as D
import npz_to_mat as NM

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--runs", type=int, default=100); ap.add_argument("--meas-seed", type=int, default=3003)
a = ap.parse_args()
cfg = D.load_scenario_cfg(seed_top=a.meas_seed)
truth = D.load_fig3_truth()
reg = {"meas_seed": a.meas_seed, "runs": a.runs, "truth_file": "data/fig3_truth.npz",
       "truth_file_sha256": D.sha256_file(os.path.join(HERE, "..", "data", "fig3_truth.npz")), "cells": {}}
for pd in D.DATA_TABLE2_PD:
    for lam in D.DATA_TABLE2_LAMBDA:
        cid = D.DATA_TABLE2_PD.index(pd) * 3 + D.DATA_TABLE2_LAMBDA.index(lam)
        z, zv, nd = D.make_measurements_runs(jax.random.PRNGKey(cfg["seed_top"]), truth, cid, pd, float(lam), a.runs, cfg, 48, on_drop="raise")
        z, zv = np.asarray(z), np.asarray(zv)
        d = os.path.join(a.out, f"pd{pd}_l{lam}"); os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "meas.npz"), X=np.asarray(truth["X"]), alive=np.asarray(truth["alive"]), z=z, zvalid=zv, meta_M_max=np.array([48]),
                 meta_config_id=np.array([cid]), meta_lambda_c=np.array([float(lam)]), meta_pd=np.array([pd]), meta_seed_top=np.array([a.meas_seed]))
        NM.write_input_mat(os.path.join(d, "in_williams.mat"), z, zv,
                           NM.oracle_params("paper", p_d=pd, lambda_c=float(lam), dump_runs=np.array([-1.0])))
        sha = D.sha256_arrays(dict(z=z, zvalid=zv))
        reg["cells"][f"{pd}_{lam}"] = dict(config_id=cid, sha256_z_zvalid=sha, max_meas_per_step=int(zv.sum(-1).max()), n_dropped=int(np.asarray(nd).sum()))
        print(pd, lam, "cid", cid, "max meas", int(zv.sum(-1).max()), "sha", sha[:16], flush=True)
reg["truth_sha256_arrays"] = D.sha256_arrays(dict(X=np.asarray(truth["X"]), alive=np.asarray(truth["alive"])))
json.dump(reg, open(os.path.join(a.out, "table2_meas_sha256.json"), "w"), indent=1)
