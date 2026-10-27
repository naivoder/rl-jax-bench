"""Write the parity cases (measurement sets from the PROJECT generator src/data.py + explicit parameters) to .mat files
for scripts/official/drive_pmbm.m.  Case table below; every case is run through the authors' code (Octave) and through ours.

    python make_parity_inputs.py OUTDIR [case ...]
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import jax  # noqa: E402

from src import data  # noqa: E402
from npz_to_mat import oracle_params, write_input_mat  # noqa: E402

# name: (setting, p_d, lambda_c, K, R, overrides)
CASES = {
    "A_paper":   ("paper", 0.9, 10.0, 81, 2, {}),                                   # Figure-4 operating point, paper path
    "B_code":    ("code", 0.9, 10.0, 81, 2, {}),                                    # official-code path: q=0.02, lambda0=3, no predict at k=1
    "C_stress":  ("paper", 0.6, 20.0, 81, 2, {}),                                   # worst Table II corner: many hypotheses
    "D_smallNh": ("paper", 0.7, 15.0, 81, 2, dict(N_h=25.0, T_pruning=1e-3)),       # caps/prune bind hard
    "E_gate":    ("paper", 0.9, 15.0, 81, 2, dict(gating=6.0, est1_thr=0.7)),       # tight gate, non-default estimator-1 threshold
    "R_code":    ("code", 0.9, 10.0, 81, 10, {}),                                   # reference run, authors' DEFAULTS (q=0.02, lambda0=3, no predict at k=1)
    "R_paper":   ("paper", 0.9, 10.0, 81, 20, {}),                                  # reference run, paper settings (20 runs, project generator)
    "Z_noprune": ("paper", 0.9, 10.0, 81, 2, dict(T_pruning=0.0)),                  # NO global-weight floor (Alg. 1: top N_h only; the 2019 release's default)
}


def build(name, outdir, top_seed=18, M_max=48):
    setting, pd, lc, K, R, over = CASES[name]
    q = {"paper": 0.01, "code": 0.02}[setting]
    cfg = data.load_scenario_cfg(q=q)
    root = jax.random.PRNGKey(top_seed + 1000 + sorted(CASES).index(name))
    truth = data.make_truth(jax.random.fold_in(root, 0), cfg)
    z, zv, nd = data.make_measurements_runs(root, truth, 0, pd, lc, R, cfg, M_max, on_drop="raise")
    prm = oracle_params(setting, p_d=pd, lambda_c=lc, dump_runs=np.array([0.0]), **over)
    path = os.path.join(outdir, f"{name}.mat")
    write_input_mat(path, np.asarray(z), np.asarray(zv), prm)
    np.savez(os.path.join(outdir, f"{name}_truth.npz"), X=np.asarray(truth["X"]), alive=np.asarray(truth["alive"]))
    print(name, path, "z", z.shape, "mean #meas", float(np.asarray(zv).sum(-1).mean()), "max #meas", int(np.asarray(zv).sum(-1).max()))


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for n in (sys.argv[2:] or list(CASES)):
        build(n, out)
