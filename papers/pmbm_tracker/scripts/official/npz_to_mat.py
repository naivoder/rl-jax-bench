"""Export measurement sets + EXPLICIT filter parameters to the .mat input of drive_pmbm.m.

Input npz layout (src/data.py):  z[R,K,M,2] float64, zvalid[R,K,M] bool (valid entries first).
CLI:  python npz_to_mat.py meas.npz out.mat --setting paper|code [--dump-runs 0 1] [--p-d 0.9 ...]
Library: write_input_mat(path, z, zvalid, params)  with params = oracle_params('paper'|'code', **overrides).
"""
import argparse
import sys

import numpy as np
from scipy.io import savemat

AREA = (300.0, 300.0)  # paper §VII region [0,300]^2


def oracle_params(setting="paper", **over):
    """Explicit filter parameters.  paper: q=0.01, empty prior at k=0 then predict.  code: q=0.02, lambda0=3, no predict."""
    T = 1.0  # paper §VII
    F = np.kron(np.eye(2), [[1, T], [0, 1]])  # paper §VII
    q = {"paper": 0.01, "code": 0.02}[setting]  # paper §VII / official code ScenarioWilliams15.m
    Q = q * np.kron(np.eye(2), [[T ** 3 / 3, T ** 2 / 2], [T ** 2 / 2, T]])
    H = np.kron(np.eye(2), [[1.0, 0.0]])
    p = dict(
        F=F, Q=Q, H=H, Rcov=np.eye(2),  # paper §VII
        p_s=0.99, p_d=0.9,  # paper §VII
        w_b=np.array([0.005]), m_b=np.array([[100.0, 0, 100.0, 0]]),  # paper §VII birth rate / mean
        P_b=np.diag([150.0 ** 2, 1.0, 150.0 ** 2, 1.0])[None],  # paper §VII birth cov
        # code setting: Poisson prior at k=0 with lambda0=3 on the birth density, first step is an UPDATE (official code)
        lambda0=np.array([3.0]), m0=np.array([[100.0, 0, 100.0, 0]]), P0=np.diag([150.0 ** 2, 1.0, 150.0 ** 2, 1.0])[None],
        predict_first=1.0 if setting == "paper" else 0.0,  # paper: empty prior then predict; code: no predict before k=1
        lambda_c=10.0,  # paper §VII / Figure 4 (ASSUMED value, see sources.yaml)
        N_h=200.0, T_pruning=1e-4, T_pruningPois=1e-5, gating=20.0, existence_threshold=1e-5, est1_thr=0.4,  # official code / paper §VII
        dump_runs=np.array([0.0, 1.0]), dump_pred=1.0, murty_log=1.0,
    )
    p.update(over)
    if "intensity_clutter" not in p:
        p["intensity_clutter"] = p["lambda_c"] / (AREA[0] * AREA[1])  # paper §VII
    return p


def write_input_mat(path, z, zvalid, params):
    d = {k: np.asarray(v, dtype=np.float64) for k, v in params.items() if k != "lambda_c"}
    d["z"] = np.asarray(z, dtype=np.float64)
    d["zvalid"] = np.asarray(zvalid, dtype=np.float64)
    savemat(path, d, format="5", do_compression=False)  # v6/v7-compatible with Octave


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"), ap.add_argument("out")
    ap.add_argument("--setting", default="paper", choices=["paper", "code"])
    ap.add_argument("--p-d", type=float, default=0.9), ap.add_argument("--lambda-c", type=float, default=10.0)
    ap.add_argument("--dump-runs", type=float, nargs="*", default=[0, 1])
    a = ap.parse_args(argv)
    f = np.load(a.npz)
    prm = oracle_params(a.setting, p_d=a.p_d, lambda_c=a.lambda_c, dump_runs=np.array(a.dump_runs))
    write_input_mat(a.out, f["z"], f["zvalid"], prm)
    print(f"wrote {a.out}: z{f['z'].shape} setting={a.setting}")


if __name__ == "__main__":
    main()
