"""Between-truth spread of the Table-II statistic.  The authors' protocol draws the truth ONCE and shares it over all Monte
Carlo runs (PMBMtarget_filter.m), so a published RMS OSPA is conditional on one scenario draw.  Our filter is
verified against the authors' code to ~1e-12 (tests/test_official_parity.py), so it is used here as a fast stand-in to
measure how much the RMS OSPA moves between truth draws.

    python truth_spread.py out.json --draws 0 1 2 --runs 5 [--setting paper|code]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import jax  # noqa: E402

from src import data  # noqa: E402
import diff_run  # noqa: E402
from src import eval as ev  # noqa: E402
from npz_to_mat import oracle_params  # noqa: E402
from reference_run import pad  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--draws", type=int, nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--setting", default="paper")
    a = ap.parse_args()
    q = {"paper": 0.01, "code": 0.02}[a.setting]
    cfg = data.load_scenario_cfg(q=q)
    prm = oracle_params(a.setting, p_d=0.9, lambda_c=10.0)
    P = {k: np.asarray(v, float).reshape(np.asarray(v).shape if np.ndim(v) > 0 else (1, 1)) for k, v in prm.items() if k != "lambda_c"}
    res = {}
    for dr in a.draws:
        root = jax.random.PRNGKey(5000 + dr)
        truth = data.make_truth(jax.random.fold_in(root, 0), cfg)
        z, zv, nd = data.make_measurements_runs(root, truth, 0, 0.9, 10.0, a.runs, cfg, 48, on_drop="raise")
        P["z"], P["zvalid"] = np.asarray(z), np.asarray(zv).astype(float)
        tr = dict(X=np.asarray(truth["X"]), alive=np.asarray(truth["alive"]))
        o2 = {e: [] for e in (1, 2, 3)}
        for r in range(a.runs):
            rec = diff_run.run_ours(P, r, 81, cfg_over=dict(T=192, H=200), M=40, record=False)
            assert rec[-1]["ovf"][:3] == [0, 0, 0], rec[-1]["ovf"]
            for e in (1, 2, 3):
                m, x = pad([rec[k]["est"][e] for k in range(81)], 81)
                o2[e].append(ev.evaluate_run(m, x, tr)["ospa2"])
        res[str(dr)] = {f"est{e}": ev.joint_rms(np.array(o2[e])) for e in (1, 2, 3)}
        print(f"draw {dr}: " + "  ".join(f"est{e} {res[str(dr)][f'est{e}']:.4f}" for e in (1, 2, 3)), flush=True)
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
