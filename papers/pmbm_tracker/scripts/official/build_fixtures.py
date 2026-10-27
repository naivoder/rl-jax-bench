"""Record a parity fixture (small .npz, committed under tests/fixtures/official/) from the AUTHORS' dump of one run.

    python build_fixtures.py input.mat dump_dir run K out.npz [--murty-n 60]

Contents: model/filter parameters (param_*), the measurements of that run (z, zvalid), and per step k: digest of the
pruned posterior (see parity_lib.digest_state), pre-pruning estimates 1-3, number of pre-pruning children.  With
--murty-n: a sample of the authors' Murty problems (input cost matrices + their k-best outputs) from the same dump.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402

import parity_lib as pl  # noqa: E402


def build(in_mat, dump_dir, run, K, out, murty_n=0, seed=0):
    P = loadmat(in_mat)
    d = pl.load_run(dump_dir, run)
    assert int(d["full_dump"].reshape(-1)[0]) == 1, "fixture needs a full-dump run"
    K = min(K, int(d["K"].reshape(-1)[0]))
    arrs = {}
    for k, v in P.items():
        if k.startswith("__") or k in ("z", "zvalid"):
            continue
        arrs["param_" + k] = np.asarray(v, float)
    arrs["z"] = np.asarray(P["z"])[run, :K]
    arrs["zvalid"] = np.asarray(P["zvalid"])[run, :K] != 0
    arrs["run"] = np.array(run)
    dig = []
    est = []
    nch = []
    for k in range(K):
        pre = f"k{k + 1:03d}"
        dig.append(pl.digest_state(pl.official_state(d, "p" + pre)))
        est.append(pl.digest_est({e: pl.official_estimates(d, e, k) for e in (1, 2, 3)}))
        nch.append(len(pl.official_state(d, "u" + pre)[1]))
    for key in dig[0]:
        arrs["dig_" + key] = np.stack([x[key] for x in dig])
    arrs["est"] = np.stack(est)
    arrs["n_children"] = np.array(nch)
    arrs["n_glob_pruned"] = np.asarray(d["n_glob_pruned"]).reshape(-1)[:K]
    if murty_n:
        arrs.update(sample_murty(d, K, murty_n, seed))
    np.savez_compressed(out, **arrs)
    print(f"{out}: {os.path.getsize(out) / 1e6:.2f} MB, K={K}, hyps (pruned) max {int(np.nanmax(arrs['dig_hyp_nb'].shape))}")


def rebuild_cost_matrix(d, key, nprev, M):
    """Full log-weight matrix [nprev + M, M] (old tracks, then one private 'new' row per measurement; -Inf = not allowed)
    from the dumped pieces: the trimmed matrix, its row/column maps, and the FIXED columns (a single finite entry, always the
    private 'new' row of that measurement).  The individual fixed entries are not dumped, only their sum cost_fixed: it is put
    on the first fixed column (every assignment contains all fixed entries, so only the sum matters)."""
    g = lambda n: np.asarray(d[key + n], float)
    Ct = g("cost_trimmed")
    sr = g("indices_stay_row").reshape(-1).astype(int) - 1
    sc = g("indices_stay_column").reshape(-1).astype(int) - 1
    fr = g("fixed_row").reshape(-1).astype(int) - 1
    fc = g("fixed_column").reshape(-1).astype(int) - 1
    C = np.full((nprev + M, M), -np.inf)
    if Ct.size:
        C[np.ix_(sr, sc)] = Ct.reshape(len(sr), len(sc))
    for j, (r, c) in enumerate(zip(fr, fc)):
        assert r == nprev + c, "a fixed column must be the private new row of its measurement"
        C[r, c] = float(g("cost_fixed").reshape(-1)[0]) if j == 0 else 0.0
    return C


def sample_murty(d, K, n, seed):
    """Sample of the authors' Murty problems with kbest > 1 and a non-empty trimmed matrix (cost matrices contain -Inf)."""
    cands = []
    for k in range(K):
        pre = f"uk{k + 1:03d}"
        nm = int(np.asarray(d.get(pre + "_mur_n", [[0]])).reshape(-1)[0])
        for i in range(1, nm + 1):
            key = f"{pre}_mur{i:02d}_"
            if d[key + "cost_trimmed"].size == 0:
                continue
            cands.append((k, i))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
    out = {}
    L = []
    for q, ci in enumerate(sorted(pick)):
        k, i = cands[ci]
        key = f"uk{k + 1:03d}_mur{i:02d}_"
        names = ("kbest", "cost_fixed", "cost_misdetection_sum", "globWeightLog_pred", "assign_meas_to_track",
                 "glob_weight_log_children")
        rec = {name: np.asarray(d[key + name], float) for name in names}
        nprev = int(np.asarray(d[f"rk{k + 1:03d}_trk_tini"]).size)     # old tracks in the PREDICTED state
        M = np.asarray(d[f"zk{k + 1:03d}"]).reshape(2, -1).shape[1]
        rec["cost_matrix_log"] = rebuild_cost_matrix(d, key, nprev, M)
        rec["nprev"] = np.array(nprev)
        for name, v in rec.items():
            out[f"mur{q:03d}_{name}"] = v
        L.append((k, i))
    out["mur_n"] = np.array(len(L))
    out["mur_index"] = np.array(L)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_mat"), ap.add_argument("dump_dir"), ap.add_argument("run", type=int), ap.add_argument("K", type=int)
    ap.add_argument("out")
    ap.add_argument("--murty-n", type=int, default=0)
    a = ap.parse_args()
    build(a.in_mat, a.dump_dir, a.run, a.K, a.out, a.murty_n)
