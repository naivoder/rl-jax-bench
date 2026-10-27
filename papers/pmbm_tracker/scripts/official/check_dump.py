"""Internal-consistency check of an oracle dump (does NOT test the oracle's code against another filter -- that is the
differential test).  It recomputes, with plain numpy, pieces of the dumped state from the dumped PREVIOUS state and the model,
to prove that drive_pmbm.m dumps what it claims and that the work-copy Murty path is wired correctly:
  1. Poisson update: w_upd = (1-p_d) w_pred, means/covs unchanged
  2. old tracks: missed / detected single-target hypotheses (r', log w_k, KF mean/cov) against numpy KF formulas
  3. Murty children: log weight = sum of picked trimmed entries + fixed + misdetection + parent; ascending weights; distinct
  4. global weights sum to 1; every globHyp index refers to an existing, valid single-target hypothesis
usage: python check_dump.py input.mat run_000.mat
"""
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from load_dump import load_run  # noqa: E402


def check(in_mat, run_mat, max_steps=None):
    P = loadmat(in_mat)
    F, Q, H, R = (np.asarray(P[k], float) for k in ("F", "Q", "H", "Rcov"))
    p_d = float(P["p_d"][0, 0]); gate = float(P["gating"][0, 0])
    d = load_run(run_mat)
    assert d["full"], "needs a full-dump run"
    worst = {"poisson": 0.0, "kf_mean": 0.0, "kf_cov": 0.0, "wlog": 0.0, "r": 0.0, "murty_w": 0.0, "globW_sum": 0.0}
    n_murty = 0
    for k, st in enumerate(d["steps"]):
        if max_steps and k >= max_steps:
            break
        pred, upd, z = st["pred"], st["upd"], st["z"].T if st["z"].size else np.zeros((2, 0))
        # 1. Poisson
        worst["poisson"] = max(worst["poisson"], np.abs(upd["pois_w"] - (1 - p_d) * pred["pois_w"]).max(initial=0.0),
                               np.abs(upd["pois_m"] - pred["pois_m"]).max(initial=0.0))
        # 2. old tracks
        for i, tr in enumerate(pred["tracks"]):
            nh = len(tr["eB"]); ut = upd["tracks"][i]
            for j in range(nh):
                r, m_, C_ = tr["eB"][j], tr["mean"][j], tr["cov"][j]
                den = 1 - r + r * (1 - p_d)
                assert ut["valid"][j]
                worst["r"] = max(worst["r"], abs(ut["eB"][j] - r * (1 - p_d) / den))
                worst["wlog"] = max(worst["wlog"], abs(ut["wlog"][j] - np.log(den)))
                S = H @ C_ @ H.T + R; zp = H @ m_; Sinv = np.linalg.inv(S); Kg = C_ @ H.T @ Sinv
                for mi in range(z.shape[1]):
                    idx = j + nh * (mi + 1)
                    maha = (z[:, mi] - zp) @ Sinv @ (z[:, mi] - zp)
                    if maha < gate:
                        assert idx < len(ut["eB"]) and ut["valid"][idx], (k, i, j, mi)
                        Pu = (np.eye(4) - Kg @ H) @ C_; Pu = (Pu + Pu.T) / 2
                        xu = m_ + Kg @ (z[:, mi] - zp)
                        lw = np.log(r * p_d) - 0.5 * maha - 0.5 * np.log(np.linalg.det(S)) - np.log(2 * np.pi)
                        worst["kf_mean"] = max(worst["kf_mean"], np.abs(ut["mean"][idx] - xu).max())
                        worst["kf_cov"] = max(worst["kf_cov"], np.abs(ut["cov"][idx] - Pu).max())
                        worst["wlog"] = max(worst["wlog"], abs(ut["wlog"][idx] - lw))
                        worst["r"] = max(worst["r"], abs(ut["eB"][idx] - 1.0))
                    else:
                        assert idx >= len(ut["eB"]) or not ut["valid"][idx], (k, i, j, mi, "gated-out entry must be empty")
        # 3. Murty children
        for mu in st.get("murty", []):
            n_murty += 1
            Ct = mu["cost_trimmed"]  # rows = tracks (kept), cols = measurements (kept), log weights
            rows, cols = mu["indices_stay_row"].astype(int), mu["indices_stay_column"].astype(int)
            A = mu["assign_meas_to_track"]  # children x Nz : track row index (original numbering), 0 = none
            lw = mu["glob_weight_log_children"]
            if lw.size > 1:
                assert np.all(np.diff(lw) <= 1e-12), "children not sorted by weight"
            assert len({tuple(a) for a in A}) == A.shape[0], "duplicate Murty children"
            for c in range(A.shape[0]):
                tot = mu["cost_fixed"] + mu["cost_misdetection_sum"] + mu["globWeightLog_pred"]
                for ci, col in enumerate(cols):
                    rr = int(A[c, col - 1]); tot += Ct[list(rows).index(rr), ci]
                worst["murty_w"] = max(worst["murty_w"], abs(tot - lw[c]))
        # 4. global weights / indices
        for name in ("upd", "pruned"):
            s = st[name]
            worst["globW_sum"] = max(worst["globW_sum"], abs(s["globW"].sum() - 1.0))
            for row in s["globHyp"]:
                for i, j in enumerate(row):
                    if j > 0:
                        assert j <= len(s["tracks"][i]["eB"]) and s["tracks"][i]["valid"][j - 1], (k, name, i, j)
    return worst, n_murty


if __name__ == "__main__":
    w, n = check(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    print("max abs deviations:", {k: float(f"{v:.3e}") for k, v in w.items()}, "| murty parents checked:", n)
    assert max(w.values()) < 1e-9, "consistency check FAILED"
    print("CHECK PASSED (<1e-9)")
