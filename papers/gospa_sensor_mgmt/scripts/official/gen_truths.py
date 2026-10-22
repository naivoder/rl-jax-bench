"""Run the authors' Groundtruth generator (official venv) for a list of seeds and save the truths for the S-TRUTH sensitivity row (ten alternative truths) (data only; no code copied).
usage (official venv python): gen_truths.py <official_repo> <out.npz> <seed0> <seed1>"""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import Parameters
from groundtruth import Groundtruth
out, s0, s1 = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
res = {}
for seed in range(s0, s1):
    np.random.seed(seed)
    g = Groundtruth(); g.generate_multitarget_groundtruth()
    T = np.array(g.target_states)
    alive = ~np.isnan(T[:, 0, :])
    res[f"seed{seed}"] = T
    print(seed, "n_targets", T.shape[0], "max_alive", int(alive.sum(0).max()), flush=True)
np.savez(out, **res)
