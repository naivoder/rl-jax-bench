"""Input for gospa_ref.m: random point-set pairs (incl. empty sets, pairs beyond the cut-off) + the pruned estimates of a
dumped run vs its truth.  usage: make_gospa_input.py case_truth.npz dump_dir run out.mat"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.io import savemat
import parity_lib as pl

truth_npz, dump_dir, run, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(7)
Np, Nmax = 400, 7
xs = np.full((Np, 2, Nmax), np.nan); ys = xs.copy(); nx = np.zeros(Np); ny = np.zeros(Np)
for i in range(Np):
    a, b = rng.integers(0, Nmax + 1, size=2)
    scale = rng.choice([3.0, 12.0, 40.0])
    pts = rng.random((2, Nmax)) * scale
    xs[i, :, :a] = pts[:, :a]
    ys[i, :, :b] = rng.random((2, Nmax))[:, :b] * scale
    nx[i], ny[i] = a, b
T = np.load(truth_npz); X, alive = T["X"], T["alive"]          # [4,K,4], [4,K]
K = X.shape[1]
d = pl.load_run(dump_dir, run)
est = np.asarray(d["est1_x"]); n_est = np.asarray(d["est1_n"]).reshape(-1)
est = np.nan_to_num(est, nan=0.0)
Xo = np.concatenate([X[i].T for i in range(4)], axis=0)       # [16,K] official layout (zero outside lifetime)
t_birth = np.array([[1, 1, 1, 1]], float); t_death = np.array([[41, 82, 82, 82]], float)
savemat(out, dict(xs=xs, ys=ys, nx=nx, ny=ny, est=est, n_est=n_est, X_truth=Xo, t_birth=t_birth, t_death=t_death,
                  c=10.0, p=2.0, alpha=2.0), format="5", do_compression=False)
print("wrote", out)
