"""Tiny numpy truth + measurement generator for the Octave oracle smoke test (NOT the project's src/data.py).
Scenario of src/configs/sources.yaml: 4 targets, targets 2-4 alive all steps, target 1 dies at k=41, truth drawn once from
the midpoint (forward and backward), p_d, Poisson clutter uniform on [0,300]^2.  Writes npz with z[R,K,M,2], zvalid[R,K,M]."""
import sys

import numpy as np


def gen(R=3, K=12, p_d=0.9, lambda_c=10.0, q=0.01, seed=0, M=None):
    rng = np.random.default_rng(seed)
    T = 1.0
    F = np.kron(np.eye(2), [[1, T], [0, 1]])
    Q = q * np.kron(np.eye(2), [[T ** 3 / 3, T ** 2 / 2], [T ** 2 / 2, T]])
    L = np.linalg.cholesky(Q)
    mid = (K + 1) // 2  # 1-based midpoint (K odd in the paper)
    X = np.zeros((4, 4, K))  # target, state, time
    x = np.linalg.cholesky(0.1 * np.eye(4)) @ rng.standard_normal((4, 4)) + np.array([150, 0, 150, 0.0])[:, None]
    X[:, :, mid - 1] = x.T
    xf, xb = x.copy(), x.copy()
    for t in range(1, mid):
        xf = F @ xf + L @ rng.standard_normal((4, 4))
        xb = np.linalg.solve(F, xb + L @ rng.standard_normal((4, 4)))
        if mid - 1 - t >= 0: X[:, :, mid - 1 - t] = xb.T
        if mid - 1 + t < K: X[:, :, mid - 1 + t] = xf.T
    t_death = np.array([(K + 1) // 2, K + 1, K + 1, K + 1])
    Mmax = M or 60
    z = np.zeros((R, K, Mmax, 2)); zv = np.zeros((R, K, Mmax), bool)
    for r in range(R):
        for k in range(1, K + 1):
            alive = k < t_death
            det = (rng.random(4) < p_d) & alive
            zt = [X[i, [0, 2], k - 1] + rng.standard_normal(2) for i in range(4) if det[i]]
            nc = rng.poisson(lambda_c)
            zc = [rng.random(2) * 300 for _ in range(nc)]
            zs = zt + zc
            idx = rng.permutation(len(zs))
            zs = [zs[i] for i in idx]
            n = min(len(zs), Mmax)
            for m in range(n):
                z[r, k - 1, m] = zs[m]; zv[r, k - 1, m] = True
    return z, zv, X


if __name__ == "__main__":
    out, R, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    pd = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
    lc = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    z, zv, X = gen(R, K, pd, lc, seed=seed)
    np.savez(out, z=z, zvalid=zv, truth=X)
    print(out, z.shape, 'mean #meas', zv.sum(-1).mean())
