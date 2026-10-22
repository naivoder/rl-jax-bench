"""CFG-P path of the filter (eq. 48 detection at the mean, converged LBP) against an independent exhaustive-enumeration Bernoulli update
(paper eqs. 9-17, 21) on small cases; the association graph is kept a forest so that loopy BP is exact."""
import itertools
import numpy as np
import jax.numpy as jnp
from src import filter as FL
from src import model as M

CFG = M.CFG_P.replace(lbp_tol=1e-13, lbp_max_iter=5000)
H = M.H_FULL


def exact_update(r, x, P, z, spos, lam_dens, Rm):
    n, m = len(r), len(z)
    pD = M.PD_MAX * np.exp(-0.5 * (np.linalg.norm(x[:, [0, 2]] - spos, axis=1) / M.FOV) ** 2)
    S = [H @ P[i] @ H.T + Rm for i in range(n)]
    K = [P[i] @ H.T @ np.linalg.inv(S[i]) for i in range(n)]
    # per-Bernoulli single-hypothesis quantities
    w0 = 1 - r * pD
    r0 = r * (1 - pD) / w0
    wj = np.zeros((n, m)); xj = {}; Pj = {}
    for i in range(n):
        for j in range(m):
            v = z[j] - H @ x[i]
            wj[i, j] = r[i] * pD[i] * np.exp(-0.5 * v @ np.linalg.solve(S[i], v)) / np.sqrt(np.linalg.det(2 * np.pi * S[i])) / lam_dens
            xj[i, j] = x[i] + K[i] @ v
            Pj[i, j] = P[i] - K[i] @ H @ P[i]
    tot = 0.0
    marg = np.zeros((n, m + 1))
    for a in itertools.product(range(m + 1), repeat=n):
        used = [q for q in a if q > 0]
        if len(used) != len(set(used)):
            continue
        wt = np.prod([w0[i] if a[i] == 0 else wj[i, a[i] - 1] for i in range(n)])
        tot += wt
        for i in range(n):
            marg[i, a[i]] += wt
    marg /= tot
    out = []
    for i in range(n):
        ws = np.concatenate([[marg[i, 0] * r0[i]], marg[i, 1:]])
        rn = ws.sum()
        ws = ws / rn
        xs = [x[i]] + [xj[i, j] for j in range(m)]
        Ps = [P[i]] + [Pj[i, j] for j in range(m)]
        xm = sum(w * xx for w, xx in zip(ws, xs))
        Pm = sum(w * (PP + np.outer(xx - xm, xx - xm)) for w, xx, PP in zip(ws, xs, Ps))
        out.append((rn, xm, Pm))
    return out


def test_update_equals_exhaustive_enumeration_smooth_pd():
    rng = np.random.default_rng(5)
    NB = M.NB_MAX
    worst = 0.0
    for _ in range(40):
        n = int(rng.integers(1, 4))
        # well separated Bernoullis (100 m apart) -> each measurement only matters to its own Bernoulli (forest)
        pos = np.stack([np.array([100.0 * i, 0.0]) + rng.normal(size=2) for i in range(n)])
        x = np.zeros((n, 4)); x[:, 0], x[:, 2] = pos[:, 0], pos[:, 1]; x[:, 1], x[:, 3] = rng.normal(size=n), rng.normal(size=n)
        P = np.zeros((n, 4, 4))
        for i in range(n):
            A = rng.normal(size=(4, 4)); P[i] = A @ A.T + np.diag(rng.uniform(1.0, 4.0, 4))
        r = rng.uniform(0.1, 0.9, n)
        spos = pos[int(rng.integers(0, n))] + rng.normal(size=2) * 10.0
        zl = [x[i, [0, 2]] + rng.normal(size=2) * 2 for i in range(n) if rng.random() < 0.7]
        z = np.array(zl).reshape(-1, 2)
        lam = 0.5 / (np.pi * 1600)
        ref = exact_update(r, x, P, z, spos, lam, CFG.R_var * np.eye(2))
        fs = FL.FState(jnp.zeros(NB).at[:n].set(r), jnp.zeros((NB, 4)).at[:n].set(x), jnp.tile(jnp.eye(4), (NB, 1, 1)).at[:n].set(P), jnp.asarray(n, jnp.int32))
        zz = np.zeros((M.M_MEAS, 2)); zv = np.zeros(M.M_MEAS, bool); zz[: len(z)] = z; zv[: len(z)] = True
        out, info = FL.update_sensor(fs, jnp.asarray(zz), jnp.asarray(zv), jnp.asarray(spos), lam, CFG)
        for i, (rn, xm, Pm) in enumerate(ref):
            if rn <= M.R_PRUNE:
                continue
            j = i                                    # no pruning among the first n (all r' > 1e-4 here)
            worst = max(worst, abs(float(out.r[j]) - rn), np.max(np.abs(np.asarray(out.x[j]) - xm)) / (1 + np.abs(xm).max()),
                        np.max(np.abs(np.asarray(out.P[j]) - Pm)) / (1 + np.abs(Pm).max()))
    assert worst < 1e-9, worst
