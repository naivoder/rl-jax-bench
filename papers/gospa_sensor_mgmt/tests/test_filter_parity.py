"""Filter differential tests: PMB update + LBP + TOMB + merge + prune of src/filter.py vs the authors' code
(update.py, lbp.py, tomb.py, utils.merge_gaussians) on random (Bernoulli list, measurement sets) pairs, CFG-O behavior."""
import numpy as np
import jax, jax.numpy as jnp
import pytest
from src import filter as FL
from src import model as M
import official

CFG = M.CFG_O


class _Sens:
    def __init__(self, p):
        self.current_sensor_position = np.array(p, float).reshape(2, 1)


def _rand_case(rng):
    n = int(rng.integers(1, 9))
    sp = [rng.uniform(-40, 40, 2), rng.uniform(-40, 40, 2)]
    pos = rng.uniform(-70, 70, (n, 2))
    # a few Bernoullis close together to trigger merging
    for i in range(1, n):
        if rng.random() < 0.3:
            pos[i] = pos[i - 1] + rng.normal(size=2) * 2.0
    x = np.zeros((4, n)); x[0], x[2] = pos[:, 0], pos[:, 1]
    x[1], x[3] = rng.normal(size=n) * 0.5, rng.normal(size=n) * 0.5
    P = np.zeros((n, 4, 4))
    for i in range(n):
        A = rng.normal(size=(4, 4)) * rng.choice([0.5, 2.0, 5.0])
        P[i] = A @ A.T + np.diag(rng.uniform(0.5, 4.0, 4))
    r = rng.uniform(0.02, 0.99, n)
    zs = []
    for s in range(2):
        m = int(rng.integers(0, 6))
        z = np.zeros((2, m))
        for j in range(m):
            if rng.random() < 0.6:
                i = rng.integers(0, n)
                z[:, j] = x[[0, 2], i] + rng.normal(size=2) * 2.0
            else:
                z[:, j] = sp[s] + rng.uniform(-40, 40, 2)
        zs.append(z)
    return r, x, P, sp, zs


def _official_step(off, r, x, P, sp, zs, lam, cap):
    Pm = off["Parameters"]
    Pm.lambda_false_alarm = lam / (np.pi * 40.0 ** 2)
    upd = off["update"].Updater(off["predict"].Predictor(), None)
    lambdau = np.zeros(Pm.lambdau.shape); mn = np.zeros(Pm.birth_means.shape); cn = np.zeros(Pm.birth_covariances.shape)
    lbp, tomb = off["lbp"].lbp, off["tomb"].tomb
    r1, x1, P1 = r.copy(), x.copy(), P.copy()
    lu, mnt, cnt, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = upd.update_pmb_empty_poisson(lambdau, mn, cn, r1, x1, P1, zs[0], [_Sens(sp[0])])
    pupd, pnew = lbp(wupd, wnew)
    r1, x1, P1 = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
    r1 = r1.reshape(-1)   # official reshapes to (-1,1), which numpy>=2 rejects in update.py; the values are identical
    lu, mn2, cn2, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew = upd.update_pmb_empty_poisson(lu, mnt, cnt, r1, x1, P1, zs[1], [_Sens(sp[1])])
    pupd, pnew = lbp(wupd, wnew)
    r1, x1, P1 = tomb(pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew)
    if len(r1) > 0:
        r1, x1, P1 = off["utils"].merge_gaussians(r1, x1, P1)
    if cap:
        r1[r1 >= 1] = 1
    keep = r1 > Pm.pruning_threshold
    r1, x1, P1 = r1[keep], x1[:, keep], P1[keep]
    xc = (x1[0] >= -250) & (x1[0] <= 250) & (x1[2] >= -250) & (x1[2] <= 250)
    return r1[xc], x1[:, xc], P1[xc]


def _mine(cfg, r, x, P, sp, zs, lam, cap):
    NB = M.NB_MAX
    n = len(r)
    fs = FL.FState(jnp.zeros(NB).at[:n].set(r), jnp.zeros((NB, 4)).at[:n].set(x.T),
                   jnp.tile(jnp.eye(4), (NB, 1, 1)).at[:n].set(P), jnp.asarray(n, jnp.int32))
    ld = lam / (np.pi * 40.0 ** 2)
    for s in range(2):
        z = np.zeros((M.M_MEAS, 2)); zv = np.zeros(M.M_MEAS, bool)
        m = zs[s].shape[1]
        z[:m] = zs[s].T; zv[:m] = True
        fs, _ = FL.update_sensor(fs, jnp.asarray(z), jnp.asarray(zv), jnp.asarray(sp[s]), ld, cfg)
    fs = FL.merge(fs, cap)
    n2 = int(fs.n)
    return np.asarray(fs.r)[:n2], np.asarray(fs.x)[:n2].T, np.asarray(fs.P)[:n2]


@pytest.mark.parametrize("cap", [False, True])
def test_update_merge_matches_official(cap):
    off = official.load()
    if off is None:
        pytest.skip(official.SKIP)
    rng = np.random.default_rng(7)
    upd = jax.jit(lambda r, x, P, n, sp, z, zv, ld: None)  # placeholder to keep imports warm
    worst = 0.0
    ncase = 0
    for it in range(120):
        r, x, P, sp, zs = _rand_case(rng)
        lam = float(rng.choice([0.1, 1.0, 2.0]))
        ro, xo, Po = _official_step(off, r, x, P, sp, zs, lam, cap)
        rm, xm, Pm_ = _mine(CFG, r, x, P, sp, zs, lam, cap)
        assert len(ro) == len(rm), (it, len(ro), len(rm))
        if len(ro):
            worst = max(worst, np.max(np.abs(ro - rm)), np.max(np.abs(xo - xm) / (1 + np.abs(xo))), np.max(np.abs(Po - Pm_) / (1 + np.abs(Po))))
        ncase += 1
    print(f"[METRIC] filter parity cap={cap}: {ncase} cases, worst rel err vs official = {worst:.2e}")
    assert worst < 1e-8, worst
