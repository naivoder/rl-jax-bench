import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import data  # noqa: E402
from src import eval as ev  # noqa: E402
from src import phd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = np.load(os.path.join(HERE, "oracle_phd_ref.npz"))  # Octave run of the OFFICIAL code (see its 'note'), z/zvalid + estimates
_RUNNERS = {}


def _cfg(mode):
    return data.load_scenario_cfg(**({"lambda0": 3.0, "q": 0.02} if mode == "code" else {}))


def _runner(mode, kind):
    """One compiled batch filter per (mode, kind), shared by all tests (shape [3, 81, 48])."""
    if (mode, kind) not in _RUNNERS:
        _RUNNERS[(mode, kind)] = phd.make_phd_batch(_cfg(mode), kind, 16)
    return _RUNNERS[(mode, kind)]


def _run(mode, kind, z=None, zv=None):
    z = FIX[f"{mode}_{kind}_z"] if z is None else z
    zv = FIX[f"{mode}_{kind}_zv"] if zv is None else zv
    em, ex, info = _runner(mode, kind)(jnp.asarray(z), jnp.asarray(zv), 0.9, 10.0)
    return np.asarray(em), np.asarray(ex), {k: np.asarray(v) for k, v in info.items()}


def _sets_equal(ea, eb, tol):
    """Estimate sets (unordered) equal within tol."""
    if len(ea) != len(eb):
        return False
    if len(ea) == 0:
        return True
    ea, eb = ea[np.argsort(ea[:, 0])], eb[np.argsort(eb[:, 0])]
    return np.abs(ea - eb).max() < tol


@pytest.mark.parametrize("kind", ["phd", "cphd"])
@pytest.mark.parametrize("mode", ["paper", "code"])
def test_matches_official_octave_oracle(kind, mode):
    """81 steps, Table II config 3, 3 runs (1 for code/phd): Octave 10.3 running the authors' GMPHD/GMCPHD functions on the same
    measurements (frozen in tests/oracle_phd_ref.npz). Includes the official off-by-one in the 30-component cap (29 survive)."""
    em, ex, _ = _run(mode, kind)
    om, ox = FIX[f"{mode}_{kind}_mask"], FIX[f"{mode}_{kind}_x"]
    worst = 0.0
    for r in range(len(om)):
        for k in range(81):
            a, b = ex[r, k][em[r, k]], ox[r, k][om[r, k]]
            assert len(a) == len(b), (r, k)
            if len(a):
                a, b = a[np.argsort(a[:, 0])], b[np.argsort(b[:, 0])]
                worst = max(worst, np.abs(a - b).max())
    assert worst < 1e-9, worst


@pytest.mark.parametrize("kind", ["phd", "cphd"])
@pytest.mark.parametrize("mode", ["paper", "code"])
def test_matches_numpy_reference(kind, mode):
    """Independent dense float64 numpy implementation vs the JAX filter, first 25 steps of run 1 (causal)."""
    cfg = _cfg(mode)
    em, ex, _ = _run(mode, kind)
    z, zv = FIX[f"{mode}_{kind}_z"], FIX[f"{mode}_{kind}_zv"]
    r, K = 1, 25
    ref = phd.ref_run_phd([z[r, k][zv[r, k]] for k in range(K)], cfg, kind)
    for k in range(K):
        assert _sets_equal(ref[k], ex[r, k][em[r, k]], 1e-9), k
        assert len(ref[k]) == int(em[r, k].sum())


def test_log_esf_matches_polynomial():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(0, 12)
        lam = rng.uniform(0.0, 50.0, n)
        full = np.concatenate([lam, np.zeros(3)])  # padded zeros -> log = -inf inert
        logL = np.where(full > 0, np.log(np.where(full > 0, full, 1.0)), -np.inf)
        got = np.exp(np.asarray(phd._phd_log_esf(jnp.asarray(logL), 10)))
        want = phd._ref_esf(list(lam))
        want = np.concatenate([want, np.zeros(11)])[:11]
        assert np.allclose(got, want, rtol=1e-12, atol=0.0)


def test_batch_equals_single_and_cap():
    em, ex, info = _run("paper", "phd")
    cfg = _cfg("paper")
    z, zv = FIX["paper_phd_z"], FIX["paper_phd_zv"]
    K = 12
    em1, ex1 = jax.jit(lambda a, b: phd.run_gmphd(a, b, cfg, E=16))(jnp.asarray(z[2, :K]), jnp.asarray(zv[2, :K]))
    assert np.array_equal(em[2, :K], np.asarray(em1))
    assert np.allclose(ex[2, :K], np.asarray(ex1), atol=1e-10, rtol=0)
    for kind in ("phd", "cphd"):
        _, _, inf = _run("paper", kind)
        assert inf["n_active"].max() <= 29  # official cap: 30 merged, 29 kept (+1 birth enters the update)
        assert inf["n_trunc"].max() > 0  # the cap is active in this scenario


def test_measurement_order_invariance():
    rng = np.random.default_rng(2)
    for kind in ("phd", "cphd"):
        z, zv = FIX[f"code_{kind}_z"].copy(), FIX[f"code_{kind}_zv"]
        zp = z.copy()
        for r in range(z.shape[0]):
            for k in range(81):
                n = zv[r, k].sum()
                zp[r, k, :n] = z[r, k, :n][rng.permutation(n)]
        ea, xa, _ = _run("code", kind)
        eb, xb, _ = _run("code", kind, zp, zv)
        # deterministic up to float summation order; compare only the first 30 steps (the later trajectory is chaotic w.r.t. the
        # merge-order of near-tied candidates)
        for r in range(z.shape[0]):
            for k in range(30):
                assert _sets_equal(xa[r, k][ea[r, k]], xb[r, k][eb[r, k]], 1e-6), (kind, r, k)


def _single_target_world(rng, K, n_runs, lam_clutter, sigma, cfg):
    """One constant-velocity target, detected at EVERY step (hand generated), Poisson clutter on [0,300]^2."""
    F = cfg["F"]
    x = np.array([110.0, 1.0, 120.0, 0.5])
    X = np.zeros((1, K, 4))
    for k in range(K):
        X[0, k] = x
        x = F @ x
    z = np.zeros((n_runs, K, 12, 2))
    zv = np.zeros((n_runs, K, 12), bool)
    for r in range(n_runs):
        for k in range(K):
            nc = min(rng.poisson(lam_clutter), 11)
            z[r, k, 0] = X[0, k, [0, 2]] + sigma * rng.normal(size=2)
            z[r, k, 1:1 + nc] = rng.uniform(0, 300, (nc, 2))
            zv[r, k, :1 + nc] = True
    return dict(X=X, alive=np.ones((1, K), bool)), z, zv


@pytest.mark.parametrize("kind", ["phd", "cphd"])
def test_single_target_converges(kind):
    cfg = data.load_scenario_cfg(pd=0.99, lambda_c=2.0, q=1e-4)
    cfg["R"] = 0.04 * np.eye(2)  # nearly noiseless
    K, R = 40, 12
    truth, z, zv = _single_target_world(np.random.default_rng(1), K, R, 2.0, 0.2, cfg)
    em, ex, info = phd.make_phd_batch(cfg, kind, 16)(jnp.asarray(z), jnp.asarray(zv), 0.99, 2.0)
    em, ex = np.asarray(em), np.asarray(ex)
    n = em.sum(-1)  # [R,K]
    assert (n[:, 12:] == 1).mean() > 0.97  # cardinality estimate converges to the true value 1
    res = ev.evaluate_runs(em, ex, truth)
    late = np.sqrt(res["ospa2"][:, 12:].mean())
    assert late < 0.6, late  # sub-metre localisation of the single target with 0.2 m measurement noise
    if kind == "phd":
        assert abs(np.asarray(info["sum_w"])[:, 25:].mean() - 1.0) < 0.1  # PHD mass ~ number of targets


def test_all_dead_measurements_empty():
    """No measurements at all: estimate empty, everything finite."""
    z = np.zeros((3, 81, 48, 2))
    zv = np.zeros((3, 81, 48), bool)
    for kind in ("phd", "cphd"):
        em, ex, info = _run("paper", kind, z, zv)
        assert not em.any() and np.isfinite(ex).all() and np.isfinite(info["sum_w"]).all()


def test_phd_estimator_variant_est1_matches_reference():
    cfg = _cfg("paper")
    cfg["phd_estimator"] = "est1"
    f = phd.make_phd_batch(cfg, "phd", 16)
    z, zv = FIX["paper_phd_z"], FIX["paper_phd_zv"]
    em, ex, _ = f(jnp.asarray(z), jnp.asarray(zv), 0.9, 10.0)
    em, ex = np.asarray(em), np.asarray(ex)
    ref = phd.ref_run_phd([z[0, k][zv[0, k]] for k in range(25)], cfg, "phd")
    for k in range(25):
        assert _sets_equal(ref[k], ex[0, k][em[0, k]], 1e-9), k
    # est1 reports fewer targets than est2 here (weights spread over several components): the cardinality differs from the default
    em2, _, _ = _run("paper", "phd")
    assert em.sum() != em2.sum()
