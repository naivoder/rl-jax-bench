"""Differential tests against Stone Soup 1.9.1 (planned check 2; the authors' code is unavailable, Stone Soup is the
library they name). The oracle runs in its own venv through scripts/official/ss_ref.py; this file builds identical inputs
with our JAX code, runs the oracle once (cached per session) and compares. float64 throughout.

  B4  UKF measurement prediction / Mahalanobis distance / update vs UnscentedKalmanUpdater   (<= 1e-9 relative)
  B5  SimpleMeasurementInitiator                                                              (<= 1e-6 relative)
  B6  full MultiTargetTracker on >= 20 scripted scenarios (same detections): track counts exact, states <= 1e-6
   GOSPA vs GOSPAMetric on >= 1000 random sets                                             (<= 1e-9 relative)
"""
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common  # noqa: F401,E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from dataclasses import replace  # noqa: E402

from src.configs.config import EnvConfig  # noqa: E402
from src import data, baselines as B  # noqa: E402

OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official"))
ORACLE_PY = os.path.join(OFFICIAL_ROOT, ".venv", "bin", "python")
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = replace(EnvConfig(), gospa_metrics=False)

pytestmark = pytest.mark.skipif(not os.path.exists(ORACLE_PY), reason="Stone Soup oracle venv missing: run sh scripts/official/setup.sh")


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-12 * (1 + np.max(np.abs(b))))))


def maxrel_scaled(a, b):
    """max |a-b| / max|b| per array (scale-aware relative error)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-300))


def make_inputs(n_ukf=200, n_init=200, n_traj=20, T=150, n_gospa=1000, mode="fd"):
    rng = np.random.default_rng(0)
    CFG = replace(globals()["CFG"], init_jacobian=mode)
    R = np.asarray(data.meas_R(CFG))
    d = dict(R=R, vel_sigma=CFG.init_vel_sigma, dt=CFG.dt, q=CFG.q_filt, gate=CFG.gate, missed=CFG.missed,
             del_thr=CFG.del_thr, k_max=CFG.k_max, c=CFG.gospa_c, p=CFG.gospa_p, analytic_jac=int(mode == "analytic"))
    # UKF cases: realistic states (initiation covariance grown by 0..40 predict steps), measurement near the state
    xs, Ps, zs = [], [], []
    F, Q = (np.asarray(m) for m in data.cv_FQ(CFG.dt, CFG.q_filt))
    for i in range(n_ukf):
        r, az, el = rng.uniform(1e4, 3e4), np.deg2rad(rng.uniform(-60, 60)), np.deg2rad(rng.uniform(-30, 30))
        z0 = np.array([el, az, r])
        x0, P0 = (np.asarray(v) for v in data.initiate_state(jnp.asarray(z0), CFG))
        x0 = x0 + np.array([0, rng.normal(0, 30), 0, rng.normal(0, 30), 0, 0])
        for _ in range(rng.integers(0, 40)):
            x0, P0 = F @ x0, F @ P0 @ F.T + Q
        zt = np.asarray(data.h_meas(jnp.asarray(x0))) + rng.normal(0, 1, 3) * np.sqrt(np.diag(R)) * 2
        xs.append(x0), Ps.append(P0), zs.append(zt)
    d.update(ukf_x=np.array(xs), ukf_P=np.array(Ps), ukf_z=np.array(zs))
    # initiator cases
    iz = np.stack([np.deg2rad(rng.uniform(-30, 30, n_init)), np.deg2rad(rng.uniform(-60, 60, n_init)),
                   rng.uniform(1e4, 3e4, n_init)], -1)
    d.update(init_z=iz)
    # trajectory cases: detections produced by our environment under the truth-cued ceiling
    cfgT = replace(CFG, T=T)
    pinit, pol = B.make_ceiling(cfgT)   # truth-cued looks: many detections, revisits, duplicates, deletions

    def episode(k):
        s = data.reset(k, cfgT)
        o = data.observe(s, cfgT)

        def body(c, i):
            s, o, p = c
            a, p = pol(p, s, o, k)
            s, o, info = data.step(s, a, cfgT)
            return (s, o, p), (info["z"], info["zvalid"], s.tr.x, s.tr.P, s.tr.valid)
        _, out = jax.lax.scan(body, (s, o, pinit(k)), jnp.arange(T))
        return out

    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(2024), i))(jnp.arange(n_traj))
    z, zv, tx, tP, tv = (np.asarray(a) for a in jax.jit(jax.vmap(episode))(keys))
    d.update(traj_z=z, traj_valid=zv)
    ours = dict(traj_x=tx, traj_P=tP, traj_valid_tr=tv)
    # GOSPA cases: 3 truths, up to K tracks, some near truths, some far, some invalid
    gt = rng.normal(0, 1, (n_gospa, 3, 6)) * np.array([8e3, 50, 8e3, 50, 3e3, 5])
    gk = np.concatenate([gt[:, rng.integers(0, 3, CFG.k_max)] + rng.normal(0, 1, (n_gospa, CFG.k_max, 6))
                         * rng.choice([10.0, 3e3, 2e4], (n_gospa, CFG.k_max, 1))], 1)
    gv = rng.uniform(size=(n_gospa, CFG.k_max)) < rng.uniform(0, 0.6, (n_gospa, 1))
    d.update(gospa_truth=gt, gospa_tracks=gk, gospa_valid=gv)
    return d, ours


# "fd": both sides use Stone Soup's forward-difference initiator Jacobian (the library as the authors ran it); its
# 1e-8 step floor amplifies last-bit differences between numpy and XLA transcendental functions into ~1e-4 relative
# differences of C0 (measured: 2.0e-3 on C0, 3.5e-3 along trajectories, counts identical), so this mode is compared
# at TOL["fd"]. "analytic": both sides use the exact Jacobian, which removes
# that round-off and must agree to the declared 1e-6 (trajectories) / 1e-9 (single operations).
TOL = {"fd": dict(init_P=1e-2, traj=1e-2), "analytic": dict(init_P=1e-9, traj=1e-6)}   # fd measured 2.0e-3 / 3.5e-3


@pytest.fixture(scope="module", params=["analytic", "fd"])
def oracle(request, tmp_path_factory):
    mode = request.param
    d, ours = make_inputs(mode=mode)
    ours["mode"] = mode
    tmp = tmp_path_factory.mktemp("oracle_" + mode)
    fin, fout = str(tmp / "in.npz"), str(tmp / "out.npz")
    np.savez(fin, **d)
    subprocess.run([ORACLE_PY, "-I", os.path.join(os.path.dirname(HERE), "scripts", "official", "ss_ref.py"), fin, fout], check=True, cwd=str(tmp))
    return d, ours, dict(np.load(fout))


def test_B4_ukf_vs_stonesoup(oracle):
    d, _, o = oracle
    R = jnp.asarray(d["R"])
    f = jax.jit(jax.vmap(lambda x, P: data.ukf_meas_pred(x, P, R, CFG)))
    zh, S, C = (np.asarray(a) for a in f(jnp.asarray(d["ukf_x"]), jnp.asarray(d["ukf_P"])))
    sig = np.sqrt(np.diag(d["R"]))
    assert np.max(np.abs(zh - o["ukf_zhat"]) / sig) < 1e-9          # per component, in measurement-noise units
    for a, b in ((S, o["ukf_S"]), (C, o["ukf_C"])):
        for i in range(len(a)):
            assert maxrel_scaled(a[i], b[i]) < 1e-9
    nu = np.asarray(jax.vmap(data.innovation)(jnp.asarray(d["ukf_z"]), jnp.asarray(zh)))
    D = np.sqrt(np.einsum("ni,nij,nj->n", nu, np.linalg.inv(S), nu))
    # Stone Soup's Mahalanobis measure carries ~1e-8 of its own round-off: our D equals the exact formula evaluated on
    # Stone Soup's own (zhat, S) to ~3e-12 (implementation diagnosis), so the measure itself is compared at 1e-7.
    nuo = np.asarray(jax.vmap(data.innovation)(jnp.asarray(d["ukf_z"]), jnp.asarray(o["ukf_zhat"])))
    Do = np.sqrt(np.einsum("ni,nij,nj->n", nuo, np.linalg.inv(o["ukf_S"]), nuo))
    assert rel(D, Do) < 1e-9
    assert rel(D, o["ukf_D"]) < 1e-7
    Kg = C @ np.linalg.inv(S)
    xu = d["ukf_x"] + np.einsum("nij,nj->ni", Kg, nu)
    Pu = d["ukf_P"] - Kg @ S @ np.swapaxes(Kg, 1, 2)
    for i in range(len(xu)):
        assert maxrel_scaled(xu[i], o["ukf_xu"][i]) < 1e-9
        assert maxrel_scaled(Pu[i], o["ukf_Pu"][i]) < 1e-9


def test_B5_initiator_vs_stonesoup(oracle):
    d, _, o = oracle
    _, ours, o = oracle
    cfg = replace(CFG, init_jacobian=ours["mode"])
    x, P = (np.asarray(a) for a in jax.vmap(lambda z: data.initiate_state(z, cfg))(jnp.asarray(d["init_z"])))
    for i in range(len(x)):
        assert maxrel_scaled(x[i], o["init_x"][i]) < 1e-9
        assert maxrel_scaled(P[i], o["init_P"][i]) < TOL[ours["mode"]]["init_P"]


def test_B6_tracker_trajectories_vs_stonesoup(oracle):
    d, ours, o = oracle
    S, T = d["traj_z"].shape[:2]
    assert S >= 20
    n_steps_with_tracks = 0
    worst = 0.0
    for s in range(S):
        for t in range(T):
            v = ours["traj_valid_tr"][s, t]
            n = int(v.sum())
            assert n == int(o["traj_count"][s, t]), f"track count differs at scenario {s} step {t}"
            if n == 0:
                continue
            n_steps_with_tracks += 1
            xo = ours["traj_x"][s, t][v]
            Po = ours["traj_P"][s, t][v]
            order = np.lexsort((xo[:, 4], xo[:, 2], xo[:, 0]))
            xo, Po = xo[order], Po[order]
            xr, Pr = o["traj_x"][s, t, :n], o["traj_P"][s, t, :n]
            worst = max(worst, maxrel_scaled(xo, xr), max(maxrel_scaled(Po[j], Pr[j]) for j in range(n)))
    assert n_steps_with_tracks > 500
    print(f"[B6 {ours['mode']}] worst scaled relative state/covariance difference {worst:.3e}")
    assert worst < TOL[ours["mode"]]["traj"], worst


def test_D2_gospa_vs_stonesoup(oracle):
    d, _, o = oracle
    f = jax.jit(jax.vmap(lambda a, b, v: jnp.stack(data.gospa(a, b, v, CFG.gospa_c, CFG.gospa_p))))
    g = np.asarray(f(jnp.asarray(d["gospa_truth"]), jnp.asarray(d["gospa_tracks"]), jnp.asarray(d["gospa_valid"])))
    ref = o["gospa"]
    assert len(ref) >= 1000
    assert rel(g[:, 0], ref[:, 0]) < 1e-9                       # distance
    for k in (1, 2, 3):                                          # localisation, missed, false
        assert np.max(np.abs(g[:, k] - ref[:, k]) / (np.abs(ref[:, 0]) ** 2 + 1.0)) < 1e-9
