"""A full 200-step CFG-O closed loop with injected identical measurement sets and injected rollout choices
reproduces the authors' per-step GOSPA (<= 1e-8) and sensor trajectories for Myopic-GD and MCTS1-GD; plus the teacher-forced filter
check (sensor positions forced to the official ones) which isolates the filter + metric from any planner tie-flip."""
import os
import numpy as np
import jax, jax.numpy as jnp
import pytest
from src import algorithm as A
from src import model as M
from src import data as D
import official
import official_loop as OL

CFG = M.CFG_O
M_MEAS = M.M_MEAS
LAM = 2.0


def _pack_meas(rec):
    K = len(rec["z"])
    z = np.zeros((K, 2, M_MEAS, 2)); zv = np.zeros((K, 2, M_MEAS), bool)
    for k in range(K):
        for s in range(2):
            a = rec["z"][k][s]
            m = a.shape[1]
            assert m <= M_MEAS
            z[k, s, :m] = a.T; zv[k, s, :m] = True
    return z, zv


def _run_mine(alg, inj, steps):
    lane = A.make_lane_step(CFG, alg, nan_mode="off", inj=inj)
    carry = jax.tree_util.tree_map(lambda a: a, A.init_carry(CFG, 1))
    carry = jax.tree_util.tree_map(lambda a: a[0], carry)
    f = jax.jit(lambda c, k: lane(c, k, jnp.asarray(0), jnp.asarray(LAM)))
    outs = []
    for k in range(steps):
        carry, out = f(carry, jnp.asarray(k))
        outs.append(out)
    return carry, {kk: np.stack([np.asarray(o[kk]) for o in outs]) for kk in outs[0]}


@pytest.mark.parametrize("name", ["Myopic-GD", "MCTS1-GD"])
def test_closed_loop_matches_official(name):
    off = official.load()
    if off is None:
        pytest.skip(official.SKIP)
    alg = M.ALGS[name]
    steps = int(os.environ.get("GOSPA_PARITY_STEPS", "200"))
    truth = M.load_truth()
    rng = np.random.default_rng(5)
    Tm = max(alg.T - 1, 1)
    UJ = rng.uniform(size=(steps, alg.b_joint, Tm)); UI = rng.uniform(size=(steps, 2, alg.b_ind, Tm))
    rec = OL.run("GOSPA", alg.T, alg.b_joint, alg.b_ind, truth, UJ, UI, lam=LAM, n_steps=steps)
    z, zv = _pack_meas(rec)
    if name == "MCTS1-GD":
        # both planning modes (joint tree and per-sensor trees) must be exercised by this run
        assert rec["joint"].any() and (~rec["joint"]).any(), rec["joint"].mean()
    # --- teacher-forced: positions from the official run, planner bypassed -> filter + merge + metric must agree to 1e-8
    _, out_f = _run_mine(alg, dict(z=z, zv=zv, spos=rec["spos"]), steps)
    ef = np.max(np.abs(out_f["gospa_sq"] - rec["gospa"]) / (1 + rec["gospa"]))
    print(f"[METRIC] closed loop {name} {steps} steps teacher-forced: max rel GOSPA^2 err = {ef:.2e}")
    assert ef < 1e-8
    assert np.max(np.abs(out_f["loc"] - rec["loc"]) / (1 + rec["loc"])) < 1e-8
    assert (out_f["n_est"] == rec["nest"]).all()
    # --- free-running: own decisions with the injected rollout uniforms
    UIf = np.zeros((steps, 2, alg.b_ind, Tm)); UIf[:] = UI
    inj = dict(z=z, zv=zv, u=(UJ, np.zeros((steps, alg.b_joint)), UIf, np.zeros((steps, 2, alg.b_ind))))
    _, out = _run_mine(alg, inj, steps)
    same = np.all(np.abs(out["spos"] - rec["spos"]) < 1e-9, axis=(1, 2))
    first_bad = None if same.all() else int(np.argmin(same))
    assert first_bad is None, f"first trajectory divergence at step {first_bad}"
    assert (out["joint"].astype(bool) == rec["joint"]).all()
    e2 = np.max(np.abs(out["gospa_sq"] - rec["gospa"]) / (1 + rec["gospa"]))
    print(f"[METRIC] closed loop {name} {steps} steps free-running: sensor tracks identical, joint-mode fraction {rec['joint'].mean():.2f}, max rel GOSPA^2 err = {e2:.2e}")
    assert e2 < 1e-8
