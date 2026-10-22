"""Smoke tests (< 30 s, CPU): configuration/sources agreement, shapes of one lane-vmapped step, health counters."""
import os
import numpy as np
import yaml
import jax, jax.numpy as jnp
from src import algorithm as A
from src import model as M


def test_config_matches_sources():
    src = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "sources.yaml")))
    s = src["sensors"]; m = src["motion"]; p = src["planner"]; f = src["filter"]
    assert M.CFG_O.r_b == m["r_b"]["CFG_O"] == 0.05 and M.CFG_P.r_b == m["r_b"]["CFG_P"] == 0.03
    assert M.N_STEPS == src["scenario"]["n_steps"]["value"] and M.PSI == p["psi_joint"]["value"]
    assert M.CFG_O.lam == p["discount_lambda"]["value"] == M.CFG_P.lam and M.CFG_O.uct_eps == p["uct_eps"]["value"]
    assert M.CFG_O.lbp_tol == f["lbp_tol"]["CFG_O"] and M.CFG_P.lbp_tol == f["lbp_tol"]["CFG_P"]
    assert M.NB_MAX == f["max_bernoullis"]["value"] and M.EST_THRESH == f["estimate_threshold"]["value"]
    assert M.GOSPA_C == src["eval"]["gospa_c"]["value"] and M.N_RUNS_PAPER == src["eval"]["n_mc_runs"]["value"]
    for name, (T, bj, bi) in {"Myopic": (1, 49, 7), "MCTS1": (5, 49, 7), "MCTS2": (10, 49, 7), "MCTS3": (5, 200, 40), "MCTS4": (10, 200, 40)}.items():
        a = M.ALGS[name + "-GD"]
        assert (a.b_joint, a.b_ind, a.T) == tuple(p["variants"][name]["value"])  # sources.yaml order: [budget_joint, budget_ind, lookahead]
        assert (a.T, a.b_joint, a.b_ind) == (T, bj, bi)
    assert abs(M.Q_ncv()[1, 1] - 0.8) < 1e-12 and np.allclose(M.F_ncv()[:2, :2], [[1, 1], [0, 1]])


def _check(cfg):
    alg = M.ALGS["Myopic-GD"]
    run = A.make_chunk_runner(cfg, alg, nan_mode="off")
    L = 3
    carry = A.init_carry(cfg, L)
    ms = jnp.arange(L); lams = jnp.array([0.1, 1.0, 2.0])
    carry, out = run(carry, jnp.arange(3), ms, lams)
    assert out["gospa_sq"].shape == (3, L) and out["spos"].shape == (3, L, 2, 2)
    assert np.isfinite(np.asarray(out["gospa_sq"])).all()
    cnt = dict(zip(A.COUNTERS, np.asarray(carry.cnt).sum(0)))
    assert all(v == 0 for k, v in cnt.items() if k != "plan_nan_steps"), cnt
    # before the first birth (t = 6) the truth is empty: GOSPA^2 is 0 without estimates
    assert np.asarray(out["gospa_sq"])[0].max() == 0.0
    # sensors moved at most one step (15) from the initial positions
    d = np.linalg.norm(np.asarray(out["spos"])[0] - np.asarray(M.SENSOR_INIT)[None], axis=-1)
    assert d.max() <= 15.0 + 1e-9


def test_one_step_shapes_and_counters_cfgO():
    _check(M.CFG_O)
