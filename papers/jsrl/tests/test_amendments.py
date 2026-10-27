"""Tests for the amendments: eps annealed on pi_e's own steps (variant eps_own), the stage-matched early statistics
(H1m / H1p) and the combined-policy early score (H1c) of src/eval.py (amendment A1), and the S-C4final relative TOST margin (A2).  CPU, < 30 s."""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax, jax.numpy as jnp, numpy as np
from src import algorithm as AL, replay as R, eval as E
from tests.test_smoke import _setup

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _chunk(eps_own):
    env, L, fns, names, hp = _setup()
    hp0 = {k: v[0] for k, v in hp.items()}                          # JSRL arm; stage 0 = h = H1 = 6 = the whole 6-step lock episode
    hp0 = dict(hp0, eps_decay=jnp.float32(200), eps_own=jnp.float32(eps_own))
    key = jax.random.PRNGKey(0)
    params = fns.init_params(key)
    c = fns.init_interact(params, params, fns.tx.init(params), R.init_buffer(L, None, hp0), key, jax.random.fold_in(key, 1005), hp0)
    c = jax.jit(fns.interact_chunk, static_argnums=(3,))(c, hp0, jax.random.fold_in(key, 3), 100, jnp.array(False))
    return c


def test_eps_own_counts_only_pi_e_steps():
    c_def, c_own = _chunk(0.0), _chunk(1.0)
    assert int(c_def["guide_steps"]) == 100 and int(c_def["tpi"]) == 0      # every step of the lock episode is a guide step at stage 0
    n = float(c_def["dg"]["steps"])
    assert abs(float(c_own["dg"]["eps"]) / n - 1.0) < 1e-6                  # schedule frozen at eps0 = 1: no pi_e step has happened
    assert float(c_def["dg"]["eps"]) / n < 0.8                              # default: global counter, eps = 1 - t/200 averaged over 100 steps = 0.75


def _fake_runs(n_seeds=3, n_eval=20, n_stages=4):
    rng = np.random.default_rng(0)
    runs = {"JSRL": {}, "PDD": {}}
    for s in range(n_seeds):
        stage = np.minimum(np.arange(n_eval + 1) // 4, n_stages)
        runs["JSRL"][s] = dict(ev_alone=rng.random(n_eval + 1), ev_thr=rng.random(n_eval + 1), ev_stage=stage, ev_greedy=np.zeros(n_eval + 1))
        runs["PDD"][s] = dict(ev_alone=rng.random(n_eval + 1), ev_thr=rng.random(n_eval + 1), ev_stage=np.zeros(n_eval + 1), ev_greedy=np.zeros(n_eval + 1))
    return runs


def test_stage_matched_indexing():
    runs = _fake_runs()
    sm = E.stage_matched(runs, 20, 4, win=5)
    assert sm["f"] == [16, 16, 16] and sm["n_excluded"] == 0                # stage 4 first reached at point 16
    for i, s in enumerate(sm["seeds"]):
        assert sm["jsrl_at"][i] == runs["JSRL"][s]["ev_alone"][16] and sm["pdd_at"][i] == runs["PDD"][s]["ev_alone"][16]
        assert np.isclose(sm["jsrl_win"][i], runs["JSRL"][s]["ev_alone"][16:21].mean())       # window clipped at the last point (20)
        assert np.isclose(sm["pdd_win"][i], runs["PDD"][s]["ev_alone"][16:21].mean())
    runs["JSRL"][1]["ev_stage"] = np.zeros(21)                              # a seed that never reaches h = 0 is excluded and counted
    sm = E.stage_matched(runs, 20, 4, win=5)
    assert sm["seeds"] == [0, 2] and sm["n_excluded"] == 1


def test_scores_early_comb_and_tost_relative_margin():
    r = dict(ev_alone=np.arange(101.0), ev_thr=np.full(101, 7.0), ev_greedy=np.zeros(101))
    sc = E.scores(r, 100)
    assert sc["early_comb"] == 7.0 and sc["early"] == 5.5 and sc["final"] == 99.0   # points 1..10 and 98..100
    d = np.array([0.1, -0.2, 0.3, 0.0, 0.2])
    assert E.tost(d, 1.7)["p"] < E.tost(d, 0.7)["p"]                         # a wider margin makes equivalence easier to establish
