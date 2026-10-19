"""review: the paper's early-stop / best-model selection (train.run_family select_best_patience) on a tiny
Flat run (CPU): with patience 0 the run stops at the first curve point after step 0 that does not improve every lane,
and the returned parameters are each lane's best by its own validation return."""
import os
import sys
import tempfile
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common  # noqa: F401,E402
import jax  # noqa: E402

from src.configs.config import Config  # noqa: E402
from src import train as TR  # noqa: E402


def test_select_best_and_early_stop():
    cfg = Config()
    cfg = replace(cfg, env=replace(cfg.env, T=40), ppo=replace(cfg.ppo, n_steps=32, n_envs=2, n_epochs=1, minibatch=32,
                                                               curve_scenarios=2),
                  pre=replace(cfg.pre, ae_steps=2, bc_episodes=1, bc_epochs=1))
    with tempfile.TemporaryDirectory() as d:
        net, lanes, meta, summ = TR.run_family(cfg, ["flat"], [0, 1], d, total_steps=64 * 6, curve_every=64,
                                               ckpt_every_it=0, select_best_patience=64)
        import pickle
        b = pickle.load(open(os.path.join(d, "params_best_flat.pkl"), "rb"))
    curves = [s["curve"] for s in summ]
    for i, c in enumerate(curves):
        rets = [p["ret_flip"] for p in c]
        j = int(np.argmax(rets))
        assert b["best_step"][i] == c[j]["env_steps"]
        assert np.isclose(b["best_ret"][i], rets[j])
    leaves_best = jax.tree.leaves(b["params"])
    leaves_ret = jax.tree.leaves(lanes.params)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(leaves_best, leaves_ret))
    n_pts = len(curves[0])
    assert n_pts <= 7


def test_select_best_deterministic_SA15():
    """: with select_deterministic the best model is chosen on the deterministic validation return."""
    import json
    cfg = Config()
    cfg = replace(cfg, env=replace(cfg.env, T=40), ppo=replace(cfg.ppo, n_steps=32, n_envs=2, n_epochs=1, minibatch=32,
                                                               curve_scenarios=2),
                  pre=replace(cfg.pre, ae_steps=2, bc_episodes=1, bc_epochs=1))
    with tempfile.TemporaryDirectory() as d:
        net, lanes, meta, summ = TR.run_family(cfg, ["flat"], [0, 1], d, total_steps=64 * 4, curve_every=64,
                                               ckpt_every_it=0, select_best_patience=10_000, select_deterministic=True)
        import pickle
        b = pickle.load(open(os.path.join(d, "params_best_flat.pkl"), "rb"))
        prog = json.load(open(os.path.join(d, "progress_flat.json")))
    cd = {int(k): v for k, v in prog["curves_det"].items()}
    for i in range(2):
        rets = [p["ret_flip"] for p in cd[i]]
        j = int(np.argmax(rets))
        assert b["best_step"][i] == cd[i][j]["env_steps"]
        assert np.isclose(b["best_ret"][i], rets[j])


def test_select_best_survives_resume():
    """Paper-budget run: a run stopped after a checkpoint and resumed selects the same per-lane best model (step,
    return and parameters) as an uninterrupted run; the best-model state is saved with each checkpoint."""
    import pickle
    cfg = Config()
    cfg = replace(cfg, env=replace(cfg.env, T=40), ppo=replace(cfg.ppo, n_steps=32, n_envs=2, n_epochs=1, minibatch=32,
                                                               curve_scenarios=2),
                  pre=replace(cfg.pre, ae_steps=2, bc_episodes=1, bc_epochs=1))
    kw = dict(curve_every=64, ckpt_every_it=2, select_best_patience=10_000, select_deterministic=True)
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        TR.run_family(cfg, ["flat"], [0, 1], d1, total_steps=64 * 6, **kw)
        TR.run_family(cfg, ["flat"], [0, 1], d2, total_steps=64 * 5, **kw)     # checkpoints at it 2 and 4, then "stops"
        assert os.path.exists(os.path.join(d2, "best_state_flat_4.pkl"))
        TR.run_family(cfg, ["flat"], [0, 1], d2, total_steps=64 * 6, **kw)     # resumes from it 4
        a = pickle.load(open(os.path.join(d1, "params_best_flat.pkl"), "rb"))
        b = pickle.load(open(os.path.join(d2, "params_best_flat.pkl"), "rb"))
    assert np.array_equal(a["best_step"], b["best_step"])
    assert np.allclose(a["best_ret"], b["best_ret"])
    for x, y in zip(jax.tree.leaves(a["params"]), jax.tree.leaves(b["params"])):
        assert np.allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-6)
