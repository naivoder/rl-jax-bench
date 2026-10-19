"""Smoke test (< 30 s on CPU): shapes and one jitted update for every arm on PD and 2D-LH."""
import jax
import numpy as np

from src import model as M
from src import train as TR
from src.data import get_spec


def _one_update(env, net, ep, arms):
    cfg = TR.TrainCfg(env=env, total_steps=400, n_envs=4, T=10, n_mb=2, early_stop=(env == "lh"))
    spec = get_spec(env)
    specs = [dict(arm=a, seed=0, lr=1e-3, alpha=8.0, ep=ep, opt_len=60.0) for a in arms]
    runs = TR.build_runs(cfg, specs)
    st = jax.vmap(lambda r: TR.init_run(cfg, spec, net, r))(runs)
    upd = jax.jit(jax.vmap(TR.make_update(cfg, spec, net)))
    st2, m = upd(st, runs)
    assert m["loss"].shape == (len(arms),)
    assert np.all(np.isfinite(np.asarray(m["loss"])))
    assert int(st2["u"][0]) == 1 and int(st2["total"][0]) == 40
    changed = jax.tree_util.tree_map(lambda a, b: np.abs(np.asarray(a) - np.asarray(b)).max(), st["params"], st2["params"])
    assert max(jax.tree_util.tree_leaves(changed)) > 0
    ev = jax.jit(jax.vmap(TR.make_eval(cfg, spec, net, 3), in_axes=(0, 0, None)), static_argnums=2)(st2["params"], runs, 0)
    assert ev["ret"].shape == (len(arms), 3)
    return m


def test_pd_all_arms_one_update():
    m = _one_update("pd", M.PDNet(hidden=16), {"public": False},
                    ["ppo", "bc", "dagger", "bc_tf1", "bc_ppo", "bc_then_ppo", "advisor"])
    assert list(np.asarray(m["mode"])) == [0, 1, 1, 1, 2, 1, 3]


def test_lh_one_update():
    _one_update("lh", M.LHNet(), {"i": 1, "j": 15}, ["bc", "ppo", "advisor"])
