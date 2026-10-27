"""Smoke test (CPU, budget 30 s): shapes, one step, jit, vmap-over-worlds == per-world loop.  Only TWO XLA compiles (the jitted
single-world scan and its vmap) are paid, both shared by the tests via a module fixture, because compile time is what the 30 s budget
buys.  The host == jax k-best backend comparison moved to test_backend_parity.py."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from src import data
from src.algorithm import PMBMConfig, run_filter, track_marginals
from src.model import make_lg_model, paper_lg_params

CFG = PMBMConfig(T=24, H=24, Nh=24, P=6, M=12, backend="host")
NW, K = 2, 4


@pytest.fixture(scope="module")
def runs():
    cfgs = data.load_scenario_cfg()
    root = jax.random.PRNGKey(18)
    truth = data.make_truth(jax.random.fold_in(root, 0), cfgs)
    z, zv, nd = data.make_measurements_runs(root, truth, 3, 0.9, 10, NW, cfgs, CFG.M, on_drop="ignore")
    z, zv = z[:, :K], zv[:, :K]
    model = make_lg_model(paper_lg_params())
    f = jax.jit(lambda a, b: run_filter(model, CFG, a, b))
    fv = jax.jit(jax.vmap(lambda a, b: run_filter(model, CFG, a, b)))
    loop = [f(z[w], zv[w]) for w in range(NW)]
    return loop, fv(z, zv), z, zv


def test_one_step_shapes(runs):
    (st, out), _, z, zv = runs[0][0], runs[1], runs[2], runs[3]
    assert st.gidx.shape == (CFG.Nh, CFG.T) and st.glogW.shape == (CFG.Nh,)
    assert st.trk_r.shape == (CFG.T, CFG.H) and st.trk_d["m"].shape == (CFG.T, CFG.H, 4)
    assert abs(float(jnp.sum(jnp.exp(st.glogW))) - 1.0) < 1e-12
    for e in ("est1", "est2", "est3"):
        m, x = out[e]                                   # stacked over the K steps
        assert m.shape == (K, CFG.est_E) and x.shape == (K, CFG.est_E, 4)
    mw, rm, used = track_marginals(st, CFG)
    assert mw.shape == (CFG.T, CFG.H) and np.all(np.asarray(rm) <= 1 + 1e-12)
    assert int(np.asarray(out["ovf"])[0].sum()) == 0 and int(np.asarray(out["est_over"])[0]) == 0   # first step (T = 24 is a toy size: later steps may fill slots)
    assert np.asarray(out["n_hyp"])[0] >= 1             # the first step produced a posterior


def test_vmap_equals_loop(runs):
    loop, (_, ov), z, zv = runs
    for w in range(NW):
        _, o = loop[w]
        for e in ("est1", "est2", "est3"):
            assert np.array_equal(np.asarray(o[e][0]), np.asarray(ov[e][0][w]))
            assert np.max(np.abs(np.asarray(o[e][1]) - np.asarray(ov[e][1][w]))) < 1e-9
