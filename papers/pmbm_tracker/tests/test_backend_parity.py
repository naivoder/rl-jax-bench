"""host == pure-JAX k-best backend on a short run (split out of test_smoke.py)."""
import numpy as np
import jax

from src import data
from src.algorithm import PMBMConfig, run_filter
from src.model import make_lg_model, paper_lg_params

CFG = PMBMConfig(T=24, H=24, Nh=24, P=6, M=12, backend="host")


def test_host_equals_jax_backend():
    cfgs = data.load_scenario_cfg()
    root = jax.random.PRNGKey(18)
    truth = data.make_truth(jax.random.fold_in(root, 0), cfgs)
    z, zv, _ = data.make_measurements_runs(root, truth, 3, 0.9, 10, 1, cfgs, CFG.M, on_drop="ignore")
    z, zv = z[0, :6], zv[0, :6]
    model = make_lg_model(paper_lg_params())
    cfgj = CFG._replace(backend="jax")
    _, oj = jax.jit(lambda a, b: run_filter(model, cfgj, a, b))(z, zv)
    _, o0 = jax.jit(lambda a, b: run_filter(model, CFG, a, b))(z, zv)
    assert np.array_equal(np.asarray(o0["est1"][0]), np.asarray(oj["est1"][0]))
    assert np.max(np.abs(np.asarray(o0["est1"][1]) - np.asarray(oj["est1"][1]))) < 1e-9
