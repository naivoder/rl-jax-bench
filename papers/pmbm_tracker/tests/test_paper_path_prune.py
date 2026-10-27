"""The paper path has NO global-hypothesis weight floor (Alg. 1 keeps the top N_h).
Seed-18 measurement set, config 3, run 0, three steps: with prune_w = 0 every 2^4 continuation subset of the four k = 1 tracks survives at
k = 2 and Est 1 confirms all four targets at k = 3; the authors' HEAD floor 1e-4 removes the multi-target continuations (12 hypotheses) and
Est 1 confirms only three.  This pins the mechanism that carries the Table II offset (see README)."""
import numpy as np
import jax

from src import data
from src.algorithm import run_filter
from src.model import make_lg_model, paper_lg_params, paper_pmbm_config


def test_global_weight_floor_changes_k2_hypotheses_and_k3_confirmation():
    cfgs = data.load_scenario_cfg()
    root = jax.random.PRNGKey(18)
    truth = data.make_truth(jax.random.fold_in(root, 0), cfgs)
    z, zv, _ = data.make_measurements_runs(root, truth, 3, 0.9, 10, 1, cfgs, 48, on_drop="raise")
    model = make_lg_model(paper_lg_params())
    res = {}
    for pw in (0.0, 1e-4):
        cfg = paper_pmbm_config(T=48, H=48, M=48, prune_w=pw)
        _, o = jax.jit(lambda a, b: run_filter(model, cfg, a, b))(z[0, :3], zv[0, :3])
        assert int(np.asarray(o["ovf"]).sum()) == 0
        res[pw] = (np.asarray(o["n_hyp"]).tolist(), [int(m.sum()) for m in np.asarray(o["est1"][0])])
    assert res[0.0][0][:2] == [1, 16] and res[1e-4][0][:2] == [1, 12]
    assert res[0.0][1][2] == 4 and res[1e-4][1][2] == 3
