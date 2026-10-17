"""Orbax resume is exact: a run resumed from a mid-run checkpoint reproduces the uninterrupted run's per-CPI outputs
(every random draw is fold_in(key, global step)). CPU, about 1 min."""
import os
import shutil
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402

import train as TR  # noqa: E402

CFG = TR.load_cfg(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "default.yaml"))


import pytest  # noqa: E402


@pytest.mark.parametrize("arm", ["dqn", "ddrqn"])
def test_resume_exact(tmp_path, arm):
    """Resume from the checkpoint written after CPI 3 (inside evaluation, past the train/eval boundary at 2)."""
    cfg = dict(CFG, train_cpis=2, eval_cpis=2, chunk_cpis=1, ckpt_minutes=0, frozen_eval=True)
    lanes, ids = TR.build_lanes(cfg, ["mc_p04"], ["eq3"], [0, 1])
    qstar = TR.exact_tables(ids, cfg)[1]
    full = TR.run_arm(cfg, arm, lanes, ids, str(tmp_path / "a"), qstar)
    ck = tmp_path / "a" / "ckpt" / arm
    steps = sorted(int(d) for d in os.listdir(ck) if d.isdigit())
    assert steps == [2, 3], steps                     # max_to_keep = 2
    shutil.rmtree(ck / "2")
    res = TR.run_arm(cfg, arm, lanes, ids, str(tmp_path / "a"), qstar)
    np.testing.assert_array_equal(res["outs"], full["outs"])
    np.testing.assert_array_equal(res["fps"], full["fps"])
    np.testing.assert_array_equal(res["frozen_outs"], full["frozen_outs"])
