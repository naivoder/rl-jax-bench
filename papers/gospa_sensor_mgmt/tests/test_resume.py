"""orbax checkpoint/resume of the experiment driver: a run continued from a chunk checkpoint equals an uninterrupted run."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_resume_equals_uninterrupted(tmp_path):
    from src import train as T          # imports gpu_bootstrap (no-op on CPU) and enables x64
    from src import model as M
    cfg, alg = M.CFG_O, M.ALGS["Myopic-GD"]
    a = T.run_cell(cfg, alg, 2, [1.0], str(tmp_path / "a"), chunk=5, steps=10)        # writes checkpoints at k = 5, 10
    b = T.run_cell(cfg, alg, 2, [1.0], str(tmp_path / "a"), chunk=5, steps=15)        # resumes from k = 10
    c = T.run_cell(cfg, alg, 2, [1.0], str(tmp_path / "c"), chunk=5, steps=15, ckpt=False)   # uninterrupted
    assert b["outs"]["gospa_sq"].shape[0] == 15
    for k in c["outs"]:
        assert np.array_equal(b["outs"][k], c["outs"][k]), k
    assert np.array_equal(a["outs"]["gospa_sq"], c["outs"]["gospa_sq"][:10])
