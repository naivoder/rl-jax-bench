"""Checkpoint/resume: a run interrupted at iteration 10 and resumed to 20 ends with the same parameters as an uninterrupted 20-iteration
run (orbax round trip of the vmapped QState incl. optimizer state; keys are pure functions of (key, iteration)).  Tiny network, CPU."""
import os
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402

import train  # noqa: E402

COMMON = ["--exp", "E0", "--seeds", "1", "2", "--chunk", "10", "--ckpt-every", "10", "--monitor-every", "1000", "--n-exp", "8",
          "--lstm", "8", "--sched-iters", "20"]


def test_resume_equals_uninterrupted():
    with tempfile.TemporaryDirectory() as d:
        train.main(COMMON + ["--iters", "20", "--out", os.path.join(d, "full")])
        train.main(COMMON + ["--iters", "10", "--out", os.path.join(d, "split")])
        train.main(COMMON + ["--iters", "20", "--out", os.path.join(d, "split")])        # resumes at step 10
        tag = os.listdir(os.path.join(d, "full"))[0]
        for s in (1, 2):
            a = np.load(os.path.join(d, "full", tag, f"seed_{s}", "params.npz"))
            b = np.load(os.path.join(d, "split", tag, f"seed_{s}", "params.npz"))
            assert sorted(a.files) == sorted(b.files)
            for k in a.files:
                np.testing.assert_allclose(a[k], b[k], atol=1e-6, rtol=1e-5, err_msg=k)
