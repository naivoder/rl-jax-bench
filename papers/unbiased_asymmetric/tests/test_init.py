"""Initialization distributions: moment checks on fresh parameters against the official
torch initializers (asym-rlpo@1a065b0 modules.make_module + torch defaults):
Linear kernels xavier_normal (std = gain*sqrt(2/(fan_in+fan_out)), gain sqrt(2) hidden / 1 output), zero bias;
GRU weights and biases U(-1/sqrt(H), 1/sqrt(H)); Embedding N(0, 1)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax  # noqa: E402

from envs import make_env  # noqa: E402
from model import NetSpec, init_params  # noqa: E402


@pytest.mark.parametrize("env,method", [("heavenhell_3", "a2c_asym_hs"), ("shopping_5", "a2c"),
                                        ("cleaner", "a2c_asym_s")])
def test_init_moments(env, method):
    e = make_env(env)
    sp = NetSpec(e.kind, e.n_actions, e.n_obs, e.n_states)
    pa, pc = init_params(sp, method, jax.random.PRNGKey(123))
    n_checked = 0
    for tree in (pa, pc):
        for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
            name = jax.tree_util.keystr(path)
            x = np.asarray(leaf, np.float64)
            if name.endswith("['bias']"):
                assert np.all(x == 0.0), name
            elif name.endswith("['kernel']"):
                fi, fo = x.shape
                gain = 1.0 if "['out']" in name else math.sqrt(2.0)
                sd = gain * math.sqrt(2.0 / (fi + fo))
                # sample std of n iid normals: relative s.e. ~ 1/sqrt(2n); allow 6 s.e. (+ small n floor)
                tol = 6.0 / math.sqrt(2 * x.size) + 0.02
                assert abs(x.std() / sd - 1.0) < tol, (name, x.std(), sd)
                assert abs(x.mean()) < 6 * sd / math.sqrt(x.size), name
            elif "['cell']" in name:
                b = 1.0 / math.sqrt(sp.hidden)
                assert x.min() >= -b and x.max() <= b, name
                assert abs(x.std() / (b / math.sqrt(3.0)) - 1.0) < 6.0 / math.sqrt(x.size) + 0.02, name
            elif name.endswith("_emb']"):
                assert abs(x.std() - 1.0) < 6.0 / math.sqrt(2 * x.size) + 0.05, name
                assert abs(x.mean()) < 6.0 / math.sqrt(x.size), name
            else:
                continue
            n_checked += 1
    assert n_checked >= 8
