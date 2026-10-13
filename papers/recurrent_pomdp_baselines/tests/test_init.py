"""Initialisation parity with the official code (review).  Every differential test loads the official weights,
so none of them checks OUR initialiser; this test does.  For every parameter tensor of our freshly initialised
separate / shared agents (sac, td3 x lstm, gru) and the official initial tensors in tests/fixtures/nets_*.npz
(``init/``; recorded by scripts/official/dump_fixtures.py from the unmodified official modules):
  * MLP hidden layers: U(+-1/sqrt(OUT dim)) (rlkit fanin_init quirk, README), bias exactly 0.1;
  * MLP output layers: U(+-1e-3) for the actor (MarkovPolicyBase init_w=1e-3), U(+-3e-3) for the critic,
    weight and bias;
  * embedders / shortcuts: torch nn.Linear default U(+-1/sqrt(IN dim)) for weight and bias;
  * RNN: orthogonal weight matrices (W^T W = I for the (4H or 3H) x n weight), zero biases;
and the two samples agree in distribution (two-sample KS, tensors with >= 32 entries).
"""
import os

import jax
import numpy as np
import pytest
from scipy import stats

from src import common  # noqa: F401
from src.algorithm import Agent
from src.model import Nets
from tests import oracle_map as M

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
CASES = [(a, c, r) for a in ("sac", "td3") for c in ("lstm", "gru") for r in ("separate", "shared")]


def _bound(path, x):
    names = [getattr(p, "key", str(p)) for p in path]
    leaf, mod = names[-1], names[-2]
    if mod in ("w_ih", "w_hh") or leaf in ("w_ih", "w_hh"):
        return "orth", None
    if leaf in ("b_ih", "b_hh"):
        return "const", 0.0
    if mod.startswith("last_fc"):
        return "unif", 1e-3 if ("pi" in names) else 3e-3
    if mod.startswith("fc") and mod[2:].isdigit():
        return ("unif", 1.0 / np.sqrt(x.shape[0])) if leaf == "weight" else ("const", 0.1)
    # embedders and shortcut embedders: torch Linear default, bound 1/sqrt(in_features) for weight AND bias
    return "unif", None


@pytest.mark.parametrize("algo,cell,arch", CASES)
def test_init_matches_official(algo, cell, arch):
    z = np.load(os.path.join(FIX, f"nets_{algo}_{cell}_{arch}.npz"))
    cfg = dict(M.ORACLE_SIZES, algo=algo, cell=cell, arch=arch)
    ag = Agent(cfg, Nets(cfg, obs_dim=1, act_dim=1, discrete=False))
    ours = ag.init(jax.random.PRNGKey(123))["params"]
    sd = M.state_dict(z, "init/")
    off = M.separate_params(sd, algo) if arch == "separate" else M.shared_params(sd, algo)
    po, to = jax.tree_util.tree_flatten_with_path(ours)
    pf, tf = jax.tree_util.tree_flatten_with_path(off)
    assert to == tf
    # embedder bias bound needs the weight's in_features: look it up from the sibling weight
    flat_off = {tuple(getattr(p, "key", str(p)) for p in path): np.asarray(v) for path, v in pf}
    n_checked = 0
    for (path, a), (_, b) in zip(po, pf):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        assert a.shape == b.shape
        kind, bound = _bound(path, a)
        key = tuple(getattr(p, "key", str(p)) for p in path)
        if kind == "orth":
            for w in (a, b):
                assert np.allclose(w.T @ w, np.eye(w.shape[1]), atol=1e-5), key
        elif kind == "const":
            assert np.all(a == np.float32(bound)) and np.all(b == np.float32(bound)), key
        else:
            if bound is None:
                wkey = key[:-1] + ("weight",)
                bound = 1.0 / np.sqrt(flat_off[wkey].shape[1])
            for w in (a, b):
                assert np.abs(w).max() <= bound * (1 + 1e-6), (key, np.abs(w).max(), bound)
            if a.size >= 32:
                assert np.abs(a).max() > 0.8 * bound, (key, np.abs(a).max(), bound)
                p = stats.ks_2samp(a.ravel(), b.ravel()).pvalue
                assert p > 1e-4, (key, p)
        n_checked += 1
    assert n_checked == len(po)
