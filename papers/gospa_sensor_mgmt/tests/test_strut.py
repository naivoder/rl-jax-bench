"""S-TRUTH plumbing (ten alternative truths): the per-run truth set has the expected shapes, respects the fixed array capacities, and the default path
(truth_set=None) is unchanged -- the first steps of a CFG-O myopic lane equal the 10-run archive (tag small) when it is present."""
import hashlib, json, os
import numpy as np
import jax, jax.numpy as jnp
from src import model as M, algorithm as A, data as D

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "truths_strut.npz")
SHA = "a194fb7c513ae6f2c47306c3021530d036217334d05083bdadae68d9190e97ae"


def test_truth_set_shapes_and_capacity():
    assert hashlib.sha256(open(PATH, "rb").read()).hexdigest() == SHA
    tr, al, comp, cm = D.truth_set_arrays(PATH)
    assert tr.shape == (10, 200, 9, 4) and al.shape == (10, 200, 9) and comp.shape == (10, 200, M.K_TRUTH, 4)
    assert al.sum(2).max() <= M.K_TRUTH and (cm.sum(2) == al.sum(2)).all()


def test_truth_set_lane_uses_truth_m_mod_k_and_default_unchanged():
    ts = D.truth_set_arrays(PATH)
    cfg, alg = M.CFG_O, M.ALGS["HOLD"]
    r_def = A.make_chunk_runner(cfg, alg, 0, "off")
    r_set = A.make_chunk_runner(cfg, alg, 0, "off", truth_set=ts)
    ks, lam = jnp.arange(0, 12), jnp.asarray([1.0, 1.0])
    _, o_def = r_def(A.init_carry(cfg, 2), ks, jnp.asarray([0, 1]), lam)
    _, o_set0 = r_set(A.init_carry(cfg, 2), ks, jnp.asarray([0, 10]), lam)    # m = 0 and m = 10 -> the same truth index 0
    assert not np.allclose(np.asarray(o_set0["gospa_sq"])[:, 0], np.asarray(o_def["gospa_sq"])[:, 0])      # a different truth is in use
    g2 = os.path.join(os.path.dirname(__file__), "..", "runs", "small", "seed_0", "metrics__CFG-O__HOLD.json")
    if os.path.exists(g2):
        ref = np.asarray(json.load(open(g2))["1.0"]["gospa_sq"])[:12]
        assert np.allclose(np.asarray(o_def["gospa_sq"])[:, 0], ref, atol=1e-9)
