"""Shared, framework-free recipe for the compact official-parity fixtures (imported by the official-venv exporter AND by
tests/test_official_parity.py, so both sides build bit-identical weights and projection directions from names alone).

Weights: every tensor is drawn from numpy's PCG64 seeded by crc32("<tag>/<canonical name>") -- no torch RNG involved, so
our side regenerates them without the oracle. Aliased tensors (the official HistoryIntegrator re-registers the action /
observation embeddings under history_model.*) map to one canonical name. The reset-observation row (padding_idx=-1) is 0.
Fingerprints: per-tensor L2 norm, K unit-vector projections, and the full values for tensors with <= SMALL elements."""
import zlib

import numpy as np

K, SMALL = 4, 2048


def canonical(name):
    return name.replace("history_model.action_model.", "action_model.").replace(
        "history_model.observation_model.", "observation_model.")


def gen_tensor(tag, name, shape):
    rng = np.random.default_rng(zlib.crc32(f"{tag}/{canonical(name)}".encode()))
    x = rng.standard_normal(shape)
    if name.endswith("embeddings.weight"):
        x = x * 1.0                                     # torch default N(0,1) embeddings
    elif len(shape) == 2:
        x = x / np.sqrt(shape[1])                       # fan-in scaled
    else:
        x = 0.1 * x                                     # biases
    if "observation_model.embeddings.weight" in name:
        x[-1] = 0.0                                     # padding_idx=-1 row stays zero
    return x                                            # float64; callers cast


def gen_weights(tag, shapes):
    return {k: gen_tensor(tag, k, tuple(s)) for k, s in shapes.items()}


def directions(name, n):
    rng = np.random.default_rng(zlib.crc32(f"proj/{canonical(name)}".encode()))
    u = rng.standard_normal((K, n))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def fingerprint(name, x):
    """x: float64 array (an official gradient or update). Returns dict of small arrays."""
    v = np.asarray(x, np.float64).ravel()
    out = {"norm": np.float64(np.linalg.norm(v)), "proj": directions(name, v.size) @ v, "n": np.int64(v.size)}
    if v.size <= SMALL:
        out["full"] = v
    return out
