"""Matmul-precision fix / run control: the matmul-precision key and run_offset keep stored runs' config hashes and keys."""
import json
import os

import numpy as np

import _helpers  # noqa: F401  (import path, CPU)
import jax
import train


def test_prefix_config_hash_unchanged_for_stored_runs():
    # hashes written by the original training runs, which used an earlier run-folder tag (part of the hash)
    hashes = json.load(open(os.path.join(train.REPO, "tests", "fixtures", "stored_config_hashes.json")))
    for yml, m in (("full_heavenhell4.yaml", "a2c_asym_hs"), ("small_shopping5.yaml", "a2c_react_4")):
        c = train.resolve_config(os.path.join(train.REPO, "src", "configs", yml), m)
        assert c["matmul_precision"] == "highest"            # full-f32 default
        rec = hashes[f"{yml}/{m}"]
        c["tag"] = rec["original_tag"]
        stored = rec["sha256"]
        assert train.config_hash(c) != stored                  # new default = a different config
        c["matmul_precision"] = None                           # pre-fix reproduction
        assert train.config_hash(c) == stored


def test_run_offset_gives_the_same_per_run_keys():
    y = os.path.join(train.REPO, "src", "configs", "full_heavenhell4.yaml")
    r4, _ = train.build(train.resolve_config(y, "a2c_asym_s", dict(runs=4)))
    r2, _ = train.build(train.resolve_config(y, "a2c_asym_s", dict(runs=2, run_offset=2)))
    for a, b in zip(jax.tree.leaves((r4.pa, r4.pc, r4.key)), jax.tree.leaves((r2.pa, r2.pc, r2.key))):
        assert np.array_equal(np.asarray(a)[2:4], np.asarray(b))
