"""Exogenous input processes (the ``z`` of an input-driven MDP) and the PRNG key tree.

Every input sequence is a pure function of a PRNG key, generated *before* the rollout and
passed to the simulator as an explicit array. Re-using the same key therefore replays the
identical input sequence under any action sequence -- the input-repeatability that the
multi-value and meta baselines rely on (paper §5).

PRNG key tree (one top-level seed per training run):

    root = PRNGKey(seed)
    ├─ k_init -> split into actor / critic(s) parameter init
    ├─ k_inputs_train -> fold_in(iteration) -> fresh training input sequences
    ├─ k_inputs_fixedN -> split(N) -> the N fixed sequences of the multi-value arm
    ├─ k_rollout -> fold_in(iteration) -> split(n_envs) -> per-env action noise
    ├─ k_sched -> fold_in(iteration) -> which fixed index / which rollouts (multi-value)
    └─ k_eval -> fold_in(checkpoint) -> action noise during evaluation
    k_inputs_test = PRNGKey(TEST_SEED) # NOT derived from seed: fixed across seeds and arms
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

TEST_SEED = 100_000_000  # official code seeds test episodes from 1e8; disjoint from seeds 0..99
PROBE_SEED = 200_000_000  # gradient-variance probes (small-scale) -- disjoint from train and test
VAL_SEED = 300_000_000  # validation inputs for hyper-parameter selection (full-scale tuning) -- never the test set


class KeyTree(NamedTuple):
    init: jax.Array
    inputs_train: jax.Array
    inputs_fixedN: jax.Array
    rollout: jax.Array
    sched: jax.Array
    eval: jax.Array


def key_tree(seed: int) -> KeyTree:
    root = jax.random.PRNGKey(seed)
    return KeyTree(*jax.random.split(root, 6))


def test_key() -> jax.Array:
    return jax.random.PRNGKey(TEST_SEED)


def val_key() -> jax.Array:
    return jax.random.PRNGKey(VAL_SEED)


# ---------------------------------------------------------------- load balancing input
class LBInputs(NamedTuple):
    """One episode's job stream: integer arrival times and integer sizes, both shape [n_jobs]."""

    arrivals: jax.Array  # int32, non-decreasing
    sizes: jax.Array  # int32 >= pareto_scale


def sample_lb_inputs(key, n_jobs: int, mean_interarrival: float, pareto_shape: float,
                     pareto_scale: float) -> LBInputs:
    """size = floor(Pareto(shape) * scale)  (x_m = scale; paper App. J, official int()),
    inter-arrival = floor(Exp(1) * mean)  (official int(np.random.exponential)).
    The first job arrives at the first inter-arrival time (official generate_jobs)."""
    k_size, k_gap = jax.random.split(key)
    u = jax.random.uniform(k_size, (n_jobs,), minval=1e-7, maxval=1.0)  # avoid u=0 -> inf size
    sizes = jnp.floor(pareto_scale * u ** (-1.0 / pareto_shape))
    # The effective truncation is the uniform floor, not the 2e7 cap: u >= 1e-7 gives size <= 100 * (1e-7)^(-1/1.5)
    # ~= 4.64e6 at the default shape/scale (P(size > 4.64e6) = 1e-7 per job is cut off; float32 uniforms are also
    # quantised near 0 in steps of ~2^-23).  The 2e7 cap below never binds at shape 1.5; it is kept as an int32
    # guard (size * 100 must fit) for other shapes.
    sizes = jnp.minimum(sizes, 2.0e7).astype(jnp.int32)
    gaps = jnp.floor(jax.random.exponential(k_gap, (n_jobs,)) * mean_interarrival).astype(jnp.int32)
    arrivals = jnp.cumsum(gaps).astype(jnp.int32)
    return LBInputs(arrivals, sizes)


def sample_lb_batch(key, n: int, **kw) -> LBInputs:
    keys = jax.random.split(key, n)
    return jax.vmap(lambda k: sample_lb_inputs(k, **kw))(keys)


def lb_input_kwargs(cfg: dict) -> dict:
    return dict(n_jobs=int(cfg["jobs_per_episode"]), mean_interarrival=float(cfg["mean_interarrival"]),
                pareto_shape=float(cfg["pareto_shape"]), pareto_scale=float(cfg["pareto_scale"]))


# ---------------------------------------------------------------- 1-D toy input (App. A)
def sample_toy_inputs(key, n: int, T: int, sigma: float) -> jax.Array:
    """z_t i.i.d. uniform on {-sigma, +sigma}; shape [n, T] (paper App. A has sigma = 1)."""
    return sigma * (2.0 * jax.random.bernoulli(key, 0.5, (n, T)).astype(jnp.float32) - 1.0)


# ---------------------------------------------------------------- fingerprints
def seq_hashes(inputs: LBInputs) -> list[str]:
    arr = np.asarray(inputs.arrivals)
    siz = np.asarray(inputs.sizes)
    return [hashlib.sha256(arr[i].tobytes() + siz[i].tobytes()).hexdigest() for i in range(arr.shape[0])]


def fingerprint(inputs: LBInputs, n: int = 3) -> str:
    h = hashlib.sha256()
    for s in seq_hashes(inputs)[:n]:
        h.update(s.encode())
    return h.hexdigest()[:16]


def full_sha(inputs: LBInputs) -> str:
    h = hashlib.sha256()
    for s in seq_hashes(inputs):
        h.update(s.encode())
    return h.hexdigest()
