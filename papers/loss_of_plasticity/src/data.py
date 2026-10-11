"""Data streams: Slowly-Changing Regression (SCR, paper App. B) and Online Permuted MNIST (paper §3).

SCR -- official-code definition (lop/nets/fix_ltu_net.py + slowly_changing_regression.py; ):
  input x in {0,1}^20 = [15 flipping bits | 5 i.i.d. Bernoulli(1/2) bits]; target LTU net 20 -> 100 -> 1 with every
  weight and bias uniform in {-1,+1}; LTU j fires iff W_j x + b_j > tau_j, tau_j = beta (n_in + 1) - S_j,
  S_j = (n_in - sum_k W_jk + b_j) / 2, beta = 0.7; y = w2 . features + b2.
  'flip_one' (Fig. B.10 / D.12): every T = 10 000 examples one uniformly chosen flipping bit is toggled.
  'resample' (official cfg/prob.json, used by the L2/S&P/CBP SCR configs): all 15 flipping bits re-drawn every T.
  Paper-text variant (``paper_text=True``, Table B.3 literally): 22 inputs = 15 flip + 6 random + constant 1, no LTU
  bias, tau = beta * 22 - #neg.

PRNG tree of one SCR stream (k_data = fold_in(root, 0), k_target = fold_in(root, 1)):
  bits_0           = bernoulli(fold_in(k_data, 0), 15)
  flip / resample  : fold_in(fold_in(k_data, 1), p) for period p >= 1
  random bits      : bernoulli(fold_in(fold_in(k_data, 2), p), (T, n_rand)) for period p
MNIST: per task t, (k_perm, k_order) = split(fold_in(k_data, t)); pixel permutation and example order.
"""
from __future__ import annotations

import gzip
import hashlib
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

MNIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mnist")
MNIST_MD5 = {"train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
             "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432"}


# --------------------------------------------------------------------------------------------- SCR
class SCRSpec(NamedTuple):
    n_flip: int = 15          # Table B.3 f = 15
    n_rand: int = 5           # official code: 20 inputs - 15 flipping (paper text: 6)
    n_target: int = 100       # Table B.3 n = 100
    beta: float = 0.7         # Table B.3
    T: int = 10_000           # Table B.3 flip period
    mode: str = "flip_one"    # 'flip_one' | 'resample'
    paper_text: bool = False  # 22-bit paper-text variant (constant input bit, no LTU bias)

    @property
    def n_in(self) -> int:
        return self.n_flip + self.n_rand + (1 if self.paper_text else 0)


def scr_spec(mode="flip_one", paper_text=False) -> SCRSpec:
    return SCRSpec(n_rand=6 if paper_text else 5, mode=mode, paper_text=paper_text)


def scr_target(k_target, spec: SCRSpec):
    """Target LTU network, torch layout: W (n_target, n_in), b (n_target,), w2 (n_target,), b2 (), tau (n_target,)."""
    k1, k2, k3, k4 = jax.random.split(k_target, 4)
    pm = lambda k, s: jax.random.randint(k, s, 0, 2).astype(jnp.float32) * 2 - 1
    W = pm(k1, (spec.n_target, spec.n_in))
    b = pm(k2, (spec.n_target,))
    w2 = pm(k3, (spec.n_target,))
    b2 = pm(k4, ())
    if spec.paper_text:
        b = jnp.zeros_like(b)
        S = jnp.sum(W < 0, axis=1).astype(jnp.float32)
        tau = spec.beta * spec.n_in - S                                   # paper App. B: (m+1) beta - S_i
    else:
        S = (spec.n_in - jnp.sum(W, axis=1) + b) / 2                      # fix_ltu_net.py
        tau = spec.beta * (spec.n_in + 1) - S
    return {"W": W, "b": b, "w2": w2, "b2": b2, "tau": tau}


def scr_target_apply(tgt, X):
    feats = (X @ tgt["W"].T + tgt["b"] > tgt["tau"]).astype(jnp.float32)
    return feats @ tgt["w2"] + tgt["b2"], feats


def scr_bits0(k_data, spec: SCRSpec):
    return jax.random.bernoulli(jax.random.fold_in(k_data, 0), 0.5, (spec.n_flip,)).astype(jnp.float32)


def scr_next_bits(k_data, p, bits, spec: SCRSpec):
    """Flipping bits of period p >= 1 given those of period p - 1."""
    k = jax.random.fold_in(jax.random.fold_in(k_data, 1), p)
    if spec.mode == "flip_one":
        c = jax.random.randint(k, (), 0, spec.n_flip)
        return bits.at[c].set(1.0 - bits[c])
    return jax.random.bernoulli(k, 0.5, (spec.n_flip,)).astype(jnp.float32)


def scr_period_inputs(k_data, p, bits, spec: SCRSpec):
    """X (T, n_in) of period p given that period's flipping bits."""
    r = jax.random.bernoulli(jax.random.fold_in(jax.random.fold_in(k_data, 2), p), 0.5,
                             (spec.T, spec.n_rand)).astype(jnp.float32)
    parts = [jnp.broadcast_to(bits, (spec.T, spec.n_flip)), r]
    if spec.paper_text:
        parts.append(jnp.ones((spec.T, 1), jnp.float32))
    return jnp.concatenate(parts, axis=1)


def scr_stream(k_data, k_target, spec: SCRSpec, n_periods: int):
    """Materialise (X, Y) for the first n_periods periods (tests / oracle export only; training generates on the fly)."""
    tgt = scr_target(k_target, spec)
    bits = scr_bits0(k_data, spec)
    Xs = []
    for p in range(n_periods):
        if p > 0:
            bits = scr_next_bits(k_data, p, bits, spec)
        Xs.append(scr_period_inputs(k_data, p, bits, spec))
    X = jnp.concatenate(Xs)
    Y, _ = scr_target_apply(tgt, X)
    return X, Y, tgt


# --------------------------------------------------------------------------------------------- MNIST
def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_mnist(directory: str = MNIST_DIR):
    """Canonical MNIST training set -> (x uint8 (60000, 784), y uint8 (60000,), checksums dict)."""
    out, sums = {}, {}
    for name, md5 in MNIST_MD5.items():
        path = os.path.join(directory, name)
        got = _md5(path)
        if got != md5:
            raise RuntimeError(f"MNIST checksum mismatch for {path}: {got} != {md5}")
        with gzip.open(path, "rb") as f:
            raw = f.read()
        if "images" in name:
            arr = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784)
        else:
            arr = np.frombuffer(raw, np.uint8, offset=8)
        sums[name] = {"md5_gz": got, "sha256_raw": hashlib.sha256(arr.tobytes()).hexdigest()}
        out["x" if "images" in name else "y"] = arr
    return out["x"], out["y"], sums


def mnist_task_keys(k_data, task):
    return jax.random.split(jax.random.fold_in(k_data, task))


def mnist_task(k_data, task, n_examples: int = 60_000, n_pix: int = 784):
    """(pixel permutation (784,), example order (n_examples,)) of task ``task`` (fresh uniform draws, paper §3)."""
    kp, ko = mnist_task_keys(k_data, task)
    return jax.random.permutation(kp, n_pix), jax.random.permutation(ko, n_examples)


def mnist_example(x_u8, y_u8, perm, order, t):
    """Example t of a task: pixels / 255 (official ToTensor scaling), permuted."""
    i = order[t]
    x = x_u8[i][perm].astype(jnp.float32) / 255.0
    return x[None, :], y_u8[i][None].astype(jnp.int32)
