"""Single-target models for the PMBM core.

make_lg_model: the paper's linear/Gaussian point-target model (§V-C, §VII): density = dict(m[nx], P[nx,nx]),
constant p_s, p_d, ellipsoidal gate on the squared Mahalanobis distance (strict <), moment-matched merge.
This file is deliberately small: the paper has no network or dataset (see data.py / train.py notes).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithm import PMBMConfig, PMBMModel



class LGParams(NamedTuple):
    F: np.ndarray           # paper §VII
    Q: np.ndarray           # paper §VII (q = 0.01; official code 0.02 -> sensitivity row)
    H: np.ndarray           # paper §VII
    R: np.ndarray           # paper §VII
    ps: float               # paper §VII
    pd: float               # paper §VII / Table II
    clutter_density: float  # lambda_c / area (paper §VII: uniform on [0,300]^2)
    gate: float             # paper §VII (squared Mahalanobis)
    birth_w: np.ndarray     # [B] linear weights, paper §VII (0.005)
    birth_m: np.ndarray     # [B,nx]
    birth_P: np.ndarray     # [B,nx,nx]


def paper_lg_params(pd=0.9, lambda_c=10.0, q=0.01, T=1.0, birth_w=0.005, gate=20.0, area=90000.0) -> LGParams:
    """Paper §VII scenario parameters (sources.yaml 'scenario' and 'pmbm' keys)."""
    I2 = np.eye(2)
    F = np.kron(I2, np.array([[1, T], [0, 1]]))                                  # paper §VII
    Q = q * np.kron(I2, np.array([[T ** 3 / 3, T ** 2 / 2], [T ** 2 / 2, T]]))   # paper §VII
    Hm = np.kron(I2, np.array([[1.0, 0.0]]))                                     # paper §VII
    return LGParams(F, Q, Hm, np.eye(2), 0.99, pd, lambda_c / area, gate,   # area = |[0,300]^2| (paper §VII); train.py passes cfg['area']
                    np.array([birth_w]), np.array([[100.0, 0.0, 100.0, 0.0]]),
                    np.array([np.diag([150.0 ** 2, 1.0, 150.0 ** 2, 1.0])]))


# Paper-path capacities.  PMBMConfig's own defaults (T = 128, H = 64) are generic sizes and are NOT the
# paper filter: the paper keeps N_h = 200 global hypotheses and no per-track cap, so H = N_h; T covers the live tracks plus one new
# track per measurement; M = 48 makes clutter truncation ~1e-7 per step at lambda_c = 20.  ASSUMED array sizes; counters must read 0.
PAPER_T, PAPER_H, PAPER_M, PAPER_P = 256, 200, 48, 8


def paper_pmbm_config(**over) -> PMBMConfig:
    """PMBMConfig of the paper's Section VII filter (N_h = 200, gate/Poisson/Bernoulli prunes at their PMBMConfig defaults = paper values,
    no per-track cap, and NO global-hypothesis weight threshold: Alg. 1 keeps the highest N_h global hypotheses, Sec. VII lists no other
    threshold).  The 1e-4 global-weight floor of the authors' current code (T_pruning, introduced 2019-05-08, commit 7f2d1b8; the first
    release 4950e45 has 0) is a labelled sensitivity row: pass prune_w=1e-4."""
    kw = dict(T=PAPER_T, H=PAPER_H, Nh=200, P=PAPER_P, M=PAPER_M, prune_w=0.0)
    kw.update(over)
    return PMBMConfig(**kw)


def make_lg_model(p: LGParams) -> PMBMModel:
    F, Q, H, R = (jnp.asarray(a, jnp.float64) for a in (p.F, p.Q, p.H, p.R))
    nx, nz = F.shape[0], H.shape[0]
    ps, pd = float(p.ps), float(p.pd)
    bw = jnp.log(jnp.asarray(p.birth_w, jnp.float64))
    bm = jnp.asarray(p.birth_m, jnp.float64)
    bP = jnp.asarray(p.birth_P, jnp.float64)
    logc = float(np.log(p.clutter_density))
    gate = float(p.gate)

    def predict(d, ctx):
        return dict(m=F @ d["m"], P=F @ d["P"] @ F.T + Q)

    def survive(d, ctx):
        return jnp.asarray(ps, jnp.float64)

    def det(d, z, ctx):
        m, P = d["m"], d["P"]
        S = H @ P @ H.T + R
        zp = H @ m
        innov = z - zp
        L = jnp.linalg.cholesky(S)
        y = jax.scipy.linalg.solve_triangular(L, innov, lower=True)
        maha = jnp.sum(y * y)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        logm = jnp.log(pd) - 0.5 * maha - 0.5 * logdet - 0.5 * nz * jnp.log(2 * jnp.pi)
        K = P @ H.T @ jax.scipy.linalg.cho_solve((L, True), jnp.eye(nz))
        Pu = (jnp.eye(nx) - K @ H) @ P
        Pu = 0.5 * (Pu + Pu.T)
        return logm, dict(m=m + K @ innov, P=Pu), maha < gate

    def miss(d, ctx):
        return jnp.asarray(1.0 - pd, jnp.float64), d

    def merge(ds, logw):
        w = jax.nn.softmax(logw)
        m = jnp.sum(w[:, None] * ds["m"], axis=0)
        dm = ds["m"] - m
        P = jnp.sum(w[:, None, None] * (ds["P"] + dm[:, :, None] * dm[:, None, :]), axis=0)
        return dict(m=m, P=0.5 * (P + P.T))

    def birth(ctx):
        return bw, dict(m=bm, P=bP)

    def log_clutter(z, ctx):
        return jnp.asarray(logc, jnp.float64)

    def point(d):
        return d["m"]

    def init_density():
        return dict(m=jnp.zeros(nx), P=jnp.eye(nx))

    return PMBMModel(predict, survive, det, miss, merge, birth, log_clutter, point, init_density)
