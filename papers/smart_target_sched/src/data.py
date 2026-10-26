"""Scenarios and initial-state sampling.  The paper has no dataset: the only randomness in an experiment is the initial TEC
P_{n,0} (and the tie-break key) because the covariance recursion is deterministic.  This module therefore
holds (a) the scenario registry for Tables I-IV and Figs 5-8 and (b) the PRNG key tree.

Key tree:  root = PRNGKey(top_seed)
             exp  = fold_in(root, EXP_ID[experiment])                       E1..E10 = 1..10, X1..X3 = 11..13
             cell = fold_in(fold_in(exp, scenario_id), K)
             rep_r = fold_in(cell, r)                                       r = 0..Nmc-1  (replication = the seed unit)
               p0_key = fold_in(rep_r, 0)   -> initial TECs
               tie_base = fold_in(rep_r, 1) -> fold_in(., t) -> per-slot tie-break permutation
               mode_base = fold_in(rep_r, 2) -> fold_in(., t) -> X3 sampled modes only
All policies of a cell receive IDENTICAL keys (common random numbers => paired design).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from model import Bank, scalar_bank, matrix_bank, TOP_SEED

EXP_ID = {"E3": 3, "E4": 4, "E5": 5, "E6": 6, "E7": 7, "E8": 8, "E9": 9, "E10": 10, "X1": 11, "X2": 12, "X3": 13, "G1": 20, "G2": 21}


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str                    # 'scalar' | 'matrix'
    types: Tuple[str, ...]       # per target
    qct: Tuple[float, ...]       # Q^CT (scalar) or q^CT (matrix) per target
    d: Tuple[float, ...]
    p0: str                      # 'u02' | 'const001' | 'r0r0'
    sid: int                     # scenario id for the key tree

    @property
    def N(self) -> int:
        return len(self.types)


def build_bank(s: Scenario, ft_variant: str = "printed") -> Bank:
    if s.kind == "scalar":
        return scalar_bank(list(s.types), list(s.qct), list(s.d))
    return matrix_bank(list(s.types), list(s.qct), list(s.d), ft_variant=ft_variant)


# ----------------------------------------------------------------------------------------------- scenario registry
_D1 = (5.0,) + (1.0,) * 7                  # paper §IV-B: d_1 = 5, d_{n != 1} = 1 (n = 1 = first target: A-D1)
_D_MIX = (5.0,) * 4 + (1.0,) * 4           # paper §IV-B: d = 5 reckless, 1 cautious (reckless first: ASSUMED)
_Q2 = (2.0,) * 8                           # paper §IV-B  Q^CT = {2,...,2}
_Q29 = tuple(float(q) for q in range(2, 10))                   # {2,...,9}
_Q2345 = (2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0)              # {2,3,4,5,2,3,4,5}


def _tbl(name, kind, types, qct, d, sid):
    return Scenario(name, kind, tuple(types), tuple(qct), tuple(d), "u02" if kind == "scalar" else "r0r0", sid)


_MIXT = ["reckless"] * 4 + ["cautious"] * 4
TABLE_SCENARIOS = {s.name: s for s in [
    _tbl("t1_q2", "scalar", ["reckless"] * 8, _Q2, _D1, 1),        # Table I
    _tbl("t1_q29", "scalar", ["reckless"] * 8, _Q29, _D1, 2),
    _tbl("t2_q2", "scalar", ["cautious"] * 8, _Q2, _D1, 3),        # Table II
    _tbl("t2_q29", "scalar", ["cautious"] * 8, _Q29, _D1, 4),
    _tbl("t3_q2", "scalar", _MIXT, _Q2, _D_MIX, 5),                # Table III
    _tbl("t3_q2345", "scalar", _MIXT, _Q2345, _D_MIX, 6),
    _tbl("t4_reckless", "matrix", ["reckless"] * 8, _Q29, _D1, 7),  # Table IV
    _tbl("t4_cautious", "matrix", ["cautious"] * 8, _Q29, _D1, 8),
    _tbl("t4_mixed", "matrix", _MIXT, _Q2345, _D_MIX, 9),
]}
TABLE_EXP = {"t1": "E3", "t2": "E4", "t3": "E5", "t4": "E8"}


def figure_scenario(name: str, K: int) -> Scenario:
    """Figs 5-8 at K/N = 1/4 (paper §IV-C; ratio for Figs 7-8 ASSUMED)."""
    N = 4 * K
    h = N // 2
    if name == "fig5a":      # all reckless, Q^CT = {2..N+1}, d = 1, P0 = 0.01
        return Scenario(name, "scalar", ("reckless",) * N, tuple(float(q) for q in range(2, N + 2)), (1.0,) * N, "const001", 10)
    if name == "fig5b":
        return Scenario(name, "scalar", ("cautious",) * N, tuple(float(q) for q in range(2, N + 2)), (1.0,) * N, "const001", 11)
    if name == "fig5c":      # half/half, Q^CT = {2..N/2+1, 2..N/2+1}
        q = [float(x) for x in range(2, h + 2)]
        return Scenario(name, "scalar", ("reckless",) * h + ("cautious",) * h, tuple(q + q), (1.0,) * N, "const001", 12)
    if name == "fig6":       # half reckless d=5, half cautious d=1, Q^CT = 4, P0 ~ U(0,2)
        return Scenario(name, "scalar", ("reckless",) * h + ("cautious",) * h, (4.0,) * N, (5.0,) * h + (1.0,) * h, "u02", 13)
    if name == "fig7":       # matrix, q^CT = {2..N/2+1} per type, d = 1
        q = [float(x) for x in range(2, h + 2)]
        return Scenario(name, "matrix", ("reckless",) * h + ("cautious",) * h, tuple(q + q), (1.0,) * N, "r0r0", 14)
    if name == "fig8":       # matrix, q^CT = 4, d = 5 reckless / 1 cautious
        return Scenario(name, "matrix", ("reckless",) * h + ("cautious",) * h, (4.0,) * N, (5.0,) * h + (1.0,) * h, "r0r0", 15)
    raise KeyError(name)


def toy_scenario() -> Scenario:
    """toy: one reckless + one cautious target, Q^CT = 4, d = 1."""
    return Scenario("toy2", "scalar", ("reckless", "cautious"), (4.0, 4.0), (1.0, 1.0), "u02", 99)


# ----------------------------------------------------------------------------------------------- keys and initial states
def rep_keys(seed: int, exp: str, scen_id: int, K: int, n_rep: int):
    """Per-replication base keys (R,2) from the declared key tree.  `seed` is added to the top-level seed."""
    root = jax.random.PRNGKey(TOP_SEED + seed)
    cell = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(root, EXP_ID[exp]), scen_id), K)
    return jax.vmap(lambda r: jax.random.fold_in(cell, r))(jnp.arange(n_rep))


def split_rep_keys(rk):
    """(p0_key, tie_base, mode_base), each (R,2)."""
    f = lambda i: jax.vmap(lambda k: jax.random.fold_in(k, i))(rk)  # noqa: E731
    return f(0), f(1), f(2)


def sample_P0(s: Scenario, p0_keys) -> jnp.ndarray:
    """Initial TECs (R,N,L,L) for a scenario.

    'u02'      scalar P ~ U(0,2) i.i.d. (paper §IV-B; clipped at 1e-12 to stay in the open interval)
    'const001' P = 0.01 for every target and replication (paper §IV-C, Fig 5)
    'r0r0'     P = R0' R0 with R0 in R^{4x4} i.i.d. U(0,1) (paper §IV-D)"""
    N = s.N
    R = p0_keys.shape[0]
    if s.p0 == "u02":
        u = jax.vmap(lambda k: jax.random.uniform(k, (N,), minval=0.0, maxval=2.0))(p0_keys)
        return jnp.maximum(u, 1e-12)[..., None, None]
    if s.p0 == "const001":
        return jnp.full((R, N, 1, 1), 0.01)
    if s.p0 == "r0r0":
        r0 = jax.vmap(lambda k: jax.random.uniform(k, (N, 4, 4)))(p0_keys)
        P = jnp.swapaxes(r0, -1, -2) @ r0
        return 0.5 * (P + jnp.swapaxes(P, -1, -2))
    raise KeyError(s.p0)


def min_eig(P) -> float:
    """Smallest eigenvalue over all sampled matrix states (A-P0M: logged; the sampler never needed a redraw if > 1e-9)."""
    return float(jnp.min(jnp.linalg.eigvalsh(P)))
