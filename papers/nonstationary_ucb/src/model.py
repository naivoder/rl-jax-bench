"""Policy state structs (the "model" of a bandit policy is its sufficient statistics; there is no network).

Every struct holds the statistics of ONE replication; the runner vmaps over replications. Shapes use K =
number of arms and L = the static ring-buffer length of SW-UCB (L >= tau).

hand derivation tags: D-2/D-3 (discounted sums), D-7 (window sums), Auer et al. 2002 Fig. 6 (EXP3.S weights).
"""
from __future__ import annotations

from typing import NamedTuple

import jax


class CountStats(NamedTuple):
    """UCB-1 (undiscounted) and D-UCB (discounted) statistics. Counts are float64 (exact below 2**53)."""
    n: jax.Array  # (K,) N_t(i) or N_t(gamma, i)
    s: jax.Array  # (K,) S_t(i) or S_t(gamma, i)


class RestartStats(NamedTuple):
    """ORACLE-RESTART-UCB1 (labelled EXTENSION reference): UCB-1 statistics plus the last reset round."""
    n: jax.Array   # (K,)
    s: jax.Array   # (K,)
    t0: jax.Array  # () int32: rounds 1..t0 belong to the previous stationary segment


class WindowStats(NamedTuple):
    """SW-UCB statistics: per-arm window counts/sums and a ring buffer of the last tau (action, reward)."""
    n: jax.Array      # (K,) int32 N_t(tau, i)
    s: jax.Array      # (K,) float64 S_t(tau, i)
    buf_a: jax.Array  # (L,) int32 actions, slot (t-1) mod tau holds round t's play
    buf_x: jax.Array  # (L,) float64 rewards


class Exp3SStats(NamedTuple):
    """EXP3.S log-weights, max-normalised after each update."""
    logw: jax.Array  # (K,)


class Params(NamedTuple):
    """Dynamic (traced) hyperparameters, so sweeps over gamma/tau/xi reuse one compilation."""
    B: jax.Array      # reward bound (paper §5: 1 for Bernoulli)
    xi: jax.Array     # exploration constant (paper §5: 1/2)
    gamma: jax.Array  # D-UCB discount
    tau: jax.Array    # SW-UCB window (int32, <= static buffer length)
    g: jax.Array      # EXP3.S exploration rate gamma_Auer (renamed g to avoid clashing with D-UCB gamma)
    alpha: jax.Array  # EXP3.S weight-sharing rate
    eta_mult: jax.Array  # EXP3.S learning-rate multiplier: eta = eta_mult * g / K (1 = Auer's coupling, the paper arm;
                         # != 1 only for the labelled recheck EXTENSION arm "EXP3.S decoupled-eta", README RC3)


class Acc(NamedTuple):
    """Per-replication accumulators carried through the scan."""
    regret: jax.Array  # () float64 cumulative pseudo-regret
    arm1: jax.Array    # () int32 cumulative plays of arm 1 (index 0)
    ntilde: jax.Array  # (K,) int32 plays of arm i while suboptimal (N~_T(i))
    usum: jax.Array    # () float64 sum of all environment uniforms seen (CRN fingerprint, identical across policies)
    bad: jax.Array     # () bool any NaN index / non-finite EXP3.S weight seen
