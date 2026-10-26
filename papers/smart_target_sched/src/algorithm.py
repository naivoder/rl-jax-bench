"""Scheduling algorithms of Hao et al. (arXiv:2312.07858v1) and the comparators of this reproduction, as pure jitted functions.

Arms (every arm differs ONLY in how the K targets to illuminate are chosen; the environment step, cost, tie-break
and bookkeeping are shared):
  whittle       Algorithm 1: MP index mp*(P) = f(P,z)/g(P,z) at z = s(P) (eqs 26-32), two tau-step deterministic rollouts
  myopic        eq (34): d [s(phi^0(P)) - s(phi^1(P))]
  tev           eq (33): d s(P)   (TEV in the scalar case, TEC in the matrix case)
  passive_cont  OURS: F(P,<0,inf>) - F(P,<1,inf>) = tau-step cost saved by tracking now and never again
  passive_cont_h10  OURS: same with a 10-slot horizon (any passive_cont_h<N> parses; a shorter-memory variant of the no-search heuristic)
  passive_cont_revisit  OURS: passive_cont with horizon ceil(N/K) (the mean revisit interval of a K-of-N round robin)
  random        OURS: uniformly random K-subset (floor)
  rollout1      OURS (search, labelled): one-step lookahead over all C(N,K) joint actions + myopic base policy, 30 slots
  whittle_blind0 / whittle_blind1 / myopic_blind0 / myopic_blind1   OURS (ablation): the index is computed as if the
                target ignored the illumination (both mode-probability rows = u^0, resp. u^1), the environment keeps the real reaction
Also: exact_enumerate (EXACT optimum on tiny cases) and the sampled-mode environment of extension X3.

Tie-break: a random permutation drawn from the per-slot key is applied before a stable argsort, so exact ties are
resolved uniformly at random and reproducibly.  'skip_neg' switches on the §III-C variant (never track a negative index).
"""
from __future__ import annotations

import functools
import itertools
import re
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rl_jax_bench import jaxlog  # noqa: E402

from model import (BETA, TAU, G_EPS, Bank, phi, phi_sampled, stat, step_cost, cost_vec)  # noqa: E402

ROLLOUT1_BASE_SLOTS = 30  # ASSUMED
BIG_INDEX = 1e30          # finite stand-in for +inf in the PCLI1-violation rule (keeps the NaN/Inf guard meaningful)
PASSIVE_H_SHORT = 10      # ASSUMED

INDEX_POLICIES = ("whittle", "myopic", "tev", "passive_cont", "passive_cont_h10", "passive_cont_revisit", "random")
ALL_POLICIES = INDEX_POLICIES + ("rollout1",)


class Cfg(NamedTuple):
    """Static (hashable) configuration of a run."""
    T: int = 100
    beta: float = BETA
    tau: int = TAU
    skip_neg: bool = False       # §III-C variant
    sampled: bool = False        # X3: sampled modes instead of the mode-averaged covariance
    guard: bool = True           # NaN/Inf guard (jax.debug.callback)
    tie_first: bool = False      # deterministic lowest-index tie-break (toy only); default = uniform random
    pcli1: str = "dominant"      # g <= 0 (PCLI1 violated): 'dominant' = index +/-BIG by the sign of f; 'literal' = f / g as printed
    r1_base: int = ROLLOUT1_BASE_SLOTS   # rollout1 base-policy length (tuning-parity sweep; default = 30)


# ----------------------------------------------------------------------------------------------- rollouts (F, G of eqs 24-25 / 29-30)
def rollout_FG(bank: Bank, P, z, a0, tau: int, beta: float):
    """F, G of the policy <a0, z>: action a0 at k = 0, then 'active iff s(P_k) > z' for k = 1..tau-1 (eqs 24-25, 29-30).

    P (N,L,L), z (N,), a0 (N,) float 0/1.  Deterministic (D1).  Returns F (N,), G (N,)."""
    zero = jnp.zeros(P.shape[0])

    def body(carry, k):
        Pk, F, G, disc = carry
        act = jnp.where(k == 0, a0, (stat(Pk) > z).astype(Pk.dtype))
        F = F + disc * cost_vec(bank, Pk, act)
        G = G + disc * act
        return (phi(bank, Pk, act), F, G, disc * beta), None

    (_, F, G, _), _ = jax.lax.scan(body, (P, zero, zero, jnp.ones(())), jnp.arange(tau))
    return F, G


def fg_marginal(bank: Bank, P, z, tau: int = TAU, beta: float = BETA):
    """f(P,z) = F(P,<0,z>) - F(P,<1,z>), g(P,z) = G(P,<1,z>) - G(P,<0,z>)  (eqs 26, 31)."""
    n = P.shape[0]
    F0, G0 = rollout_FG(bank, P, z, jnp.zeros(n), tau, beta)
    F1, G1 = rollout_FG(bank, P, z, jnp.ones(n), tau, beta)
    return F0 - F1, G1 - G0


# ----------------------------------------------------------------------------------------------- indices
def index_whittle(bank: Bank, P, cfg: Cfg):
    """mp*(P) = mp(P, s(P)) = f/g (eqs 27, 32).  Returns (index (N,), number of PCLI1 violations g <= G_EPS)."""
    f, g = fg_marginal(bank, P, stat(P), cfg.tau, cfg.beta)
    bad = g <= G_EPS                                   # PCLI1 violated at this (P,z): the ratio f/g is not a marginal productivity
    if cfg.pcli1 == "literal":                         # Algorithm 1 as printed: the ratio itself (negative for g < 0; |g| ~ 0 -> +-BIG)
        return f / jnp.where(jnp.abs(g) > G_EPS, g, G_EPS), jnp.sum(bad)
    # tracking now saves cost (f > 0) and adds no work (g <= 0): infinite productivity -> top priority
    return jnp.where(bad, jnp.where(f > 0, BIG_INDEX, -BIG_INDEX), f / jnp.where(bad, 1.0, g)), jnp.sum(bad)


def index_myopic(bank: Bank, P, cfg: Cfg | None = None):
    """Eq (34)."""
    n = P.shape[0]
    return bank.d * (stat(phi(bank, P, jnp.zeros(n))) - stat(phi(bank, P, jnp.ones(n)))), jnp.zeros((), jnp.int32)


def index_tev(bank: Bank, P, cfg: Cfg | None = None):
    """Eq (33)."""
    return bank.d * stat(P), jnp.zeros((), jnp.int32)


def index_passive(bank: Bank, P, cfg: Cfg, tau: int | None = None):
    """F(P,<0,inf>) - F(P,<1,inf>): threshold +inf => never active after k = 0 (G1 - G0 = 1, so this is f/g)."""
    n = P.shape[0]
    tau = cfg.tau if tau is None else tau
    inf = jnp.full((n,), jnp.inf)
    F0, _ = rollout_FG(bank, P, inf, jnp.zeros(n), tau, cfg.beta)
    F1, _ = rollout_FG(bank, P, inf, jnp.ones(n), tau, cfg.beta)
    return F0 - F1, jnp.zeros((), jnp.int32)


def _index_fn(name: str):
    if name == "whittle":
        return index_whittle
    if name == "myopic":
        return index_myopic
    if name == "tev":
        return index_tev
    if name == "passive_cont":
        return index_passive
    if name == "passive_cont_revisit":     # horizon = ceil(N/K): the mean revisit interval of a K-of-N round robin
        return _passive_revisit
    m = re.fullmatch(r"passive_cont_h(\d+)", name)
    if m:                                   # horizon-limited variant, e.g. passive_cont_h10
        return functools.partial(_passive_h, tau=int(m.group(1)))
    raise KeyError(name)


def _passive_revisit(bank, P, cfg):
    raise RuntimeError("resolved in _decide (needs K)")


def _passive_h(bank, P, cfg, tau):
    return index_passive(bank, P, cfg, tau)


# ----------------------------------------------------------------------------------------------- selection
def select_topk(idx, key, K: int, skip_neg: bool = False, tie_first: bool = False):
    """Top-K by index with uniform random tie-break (tie_first = lowest index wins).  Returns a float 0/1 mask (N,)."""
    n = idx.shape[0]
    perm = jnp.arange(n) if tie_first else jax.random.permutation(key, n)
    order = jnp.argsort(-idx[perm], stable=True)[:K]
    chosen = perm[order]
    ok = (idx[chosen] >= 0) if skip_neg else jnp.ones(K, bool)
    return jnp.zeros(n).at[chosen].set(ok.astype(idx.dtype))


def select_topk_det(idx, K: int):
    """Deterministic top-K (lowest index first on ties): used by the base policy inside rollout1."""
    order = jnp.argsort(-idx, stable=True)[:K]
    return jnp.zeros(idx.shape[0]).at[order].set(1.0)


@functools.lru_cache(maxsize=None)
def combos_mask(n: int, K: int) -> np.ndarray:
    """All C(n,K) joint actions as a (C, n) 0/1 matrix (static)."""
    c = list(itertools.combinations(range(n), K))
    m = np.zeros((len(c), n))
    for i, cc in enumerate(c):
        m[i, list(cc)] = 1.0
    return m


# ----------------------------------------------------------------------------------------------- the search reference
def decide_rollout1(bank: Bank, P, key, K: int, cfg: Cfg):
    """ROLLOUT-1: for every joint K-subset a, apply a now, then follow the myopic base policy for
    ROLLOUT1_BASE_SLOTS slots; choose the subset with the least discounted cost of the following slots."""
    cand = jnp.asarray(combos_mask(P.shape[0], K))                      # (C, N)

    def base_value(mask):
        def body(carry, k):
            Pk, V, disc = carry
            idx, _ = index_myopic(bank, Pk)
            a = select_topk_det(idx, K)
            V = V + disc * step_cost(bank, Pk, a)                       # cost of slot t+1+k
            return (phi(bank, Pk, a), V, disc * cfg.beta), None
        (_, V, _), _ = jax.lax.scan(body, (phi(bank, P, mask), 0.0, 1.0), jnp.arange(cfg.r1_base))
        return V

    V = jax.vmap(base_value)(cand)
    perm = jax.random.permutation(key, cand.shape[0])
    best = perm[jnp.argmin(V[perm])]
    return cand[best]


# ----------------------------------------------------------------------------------------------- closed-loop simulation
def _decide(name: str, bank: Bank, P, key, K: int, cfg: Cfg):
    """One slot's decision for one replication: returns (mask (N,), PCLI1-violation count, index (N,) for the NaN guard)."""
    zero = jnp.zeros((), jnp.int32)
    m = re.fullmatch(r"(whittle|myopic)_blind([01])", name)
    if m:                                   # ablation: the INDEX is computed from a bank that ignores the reaction, the environment is unchanged
        base, row = m.group(1), int(m.group(2))
        ib = bank._replace(U=jnp.stack([bank.U[:, row], bank.U[:, row]], axis=1))
        idx, nbad = (index_whittle if base == "whittle" else index_myopic)(ib, P, cfg)
        return select_topk(idx, key, K, cfg.skip_neg, cfg.tie_first), nbad, idx
    if name == "rollout1":
        return decide_rollout1(bank, P, key, K, cfg), zero, jnp.zeros(P.shape[0])
    if name == "random":
        idx = jax.random.uniform(jax.random.fold_in(key, 7), (P.shape[0],))
        return select_topk(idx, key, K, False), zero, idx
    if name == "passive_cont_revisit":
        idx, nbad = index_passive(bank, P, cfg, tau=-(-P.shape[0] // K))
    else:
        idx, nbad = _index_fn(name)(bank, P, cfg)
    return select_topk(idx, key, K, cfg.skip_neg, cfg.tie_first), nbad, idx


@functools.partial(jax.jit, static_argnames=("name", "K", "cfg"))
def simulate(bank: Bank, P0, tie_base, mode_base, name: str, K: int, cfg: Cfg):
    """Closed-loop run of policy `name`, vmapped over replications.

    P0 (R,N,L,L); tie_base, mode_base: (R,2) PRNG keys (one per replication; the paired design reuses them for every policy).
    Returns dict: J (R,) discounted cost sum_{t<T} beta^t cost_t (cost on the start-of-slot state), n_pcli1_viol (R,),
    pmax (R,) = max_t max_n s(P_{n,t}), acts (T,R,N) int8."""
    R = P0.shape[0]

    def body(carry, t):
        P, J, nbad, pmax = carry
        keys = jax.vmap(jax.random.fold_in, (0, None))(tie_base, t)
        mask, nb, idx = jax.vmap(lambda p, k: _decide(name, bank, p, k, K, cfg))(P, keys)
        if cfg.guard:
            jaxlog.nan_guard(idx, (idx, P), tag=1)                          # any non-finite index aborts the run
        c = jax.vmap(lambda p, a: step_cost(bank, p, a))(P, mask)
        if cfg.sampled:
            mk = jax.vmap(jax.random.fold_in, (0, None))(mode_base, t)
            Pn = jax.vmap(lambda p, a, k: phi_sampled(bank, p, a, k))(P, mask, mk)
        else:
            Pn = jax.vmap(lambda p, a: phi(bank, p, a))(P, mask)
        pm = jnp.maximum(pmax, jnp.max(stat(P), axis=-1))
        return (Pn, J + cfg.beta ** t * c, nbad + nb.astype(nbad.dtype), pm), mask.astype(jnp.int8)

    init = (P0, jnp.zeros(R), jnp.zeros(R, jnp.int64), jnp.zeros(R))
    (Pf, J, nbad, pmax), acts = jax.lax.scan(body, init, jnp.arange(cfg.T))
    if cfg.guard:
        jaxlog.nan_guard(J, (J, Pf), tag=2)
    return {"J": J, "n_pcli1_viol": nbad, "pmax": pmax, "acts": acts}


# ----------------------------------------------------------------------------------------------- exact optimum (tiny cases)
def exact_enumerate(bank: Bank, P0, T: int, K: int = 1, beta: float = BETA, allow_idle: bool = True):
    """EXACT optimum of the finite-horizon cost sum_{t<T} beta^t cost_t by enumeration of ALL open-loop action sequences
    (deterministic dynamics => open loop is optimal).  Actions per slot: idle (if allowed) or a K-subset.
    P0 (N,L,L).  Returns (J*, best action index sequence (T,), action table (A,N)).  |A|^T sequences: tiny cases only."""
    n = P0.shape[0]
    A = combos_mask(n, K)
    if allow_idle:
        A = np.vstack([np.zeros((1, n)), A])
    na = A.shape[0]
    S = na ** T
    if S > 2_000_000:
        raise ValueError(f"enumeration too large: {na}^{T} = {S}")
    seq = np.zeros((S, T), np.int32)
    ar = np.arange(S)
    for t in range(T):
        seq[:, T - 1 - t] = (ar // na ** t) % na
    seq = jnp.asarray(seq)
    Aj = jnp.asarray(A)

    @jax.jit
    def run(seq):
        def body(carry, t):
            P, J = carry                                                 # P (S,N,L,L)
            a = Aj[seq[:, t]]                                            # (S,N)
            c = jax.vmap(lambda p, aa: step_cost(bank, p, aa))(P, a)
            Pn = jax.vmap(lambda p, aa: phi(bank, p, aa))(P, a)
            return (Pn, J + beta ** t * c), None
        P = jnp.broadcast_to(P0, (seq.shape[0],) + P0.shape)
        (_, J), _ = jax.lax.scan(body, (P, jnp.zeros(seq.shape[0])), jnp.arange(T))
        return J

    J = run(seq)
    i = jnp.argmin(J)
    return float(J[i]), np.asarray(seq[i]), A
