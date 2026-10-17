"""Learning-free baselines and the exact solver (float64 numpy / scipy; no GPU needed).

* exact_solve      value iteration on the KNOWN interference chain -> Q*, V*, pi* (positive control / ceiling, §IV-B)
* stationary_J     long-run per-pulse reward of a theta_obs-policy under the chain's stationary law (D4)
* SAA              a*(theta_obs) (paper §IV-A); table data.ASTAR_ACTION
* random           uniform over 15 actions (floor, §IV-B)
* policy_iteration certainty-equivalent model over the full 16,000-state MDP from the training transitions (§III-B,
                   Selvi et al. [15]); unseen pairs get R = 0 and the default action.

the dynamics are exogenous, so pi* is myopic and depends on theta_obs only; Q*(s, a) = rbar(z, a) + gamma C(z).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from data import ACT_SIZE, ASTAR_ACTION, COLL_TABLE, FULL_BAND, MISS_TABLE, N_ACT, N_POS, N_THETA, N_VEL, Chain


def rbar(chain: Chain, R: np.ndarray) -> np.ndarray:
    """[nz, 15] expected next reward E[R(a, theta(z')) | z]."""
    return chain.T @ R[:, chain.pattern].T


def exact_solve(chain: Chain, R: np.ndarray, gamma: float = 0.9, sweeps: int = 2000) -> dict:
    """Value iteration on the latent chain (equivalent to the full MDP by exogeneity).

    Returns Q* [nz, 15], V* [nz], pi* [nz] (argmax ties -> lowest index) and the action gap per state.
    """
    rb = rbar(chain, R)
    V = np.zeros(chain.nz)
    for _ in range(sweeps):
        Q = rb + gamma * (chain.T @ V)[:, None]
        V = Q.max(1)
    Q = rb + gamma * (chain.T @ V)[:, None]
    pi = Q.argmax(1)
    srt = np.sort(Q, 1)
    return dict(Q=Q, V=Q.max(1), pi=pi, gap=srt[:, -1] - srt[:, -2], rbar=rb)


def stationary(chain: Chain) -> np.ndarray:
    w, v = np.linalg.eig(chain.T.T)
    mu = np.real(v[:, np.argmin(np.abs(w - 1))])
    return mu / mu.sum()


def stationary_J(chain: Chain, R: np.ndarray, policy: np.ndarray | None) -> float:
    """J(pi) = sum_z mu(z) rbar(z, pi(z)); policy None = uniform random."""
    rb, mu = rbar(chain, R), stationary(chain)
    per = rb.mean(1) if policy is None else rb[np.arange(chain.nz), policy]
    return float(mu @ per)


def stationary_table2(chain: Chain, policy: np.ndarray) -> dict:
    """Exact long-run Table-II metrics of a deterministic theta_obs policy (coll %, missed %, BW MHz, adapt %)."""
    mu, T = stationary(chain), chain.T
    nxt = chain.pattern
    coll = sum(mu[z] * T[z, z2] * COLL_TABLE[policy[z], nxt[z2]] for z in range(chain.nz) for z2 in range(chain.nz))
    miss = sum(mu[z] * T[z, z2] * MISS_TABLE[policy[z], nxt[z2]] for z in range(chain.nz) for z2 in range(chain.nz))
    bw = sum(mu[z] * ACT_SIZE[policy[z]] for z in range(chain.nz)) * 20.0
    adapt = sum(mu[z] * T[z, z2] * float(policy[z] != policy[z2]) for z in range(chain.nz) for z2 in range(chain.nz))
    return dict(coll_pct=100 * coll, missed_pct=100 * miss, bw_mhz=bw, adapt_pct=100 * adapt)


def saa_policy(chain: Chain) -> np.ndarray:
    return ASTAR_ACTION[chain.pattern]


# ----------------------------------------------------------------------------- policy iteration on an estimated model
N_FULL = N_POS * N_VEL * N_THETA     # 16,000 (paper §III-A)


def policy_iteration(s: np.ndarray, a: np.ndarray, r: np.ndarray, s2: np.ndarray, gamma: float = 0.9,
                     default_action: int = FULL_BAND, max_iter: int = 100, strict_improvement: bool = False) -> dict:
    """Certainty-equivalent PI over the full state space from transitions (s, a, r, s').

    Model: T_hat(s'|s,a) = n(s,a,s')/n(s,a), R_hat(s,a) = mean r. Unseen (s,a): R_hat = 0 and a self-loop.
    Policy evaluation is an exact sparse linear solve of (I - gamma P_pi) V = R_pi (the fixed point that iterating
    to max|dV| < 1e-6 approximates; amendment 2026-10-16 in the README). Improvement is greedy with lowest-index
    ties, iterated until the policy is stable. Evaluation-time policy: pi(s) on states seen in training, the
    default action (full band) on unseen states.
    strict_improvement (recheck, default False = shipped): switch pi(s) only where max_a Q > Q(s, pi(s)) + 1e-12,
    which rules out cycling among tied actions (policy-improvement theorem). `converged` reports a stable stop.
    """
    s, a, s2 = s.astype(np.int64), a.astype(np.int64), s2.astype(np.int64)
    r = r.astype(np.float64)
    sa = s * N_ACT + a
    n_sa = np.bincount(sa, minlength=N_FULL * N_ACT).astype(np.float64)
    w = 1.0 / n_sa[sa]                                   # per-transition weight 1/n(s,a)
    Rhat = np.bincount(sa, weights=r * w, minlength=N_FULL * N_ACT).reshape(N_FULL, N_ACT)
    seen_sa = (n_sa > 0).reshape(N_FULL, N_ACT)
    seen_s = seen_sa.any(1)
    pi = np.full(N_FULL, default_action, np.int64)
    eye = sp.identity(N_FULL, format="csr")
    it = 0
    for it in range(1, max_iter + 1):
        on = a == pi[s]                                  # transitions consistent with pi
        P = sp.csr_matrix((w[on], (s[on], s2[on])), shape=(N_FULL, N_FULL))
        unseen = ~seen_sa[np.arange(N_FULL), pi]
        P = P + sp.diags(unseen.astype(np.float64))      # unseen (s, pi(s)): self-loop, R = 0
        Rpi = Rhat[np.arange(N_FULL), pi]
        V = spla.spsolve((eye - gamma * P).tocsc(), Rpi)
        Q = Rhat + gamma * np.bincount(sa, weights=w * V[s2], minlength=N_FULL * N_ACT).reshape(N_FULL, N_ACT)
        Q = np.where(seen_sa, Q, gamma * V[:, None])
        new = Q.argmax(1)
        if strict_improvement:
            keep = Q[np.arange(N_FULL), pi] >= Q.max(1) - 1e-12
            new = np.where(keep, pi, new)
        if np.array_equal(new, pi):
            break
        pi = new
    converged = bool(np.array_equal(new, pi))
    table = np.where(seen_s, pi, default_action).astype(np.int32)
    return dict(table=table, iters=it, n_seen_states=int(seen_s.sum()), n_seen_sa=int(seen_sa.sum()), V=V,
                converged=converged)
