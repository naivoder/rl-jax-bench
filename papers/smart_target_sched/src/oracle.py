"""Independent pure-numpy reference ("oracle") for the deterministic model, written LITERALLY from the paper's equations
(per-mode loops, K = P-bar H' (H P-bar H' + R)^-1, P-hat = (I - K H) P-bar, python-float rollouts).  It shares no code with
src/model.py / src/algorithm.py and is used by the tests and by the toy checks as the second computation path
(there is no official code to run).  Slow by design; tiny cases only.
"""
from __future__ import annotations

import itertools

import numpy as np


class Tgt:
    """One target: lists of per-mode F, Q; mode probabilities u0, u1; H, R; weights d, h."""

    def __init__(self, F, Q, u0, u1, H, R, d=1.0, h=0.0):
        self.F = [np.atleast_2d(np.asarray(f, float)) for f in F]
        self.Q = [np.atleast_2d(np.asarray(q, float)) for q in Q]
        self.u = (np.asarray(u0, float), np.asarray(u1, float))
        self.H = np.atleast_2d(np.asarray(H, float))
        self.R = np.atleast_2d(np.asarray(R, float))
        self.d, self.h = float(d), float(h)
        self.L = self.F[0].shape[0]


def scalar_tgt(u0, u1, qct, d=1.0, h=0.0, fcv=1.1, fct=1.3, qcv=1.0, H=1.0, R=2.0):
    return Tgt([fcv, fct], [qcv, qct], u0, u1, H, R, d, h)


def phi(t: Tgt, P, a: int):
    """Eqs (3)-(7) literally."""
    out = np.zeros_like(P)
    for m in range(len(t.F)):
        Pb = t.F[m] @ P @ t.F[m].T + t.Q[m]                                   # (5)
        if a == 1:
            K = Pb @ t.H.T @ np.linalg.inv(t.H @ Pb @ t.H.T + t.R)             # (6)
            Ph = (np.eye(t.L) - K @ t.H) @ Pb                                  # (4)
            out += t.u[1][m] * Ph                                              # (3)
        else:
            out += t.u[0][m] * Pb                                              # (7)
    return out


def s(P):
    return float(np.trace(P)) / P.shape[0]


def cost(t: Tgt, P, a):
    return t.d * s(P) + t.h * a


def rollout(t: Tgt, P, z, a0, tau, beta):
    """F, G of policy <a0, z>: action a0 now, then 'active iff s(P) > z'."""
    F = G = 0.0
    disc = 1.0
    for k in range(tau):
        act = a0 if k == 0 else int(s(P) > z)
        F += disc * cost(t, P, act)
        G += disc * act
        P = phi(t, P, act)
        disc *= beta
    return F, G


def fg(t, P, z, tau=100, beta=0.9):
    F0, G0 = rollout(t, P, z, 0, tau, beta)
    F1, G1 = rollout(t, P, z, 1, tau, beta)
    return F0 - F1, G1 - G0                                                   # f, g


def mp_index(t, P, tau=100, beta=0.9):
    f, g = fg(t, P, s(P), tau, beta)
    return f / g


def myopic_index(t, P):
    return t.d * (s(phi(t, P, 0)) - s(phi(t, P, 1)))


def tev_index(t, P):
    return t.d * s(P)


def passive_cont_index(t, P, tau=100, beta=0.9):
    F0, _ = rollout(t, P, np.inf, 0, tau, beta)
    F1, _ = rollout(t, P, np.inf, 1, tau, beta)
    return F0 - F1


def simulate(tgts, P0, K, T, policy, beta=0.9, tau=100, skip_neg=False):
    """Closed-loop discounted cost for an index policy; ties broken by lowest index (use tie-free inputs when comparing)."""
    P = [p.copy() for p in P0]
    J = 0.0
    acts = []
    for tt in range(T):
        idx = np.array([policy(t, p) for t, p in zip(tgts, P)])
        order = np.argsort(-idx, kind="stable")[:K]
        if skip_neg:
            order = np.array([o for o in order if idx[o] >= 0], int)
        a = np.zeros(len(tgts), int); a[order] = 1
        J += beta ** tt * sum(cost(t, p, int(ai)) for t, p, ai in zip(tgts, P, a))
        P = [phi(t, p, int(ai)) for t, p, ai in zip(tgts, P, a)]
        acts.append(a)
    return J, np.array(acts)


def exact_optimum(tgts, P0, T, K=1, beta=0.9, allow_idle=True):
    """Exact optimum over all open-loop action sequences (the dynamics are deterministic => open loop is optimal).
    Actions per slot: idle (if allowed) or any K-subset.  Returns (J*, best sequence of subsets)."""
    n = len(tgts)
    acts = [tuple(c) for c in itertools.combinations(range(n), K)]
    if allow_idle:
        acts = [()] + acts
    best = (np.inf, None)

    def rec(tt, P, J, seq):
        nonlocal best
        if tt == T:
            if J < best[0]:
                best = (J, list(seq))
            return
        for a in acts:
            av = [1 if i in a else 0 for i in range(n)]
            c = sum(cost(t, p, ai) for t, p, ai in zip(tgts, P, av))
            Pn = [phi(t, p, ai) for t, p, ai in zip(tgts, P, av)]
            rec(tt + 1, Pn, J + beta ** tt * c, seq + [a])

    rec(0, [p.copy() for p in P0], 0.0, [])
    return best
