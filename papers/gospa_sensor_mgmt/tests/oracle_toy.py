"""Independent plain-numpy implementation of the reduced-tree node recursion straight from the paper's equations (no jax, no shared code
with src/planner.py).  Used as the exact-enumeration oracle for the exact toy and the search oracle."""
import itertools
import numpy as np

C = 80.0
C2 = C * C
F = np.kron(np.eye(2), np.array([[1.0, 1.0], [0.0, 1.0]]))
Q = 0.8 * np.kron(np.eye(2), np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]]))
H = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])
R = 2.0 * np.eye(2)
PS = 0.99
RB = 0.03
BIRTH_X = np.array([0, 0.1, 0, 0.1])
BIRTH_P = 6.0 * np.eye(4)


def gamma_star(tr):
    return 1.0 / (2.0 - min(2.0 * tr / C2, 1.0))


def cost_gamma(r, tr):
    """eq. 39 / 31 literal form."""
    if r > gamma_star(tr):
        return 0.5 * C2 * (1 - r) + r * min(tr, C2)
    return 0.5 * C2 * r


def cost_min(r, tr):
    """(D1): min(r c^2/2, (1-r) c^2/2 + r min(tr, c^2))."""
    return min(0.5 * C2 * r, 0.5 * C2 * (1 - r) + r * min(tr, C2))


def trace_of(P, trace):
    return float(np.trace(P)) if trace == "full" else float(P[0, 0] + P[2, 2])


def syn_update(P):
    S = H @ P @ H.T + R
    return P - P @ H.T @ np.linalg.inv(S) @ H @ P


def hypotheses(r, P, pds, p_mode):
    """All h in {0,1}^S: list of (h, p(h), r^h, P^h) via the sequential synthetic update (34), (36)."""
    out = []
    for h in itertools.product((0, 1), repeat=len(pds)):
        rc, Pc, p = r, P, 1.0
        for s, hs in enumerate(h):
            prob = (r if p_mode == "paper" else rc) * pds[s]
            if hs:
                p *= prob
                rc, Pc = 1.0, syn_update(Pc)
            else:
                p *= 1.0 - prob
                rc = (1 - pds[s]) * rc / ((1 - rc) + (1 - pds[s]) * rc)
        out.append((h, p, rc, Pc))
    return out


def node(r, P, pds, p_mode="paper", trace="full"):
    """(cost, merged r, merged P) of one Bernoulli: eq. 43 and eqs. 54-56."""
    hs = hypotheses(r, P, pds, p_mode)
    cost = sum(p * cost_gamma(rc, trace_of(Pc, trace)) for _, p, rc, Pc in hs)
    rbar = sum(p * rc for _, p, rc, _ in hs)
    Pbar = sum(p * rc * Pc for _, p, rc, Pc in hs) / rbar
    return cost, rbar, Pbar


def predict(r, x, P):
    return PS * r, F @ x, F @ P @ F.T + Q
