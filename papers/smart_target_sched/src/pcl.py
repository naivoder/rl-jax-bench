"""Extension X1: the third PCL-indexability condition (PCLI3) that the paper never checks, scalar case, exact (numpy oracle, python floats).

    PCLI3:  F(P,z2) - F(P,z1) = integral over (z1, z2] of mp*(z) G(P, dz)      (Nino-Mora 2020)

For fixed P, z -> (F, G)(P, z) of the z-threshold policy (also the first action) is a step function of z: the deterministic trajectory changes
only when z crosses a visited state value.  The next jump above z is the smallest ACTIVE visited state x_min > z (raising z to x_min makes it passive).
At each jump z_j the identity requires  F(z_j) - F(z) = mp*(z_j) * (G(z) - G(z_j))  (cost rises by mp* times the work removed).
Sign convention: G(P, z) is NON-INCREASING in z, so the printed integral of mp* G(P, dz) is read with |dG| (work removed), which is the orientation
consistent with the paper's own f = F<0> - F<1>, g = G<1> - G<0> and mp = f/g > 0; `rhs` below is mp* x (G(z) - G(z_j)) >= 0.
`pcli3_residual` walks all jumps in (z1, z2] and returns the per-jump and the aggregate relative residual.
"""
from __future__ import annotations

import numpy as np

import oracle as O


def _traj_fg(t, P, z, tau, beta):
    """F, G of the z-threshold policy from P (first action by the threshold too) and the active visited states."""
    F = G = 0.0
    disc = 1.0
    P = np.array(P, float)
    active_states = []
    for k in range(tau):
        x = O.s(P)
        act = int(x > z)
        if act:
            active_states.append(x)
        F += disc * O.cost(t, P, act)
        G += disc * act
        P = O.phi(t, P, act)
        disc *= beta
    return F, G, active_states


def pcli3_residual(t, P0, z1, z2, tau=100, beta=0.9, max_jumps=5000):
    z = z1
    F, G, act = _traj_fg(t, P0, z, tau, beta)
    F_start = F
    jumps = []
    while True:
        nxt = [x for x in act if x > z]
        if not nxt:
            break
        zj = min(nxt)
        if zj > z2 or len(jumps) >= max_jumps:
            break
        F2, G2, act2 = _traj_fg(t, P0, zj, tau, beta)
        mp = O.mp_index(t, np.array([[zj]]), tau, beta)
        lhs = F2 - F                      # cost increase at the jump
        rhs = mp * (G - G2)               # mp* x work removed
        jumps.append({"z": zj, "dF": lhs, "mp_dG": rhs, "rel": abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-6)})
        z, F, G, act = zj, F2, G2, act2
    total_lhs = F - F_start
    total_rhs = sum(j["mp_dG"] for j in jumps)
    return {"n_jumps": len(jumps), "max_rel_jump": max([j["rel"] for j in jumps], default=0.0),
            "total_lhs": total_lhs, "total_rhs": total_rhs, "total_rel": abs(total_lhs - total_rhs) / max(abs(total_lhs), 1e-300),
            "jumps": jumps}
