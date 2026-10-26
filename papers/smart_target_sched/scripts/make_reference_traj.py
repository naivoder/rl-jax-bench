"""Frozen reference trajectories for the environment checksum.  Generated with the independent numpy oracle
(src/oracle.py: literal eqs 3-7), NOT with the JAX model, so the fixture is a second path.  Pattern: actions 1,0,0,1 repeated, 32 steps."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import oracle as O

U = {"reckless": ((0.9, 0.1), (0.2, 0.8)), "cautious": ((0.95, 0.05), (0.6, 0.4))}
PAT = [1, 0, 0, 1] * 8
out = {}
for typ in U:
    t = O.scalar_tgt(*U[typ], qct=4.0)
    P = np.array([[1.3]]); tr = [P.copy()]
    for a in PAT:
        P = O.phi(t, P, a); tr.append(P.copy())
    out[f"scalar_{typ}"] = np.array(tr)
def fcv(ts=1.0): return np.kron(np.eye(2), np.array([[1, ts], [0, 1.0]]))
def fct(var, om=3 * np.pi / 180):
    s, c = np.sin(om), np.cos(om)
    sg = 1.0 if var == "printed" else -1.0
    return np.array([[1, s / om, 0, (c - 1) / om], [0, c, 0, sg * s], [0, (1 - c) / om, 1, s / om], [0, -sg * s, 0, c]])
qs = np.kron(np.eye(2), np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]]))
H = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]); R = 2 * np.eye(2)
P0 = np.array([[1.0, .2, .1, 0], [.2, 2.0, 0, .3], [.1, 0, 1.5, .1], [0, .3, .1, 0.8]])
for var in ("printed", "standard"):
    for typ in U:
        t = O.Tgt([fcv(), fct(var)], [1.0 * qs, 4.0 * qs], *U[typ], H, R)
        P = P0.copy(); tr = [P.copy()]
        for a in PAT:
            P = O.phi(t, P, a); tr.append(P.copy())
        out[f"matrix_{var}_{typ}"] = np.array(tr)
out["matrix_P0"] = P0
np.savez(os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "reference_traj.npz"), **out)
print({k: v.shape for k, v in out.items()})
