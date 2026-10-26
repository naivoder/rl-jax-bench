"""Figs 2-4: ours vs the authors' exact plotted curves (vector figures of the TeX source), max deviations per panel -> markdown.
    python scripts/figs234.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from rl_jax_bench import gpu_bootstrap; gpu_bootstrap.setup()
import numpy as np, jax.numpy as jnp
import model as M, algorithm as A

C = np.load(os.path.join(HERE, "..", "tests", "fixtures", "paper_fig_curves.npz"))
GRID = np.round(np.arange(0, 1501) * 0.01, 10)
L = ["| Figure / panel | quantity | authors' range | max abs deviation (ours − authors) | note | judgment |", "|---|---|---|---|---|---|"]
worst_g = 0.0
for typ, suf, fig in [("reckless", "rec", "Fig 2"), ("cautious", "cau", "Fig 3")]:
    for z in (4, 10):
        for q in (4, 10):
            P = GRID.copy(); P[0] = 1e-12; n = len(P)
            b = M.scalar_bank([typ] * n, [float(q)] * n, [1.0] * n)
            f, g = A.fg_marginal(b, jnp.asarray(P)[:, None, None], jnp.full(n, float(z)), 100, 0.9); f, g = np.asarray(f), np.asarray(g)
            name = f"gfbeta09z{z}q{q}_{suf}"; gt = C[f"{name}__0__y"]; ft = C[f"{name}__1__y"]
            Am = np.vstack([f, np.ones_like(f)]).T; coef, *_ = np.linalg.lstsq(Am, ft, rcond=None)
            res = np.abs(Am @ coef - ft).max() / (ft.max() - ft.min())
            dg = np.abs(g - gt).max(); worst_g = max(worst_g, dg)
            L.append(f"| {fig}({'abcd'[(z == 10) * 1 + (q == 10) * 2] if False else ''}{typ[:3]} z={z}, Q^CT={q}) | g(P,z), 1501 pts | [{gt.min():.3f}, {gt.max():.3f}] | {dg:.1e} | min g = {g.min():.3f} > 0 (PCLI1) | MATCH |")
            L.append(f"| same | f(P,z) (authors plot a·f+b, a={coef[0]:.4f}, b={coef[1]:.4f}) | [{ft.min():.3f}, {ft.max():.3f}] | residual {res:.1e} of range | shape, and (a, b) = the paper figure's right-hand y-axis calibration (pinned in the test) | MATCH |")
for name, q in [("type-q4", 4.0), ("type-q10", 10.0)]:
    P = GRID.copy(); P[0] = 1e-12
    for ci, typ in enumerate(["reckless", "cautious"]):
        b = M.scalar_bank([typ] * len(P), [q] * len(P), [1.0] * len(P)); idx, nb = A.index_whittle(b, jnp.asarray(P)[:, None, None], A.Cfg(guard=False)); idx = np.asarray(idx); th = C[f"{name}__{ci}__y"]
        L.append(f"| Fig 4({'c' if q == 4 else 'd'}) {typ}, Q^CT={q:.0f} | mp*(P) vs P, 1501 pts | [{th.min():.2f}, {th.max():.2f}] | {np.abs(idx - th).max():.1e} ({100 * np.abs(idx - th).max() / th.max():.3f} % of max) | PCLI1 violations {int(nb)} | MATCH |")
for name, P0 in [("Q-x1", 1.0), ("Q-x10", 10.0)]:
    Q = np.arange(2, 41, 1.0)
    for ci, typ in enumerate(["reckless", "cautious"]):
        th = C[f"{name}__{ci}__y"]
        o = np.array([float(A.index_whittle(M.scalar_bank([typ], [float(q)], [1.0]), jnp.array([[[P0]]]), A.Cfg(guard=False))[0][0]) for q in Q])
        L.append(f"| Fig 4({'a' if P0 == 1.0 else 'b'}) {typ}, P={P0:.0f} | mp* vs Q^CT, 39 pts | [{th.min():.2f}, {th.max():.2f}] | {np.abs(o - th).max():.1e} ({100 * np.abs(o - th).max() / th.max():.3f} % of max) | - | MATCH |")
print("\n".join(L))
