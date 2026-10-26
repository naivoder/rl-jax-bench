"""Parity against the AUTHORS' OWN OUTPUT.  There is no released code for arXiv:2312.07858v1,
so the differential oracle is the authors' computed numbers that ARE public:

  * Tables I-IV exactly as typeset in the paper's TeX source (tests/fixtures/paper_tex_tables.json)  -> checks the PAPER dict in src/eval.py
    that every MATCH/CLOSE judgement is graded against (an independent transcription, not the PDF text the dict was typed from);
  * the vector figures in the TeX source (tests/fixtures/paper_fig_curves.npz): polyline vertices recovered exactly from the PDFs by
    scripts/extract_fig_vectors.py.  Fig 2/3 (g and f against P, 8 panels x 1501 grid points), Fig 4 (MP index against P and against Q^CT).
    These are the authors' implementation's outputs, so g, f and mp* of src/algorithm.py are differential-tested point by point.

Tolerances (stated, not tuned): the PDF coordinates are rounded and the axes are re-calibrated from tick labels, so a plotted value carries
~2.5e-4 relative error (axis-fit residual 1e-5..1e-3 of the axis range).  g: 1e-4 absolute (range 0.4-1).  f: the authors rescale f affinely to share
the g axis, i.e. f is drawn on the right-hand y-axis of the paper's own figure (MATLAB yyaxis, Figs 2-3 p.9); the affine fit a*f+b is allowed per panel for
the shape check (residual < 1e-4 of the plotted range) AND the fitted (a, b) are pinned to that right-axis calibration (FIT_AB below), so an f off by any
constant or scale factor fails.  mp*: 1e-3
relative to the curve maximum.
A fixture-free, code-vs-code parity check is in tests/test_toys.py (independent numpy oracle, src/oracle.py); here the oracle is the authors."""
import json
import os

import numpy as np
import pytest
import jax.numpy as jnp

import model as M
import algorithm as A
import eval as E

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
CURVES = np.load(os.path.join(FIX, "paper_fig_curves.npz"))
TABLES = json.load(open(os.path.join(FIX, "paper_tex_tables.json")))["tables"]


# ---------------------------------------------------------------------------------------------------------- Tables I-IV
def test_paper_dict_equals_tex_tables():
    """src/eval.py::PAPER (the numbers every Table judgement is graded against) equals the TeX-typeset tables, all 81 numbers."""
    def check(cell, rows, i0):
        for K in (1, 2, 3):
            exp = tuple(rows[i0 + p][K - 1] for p in range(3))
            assert E.PAPER[cell][K] == exp, (cell, K, E.PAPER[cell][K], exp)

    check("t1_q2", TABLES["tab:q1_rec"], 0); check("t1_q29", TABLES["tab:q1_rec"], 3)
    check("t2_q2", TABLES["tab:q1_cau"], 0); check("t2_q29", TABLES["tab:q1_cau"], 3)
    check("t3_q2", TABLES["tab:q1_rec_cau"], 0); check("t3_q2345", TABLES["tab:q1_rec_cau"], 3)
    check("t4_reckless", TABLES["tab:q4"], 0); check("t4_cautious", TABLES["tab:q4"], 3); check("t4_mixed", TABLES["tab:q4"], 6)


def test_paper_tables_internal_order():
    """Every cell of the paper: index < min(baselines)  (the claim the reproduction grades)."""
    for cell, ks in E.PAPER.items():
        for K, (w, my, tev) in ks.items():
            assert w < min(my, tev), (cell, K)


# ---------------------------------------------------------------------------------------------------------- Figs 2-3 (g, f)
GRID = np.round(np.arange(0, 1501) * 0.01, 10)
PANELS = [(typ, suf, z, q) for typ, suf in [("reckless", "rec"), ("cautious", "cau")] for z in (4, 10) for q in (4, 10)]


def _fg(typ, q, z):
    P = GRID.copy(); P[0] = 1e-12                      # the authors' grid starts at P = 0; the state space is (0, inf): 1e-12 is the same point to 1e-12
    n = len(P)
    b = M.scalar_bank([typ] * n, [float(q)] * n, [1.0] * n)
    f, g = A.fg_marginal(b, jnp.asarray(P)[:, None, None], jnp.full(n, float(z)), 100, 0.9)
    return np.asarray(f), np.asarray(g)


# Right-hand-axis calibration of each f panel = (a, b) of f_plotted = a*f + b.  Derived from the axes drawn in the paper (left axis = g, right axis = f; both span the
# same height): (a, b) = (left span / right span, left lower limit - a * right lower limit).  Left axes 0.2-1 (span 0.8), 0.12-0.72 (0.6), 0.369-1 (0.631, lower limit = min g).
FIT_AB = {
    ("reckless", 4, 4): (0.631 / 20, 0.369),                 # right axis 0..20
    ("reckless", 4, 10): (0.6 / 25, 0.12),                   # right axis 0..25
    ("reckless", 10, 4): (0.8 / 15, 0.2),                    # right axis 0..15
    ("reckless", 10, 10): (0.8 / 15, 0.2),
    ("cautious", 4, 4): (0.631 / 25, 0.369 + 0.631 * 5 / 25),  # right axis -5..20
    ("cautious", 4, 10): (0.631 / 15, 0.369),                # right axis 0..15
    ("cautious", 10, 4): (0.04, 0.4),                        # Fig 3(b)
    ("cautious", 10, 10): (0.8 / 15, 0.2),
}


@pytest.mark.parametrize("typ,suf,z,q", PANELS)
def test_g_and_f_match_authors_plotted_curves(typ, suf, z, q):
    name = f"gfbeta09z{z}q{q}_{suf}"
    assert CURVES[f"{name}__axis_resid"][1] < 1e-3                    # the fixture's own axis calibration is tight
    x = CURVES[f"{name}__0__x"]; g_th = CURVES[f"{name}__0__y"]; f_th = CURVES[f"{name}__1__y"]
    assert x.shape == GRID.shape and np.abs(x - GRID).max() < 5e-4    # same 1e-2 grid, 1501 points (paper §IV-A)
    f, g = _fg(typ, q, z)
    assert np.abs(g - g_th).max() < 1e-4, np.abs(g - g_th).max()      # g(P, z): authors' curve = ours
    Am = np.vstack([f, np.ones_like(f)]).T
    coef, *_ = np.linalg.lstsq(Am, f_th, rcond=None)
    resid = np.abs(Am @ coef - f_th).max() / (f_th.max() - f_th.min())
    assert resid < 1e-4, (resid, coef)                                # f(P, z): same shape up to the plotting rescale
    assert coef[0] > 0                                                # orientation preserved (no sign flip)
    a_ref, b_ref = FIT_AB[(typ, z, q)]                                # pinned right-axis calibration: f = ours, no undisclosed rescale
    assert abs(coef[0] / a_ref - 1) < 2e-3 and abs(coef[1] - b_ref) < 2e-3, (coef, a_ref, b_ref)


# ---------------------------------------------------------------------------------------------------------- Fig 4 (MP index)
def _mp(typ, q, P):
    n = len(P); b = M.scalar_bank([typ] * n, [float(q)] * n, [1.0] * n)
    idx, nbad = A.index_whittle(b, jnp.asarray(P)[:, None, None], A.Cfg(guard=False))
    assert int(nbad) == 0                                              # PCLI1 holds at every plotted point
    return np.asarray(idx)


@pytest.mark.parametrize("name,q", [("type-q4", 4.0), ("type-q10", 10.0)])
def test_mp_index_vs_P_matches_authors(name, q):
    P = GRID.copy(); P[0] = 1e-12
    for ci, typ in enumerate(["reckless", "cautious"]):               # blue = reckless, orange = cautious (the swapped assignment is off by 1.9-3.8)
        th = CURVES[f"{name}__{ci}__y"]
        o = _mp(typ, q, P)
        assert np.abs(o - th).max() / th.max() < 1e-3, (typ, np.abs(o - th).max())


@pytest.mark.parametrize("name,P0", [("Q-x1", 1.0), ("Q-x10", 10.0)])
def test_mp_index_vs_Q_matches_authors(name, P0):
    Q = np.arange(2, 41, 1.0)                                          # authors' Q^CT grid: 39 points, 2..40
    for ci, typ in enumerate(["reckless", "cautious"]):
        th = CURVES[f"{name}__{ci}__y"]; x = CURVES[f"{name}__{ci}__x"]
        assert len(th) == len(Q) and np.abs(x - Q).max() < 1e-2
        o = np.array([_mp(typ, q, np.array([P0]))[0] for q in Q])
        assert np.abs(o - th).max() / th.max() < 1e-3, (typ, np.abs(o - th).max())


# ---------------------------------------------------------------------------------------------------------- facts read off the exact curves
def test_authors_curves_vs_paper_text_claims():
    """Paper text (SS IV-A): 'cautious targets obtain a higher index than reckless targets in all states'.  The authors' own Fig 4(c) (Q^CT = 4) obeys
    it; their Fig 4(d) (Q^CT = 10) does NOT at small P (cautious below reckless on P < 1.5 and on 2.5 < P < 3.4).  Our index reproduces those crossings
    (pointwise parity above), so the text claim is an over-statement of the authors' own data, not a discrepancy of ours.  PCLI1: g > 0 on every panel."""
    P = GRID
    rk4, ct4 = CURVES["type-q4__0__y"], CURVES["type-q4__1__y"]
    assert np.all(ct4 > rk4)
    rk10, ct10 = CURVES["type-q10__0__y"], CURVES["type-q10__1__y"]
    assert np.any(ct10 < rk10) and np.all(ct10[P > 3.5] > rk10[P > 3.5])
    assert np.all(ct10[P < 1.4] < rk10[P < 1.4])
    for typ, suf, z, q in PANELS:
        assert CURVES[f"gfbeta09z{z}q{q}_{suf}__0__y"].min() > 0.05


# ---------------------------------------------------------------------------------------------------------- Figs 7-8 (4-D performance per target)
def _per_target_J(name, K, nmc=6):
    import data as D
    scn = D.figure_scenario(name, K)
    bank = D.build_bank(scn)
    rk = D.rep_keys(0, "G2", scn.sid, K, nmc)
    p0k, tie, mode = D.split_rep_keys(rk)
    P0 = D.sample_P0(scn, p0k)
    return {pol: float(A.simulate(bank, P0, tie, mode, pol, K, A.Cfg(guard=False))["J"].mean()) / scn.N for pol in ("whittle", "myopic", "tev")}


@pytest.mark.parametrize("name,fix,K", [("fig8", "Bound_performance_rec_cau_ratio", 2), ("fig7", "Bound_performance_rec_cau", 2)])
def test_matrix_figs_per_target_cost_matches_authors(name, fix, K):
    """The 4-D scenarios of the paper's last two figures at the smallest point (N = 4K = 8) with 6 replications: per-target discounted cost within 1.5 %
    of the authors' plotted points (blue = MP index, magenta = myopic, green = TEC; their Nmc = 100 has SE ~0.4 %, ours at 6 replications ~1 %)."""
    kk = CURVES[f"{fix}__0__x"]; i = int(np.argmin(np.abs(kk - K)))
    th = {"whittle": CURVES[f"{fix}__0__y"][i], "myopic": CURVES[f"{fix}__1__y"][i], "tev": CURVES[f"{fix}__2__y"][i]}
    ours = _per_target_J(name, K)
    for pol in th:
        assert abs(ours[pol] / th[pol] - 1) < 0.015, (pol, ours[pol], th[pol])
    assert ours["whittle"] < min(ours["myopic"], ours["tev"]) or name == "fig7"     # fig7 at K = 2 is a three-way tie in the authors' data too (82.65/82.79/82.94)


# ---------------------------------------------------------------------------------------------------------- code-vs-code complement (4-D closed loop)
def test_matrix_closed_loop_matches_literal_numpy_oracle():
    """the toy checks checks the 4-D INDEX against src/oracle.py; this closes the loop (actions and J over a whole episode, three index policies, both F^CT variants'
    primary = printed).  The oracle is the literal-equation numpy implementation, the only second code path available (no authors' code)."""
    import data as D
    import oracle as O
    U = {"reckless": ((0.9, 0.1), (0.2, 0.8)), "cautious": ((0.95, 0.05), (0.6, 0.4))}
    rng = np.random.default_rng(11)
    types, qct, d = ["reckless", "cautious", "reckless", "cautious"], [3.0, 5.0, 9.0, 2.0], [5.0, 1.0, 1.0, 5.0]
    R0 = rng.uniform(0, 1, (4, 4, 4)); P0 = np.swapaxes(R0, -1, -2) @ R0
    b = M.matrix_bank(types, qct, d)
    tg = [O.Tgt([M.f_cv_matrix(), M.f_ct_matrix()], [M.q_shape_matrix(), q * M.q_shape_matrix()], *U[t], M.H_MATRIX, 2 * np.eye(2), dd)
          for t, q, dd in zip(types, qct, d)]
    rk = D.rep_keys(0, "G1", 97, 2, 1); _, tie, mode = D.split_rep_keys(rk)
    cfg = A.Cfg(T=10, tau=20, guard=False)
    for pol, fn in [("whittle", lambda t, p: O.mp_index(t, p, 20)), ("myopic", O.myopic_index), ("tev", O.tev_index)]:
        out = A.simulate(b, jnp.asarray(P0)[None], tie, mode, pol, 2, cfg)
        Jref, acts = O.simulate(tg, [p.copy() for p in P0], 2, 10, fn, tau=20)
        assert abs(float(out["J"][0]) - Jref) < 1e-9 * abs(Jref), (pol, float(out["J"][0]), Jref)
        assert np.array_equal(np.asarray(out["acts"])[:, 0], acts), pol


# ---------------------------------------------------------------------------------------------------------- G(P, z) vs z
GZ = np.load(os.path.join(FIX, "paper_fig_Gz.npz"))          # figs/Gzbeta0.9q4_{rec,cau}.pdf: 4 curves x 14901 points, z = 0.1..15, beta 0.9, Q^CT 4 (legend not
                                                             # printed in the paper; the four curves are P = 2, 4, 6, 8 in plotting order, identified by a scan over P = 0.01..20)
GZ_P = (2.0, 4.0, 6.0, 8.0)


def _G_ours(typ, P, z):
    n = z.shape[0]
    bank = M.scalar_bank([typ] * n, [4.0] * n, 1.0)
    _, G = A.rollout_FG(bank, jnp.full((n, 1, 1), P), jnp.asarray(z), jnp.asarray((P > z).astype(float)), 100, 0.9)
    return np.asarray(G)


@pytest.mark.parametrize("typ,key", [("reckless", "Gzbeta0.9q4_rec"), ("cautious", "Gzbeta0.9q4_cau")])
@pytest.mark.parametrize("ci", range(4))
def test_G_of_P_z_matches_authors_vector_figure(typ, key, ci):
    """The authors' own G(P, z) against the threshold z (rollout of the z-threshold policy, active iff s(P) > z, tau = 100).  Tolerance 2e-4 away from the jumps
    (observed max 4.8e-5 reckless / 8.8e-5 cautious); the plot resolution in z is 0.001, so a sample next to a jump may differ by the jump height and is excluded
    (<= 3 samples either side of one of OUR jumps); every jump of ours must sit within 2 plot steps of one of the authors'; the plateau at small z is sum_{t<100} 0.9^t
    (the authors truncate the rollout at tau = 100 as we do)."""
    x, y = GZ[f"{key}__{ci}__x"], GZ[f"{key}__{ci}__y"]
    P = GZ_P[ci]
    G = _G_ours(typ, P, x)
    jo = np.where(np.abs(np.diff(G)) > 1e-3)[0]; ja = np.where(np.abs(np.diff(y)) > 1e-3)[0]
    near = np.zeros(x.size, bool)
    for j in jo:
        near[max(0, j - 3):j + 5] = True
    assert np.max(np.abs(G - y)[~near]) < 2e-4
    zo, za = 0.5 * (x[jo] + x[jo + 1]), 0.5 * (x[ja] + x[ja + 1])
    assert all(np.min(np.abs(za - z)) <= 0.0021 for z in zo)
    assert abs(float(y[0]) - sum(0.9 ** t for t in range(100))) < 1e-3 and abs(float(G[0]) - float(y[0])) < 1e-3
    # strict tie at z = P: the sample whose z equals P belongs to the PASSIVE-first branch in the authors' data, as in ours (active iff P > z)
    k = int(round((P - 0.1) / 0.001))
    G3 = _G_ours(typ, P, np.array([P - 1e-4, P, P + 1e-4]))
    assert abs(float(y[k]) - float(G3[1])) < 2e-4 and abs(float(y[k - 1]) - float(G3[0])) < 2e-4
    assert abs(float(G3[1]) - float(G3[0])) > 0.1
