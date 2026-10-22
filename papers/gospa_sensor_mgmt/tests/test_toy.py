"""Exactly computable toy and the closed-form identities, against an independent numpy oracle AND the
JAX planner primitives.  The documented numbers were derived by hand (6 decimals; listed in the README): the assertion tolerance against them is
5e-7; numpy oracle vs JAX is asserted to 1e-10."""
import itertools
import numpy as np
import jax, jax.numpy as jnp
import pytest
import oracle_toy as O
from src import planner as PL
from src import model as M

CFG_PT = M.CFG_P.replace(plan_trace="full")       # the toy uses the full-state trace "as printed"
R0, X0, P0 = 0.5, np.zeros(4), np.diag([10.0, 1.0, 10.0, 1.0])
PDS = [0.9, 0.5]


def _belief(r, x, P, NP=4):
    return PL.Belief(jnp.zeros(NP).at[0].set(r), jnp.zeros((NP, 4)).at[0].set(x),
                     jnp.tile(jnp.eye(4), (NP, 1, 1)).at[0].set(P), jnp.asarray(1, jnp.int32))


def _jax_node(cfg, b, pds=PDS):
    pc = PL.PC(cfg)
    return PL.eval_belief(b, jnp.zeros((2, 2)), pc, cfg, "GD", pd_override=jnp.asarray(pds))


def test_one_step_table_and_bound():
    hs = O.hypotheses(R0, P0, PDS, "paper")
    exp = {  # hand derivation: h -> (p paper, r^h, trP^h, C)
        (0, 0): (0.4125, 0.047619, 22.0, 152.380952), (0, 1): (0.1375, 1.0, 5.333333, 5.333333),
        (1, 0): (0.3375, 1.0, 5.333333, 5.333333), (1, 1): (0.1125, 1.0, 3.818182, 3.818182)}
    for h, p, rc, Pc in hs:
        e = exp[h]
        assert abs(p - e[0]) < 5e-7 and abs(rc - e[1]) < 5e-7 and abs(np.trace(Pc) - e[2]) < 5e-7
        assert abs(O.cost_gamma(rc, np.trace(Pc)) - e[3]) < 5e-7
    c, rbar, Pbar = O.node(R0, P0, PDS, "paper", "full")
    assert abs(c - 65.820022) < 5e-7 and abs(rbar - 0.607143) < 5e-7 and abs(np.trace(Pbar) - 5.591800) < 5e-7
    ce, rbe, Pbe = O.node(R0, P0, PDS, "exact", "full")
    assert abs(ce - 82.192424) < 5e-7 and abs(rbe - 0.5) < 1e-12 and abs(np.trace(Pbe) - 5.484848) < 5e-7
    # JAX planner primitives agree with the oracle to 1e-10 in both p(h) modes
    for mode, cfg in (("paper", CFG_PT), ("exact", CFG_PT.replace(ph_mode="sequential"))):
        cj, bj = _jax_node(cfg, _belief(R0, X0, P0))
        co, ro, Po = O.node(R0, P0, PDS, mode, "full")
        assert abs(float(cj) - co) < 1e-10 and abs(float(bj.r[0]) - ro) < 1e-12
        assert np.max(np.abs(np.asarray(bj.P[0]) - Po)) < 1e-10


def test_hypothesis_probabilities_and_martingale():
    for mode in ("paper", "exact"):
        hs = O.hypotheses(R0, P0, PDS, mode)
        assert abs(sum(p for _, p, _, _ in hs) - 1.0) < 1e-12
    exact = O.hypotheses(R0, P0, PDS, "exact")
    assert abs(sum(p * rc for _, p, rc, _ in exact) - R0) < 1e-12           # posterior-existence martingale under p*
    paper = O.hypotheses(R0, P0, PDS, "paper")
    assert abs(sum(p * rc for _, p, rc, _ in paper) - R0) > 0.05            # broken by eq. 38 (r-bar = 0.607)
    # p* closed form: r prod p^h (1-p)^(1-h) + (1-r) 1[h=0]
    for h, p, rc, Pc in exact:
        pst = R0 * np.prod([PDS[s] if h[s] else 1 - PDS[s] for s in range(2)]) + (1 - R0) * (h == (0, 0))
        assert abs(p - pst) < 1e-12


def test_information_form_identity():
    Ri = np.linalg.inv(O.R)
    for h, p, rc, Pc in O.hypotheses(R0, P0, PDS, "paper"):
        n = sum(h)
        Pi = np.linalg.inv(P0) if np.linalg.det(P0) > 0 else None
        Pinf = np.linalg.inv(np.linalg.inv(P0) + n * O.H.T @ Ri @ O.H)
        assert np.max(np.abs(Pc - Pinf)) < 1e-10


def test_closed_form_vs_gamma_branch_grid():
    for tr in [0.0, 0.5, 5.0, 24.0, 1000.0, 3199.9, 3200.0, 3200.1, 5000.0, 6400.0, 7000.0]:
        for r in [0.0, 1e-9, 0.2, O.gamma_star(tr) - 1e-9, O.gamma_star(tr), O.gamma_star(tr) + 1e-9, 0.9, 1.0]:
            r = min(max(r, 0.0), 1.0)
            assert abs(O.cost_gamma(r, tr) - O.cost_min(r, tr)) < 1e-6 * (1 + O.cost_min(r, tr)), (tr, r)
    assert abs(O.gamma_star(24.0) - 0.501882) < 5e-7 and abs(O.gamma_star(1000.0) - 0.592593) < 5e-7 and O.gamma_star(3200.0) == 1.0
    # JAX gospa_cost == the literal oracle on a random grid
    pc = PL.PC(CFG_PT)
    rng = np.random.default_rng(0)
    r = rng.uniform(0, 1, 200); tr = rng.uniform(0, 7000, 200)
    P = np.zeros((200, 4, 4)); P[:, 0, 0] = tr / 2; P[:, 2, 2] = tr / 2
    got = np.asarray(PL.gospa_cost(jnp.asarray(r), jnp.asarray(P), PL.PC(CFG_PT.replace(plan_trace="pos")), 80.0))
    ref = np.array([O.cost_gamma(a, b) for a, b in zip(r, tr)])
    assert np.max(np.abs(got - ref)) < 1e-9


def test_bound_dominates_monte_carlo():
    rng = np.random.default_rng(1)
    r, P = 0.6, np.diag([10.0, 1.0, 10.0, 1.0])
    n = 400_000
    exists = rng.random(n) < r
    d = rng.multivariate_normal(np.zeros(4), P, n)
    pos_err2 = d[:, 0] ** 2 + d[:, 2] ** 2
    report = r * 0 + 0.5 * O.C2 * (1 - r) + r * np.mean(np.minimum(pos_err2, O.C2))
    miss = 0.5 * O.C2 * r
    exact = min(report, miss)
    bound_full = O.cost_min(r, 22.0)      # full-state trace as printed (hand derivation: 1293.20)
    bound_pos = O.cost_min(r, 20.0)       # position-block trace (the coordinates GOSPA is computed on)
    assert abs(bound_full - 1293.20) < 0.01 and abs(bound_pos - 1292.0) < 1e-9
    assert bound_pos >= exact - 3.0 and bound_full >= bound_pos   # MC error ~ a few units
    assert abs(exact - 1288.64) < 12.0


def test_two_step_reduced_vs_full_tree():
    lam = 0.9
    out = {}
    for mode in ("paper", "exact"):
        c1, rbar, Pbar = O.node(R0, P0, PDS, mode, "full")
        r2, x2, P2 = O.predict(rbar, X0, Pbar)
        c2r, _, _ = O.node(r2, P2, PDS, mode, "full")
        Vr = c1 + lam * c2r
        # full tree (eq. 40): expectation over the step-1 branches of the step-2 node cost
        c2f = 0.0
        for h, p, rc, Pc in O.hypotheses(R0, P0, PDS, mode):
            rr, xx, PP = O.predict(rc, X0, Pc)
            c2f += p * O.node(rr, PP, PDS, mode, "full")[0]
        Vf = c1 + lam * c2f
        out[mode] = (c2r, Vr, c2f, Vf)
        assert Vr >= Vf                                                           # reduced tree is pessimistic on this toy
        # JAX planner primitives reproduce the reduced values
        cfg = CFG_PT if mode == "paper" else CFG_PT.replace(ph_mode="sequential")
        cj1, bj = _jax_node(cfg, _belief(R0, X0, P0))
        bp, _ = PL.predict_belief(bj._replace(nb=jnp.asarray(1, jnp.int32)), PL.PC(cfg), cfg.replace(r_b=0.0))
        cj2, _ = _jax_node(cfg, bp._replace(nb=jnp.asarray(1, jnp.int32)))
        assert abs(float(cj1) + lam * float(cj2) - Vr) < 1e-9
    assert abs(out["paper"][0] - 75.363149) < 5e-7 and abs(out["paper"][1] - 133.646856) < 5e-7
    assert abs(out["paper"][2] - 23.416898) < 5e-7 and abs(out["paper"][3] - 86.895230) < 5e-7
    assert abs(out["exact"][0] - 81.410508) < 5e-7 and abs(out["exact"][1] - 155.461882) < 5e-7
    assert abs(out["exact"][2] - 21.506198) < 5e-7 and abs(out["exact"][3] - 101.548002) < 5e-7
