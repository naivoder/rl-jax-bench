"""Search oracle: exhaustive enumeration of the reduced-tree path values V(a_k:K) with an independent numpy
implementation of the node recursion vs the JAX MCTS, on a tiny geometry (two Bernoullis, two sensors, one obstacle), CFG-P semantics
(NCV 4-D planner state, eq. 38 product p(h), random expansion, no root double predict, stored-first-cost delta).

Statements checked exactly:
  (a) per-node costs: the myopic MCTS (T = 1, budget = fan-out) root-child C-bar equals V(a) = C_1(a) for every legal joint action and
      the chosen action is argmin_a V(a);
  (b) with budget = fan-out and T = 2, 3 each root child is visited once; its C-bar equals the oracle's path value along the SAME
      random rollout (replicated from the injected uniforms) -> checks Delta = sum_d lambda^(d-1) C_d, the rollout policy and
      the random expansion order;
  (c) descriptive: with a large budget the chosen first action approaches argmin_a1 min_a2 V (the min of eq. 40 is only
      approached where UCT concentrates visits; the mean-backup C-bar is NOT the min).  Reported, with the
      agreement rate over several geometries; asserted only that C-bar lies between min and max of the subtree values.
"""
import itertools
import numpy as np
import jax, jax.numpy as jnp
import pytest
import oracle_toy as O
from src import planner as PL
from src import model as M

CFG = M.CFG_P
LAM = CFG.lam
OFF = M.action_offsets()
X0, Y0, X1, Y1 = M.OBSTACLE


def feas1(p):
    c = p[None, :] + OFF
    inreg = np.all(np.abs(c) <= 250.0, axis=1)
    inobs = (c[:, 0] > X0) & (c[:, 0] < X1) & (c[:, 1] > Y0) & (c[:, 1] < Y1)
    return inreg & ~inobs


def feas_joint(pos):
    f = feas1(pos[0])
    for s in range(1, len(pos)):
        f = np.outer(f, feas1(pos[s])).reshape(-1)
    return f


def pos_of(pos, a):
    S = len(pos)
    return np.stack([pos[s] + OFF[(a // 7 ** (S - 1 - s)) % 7] for s in range(S)])


def expand(bel, pos, depth_parent):
    """oracle node expansion: bel = list of (r, x, P) 4-D (already predicted at depth 0); returns (cost, merged belief)."""
    if depth_parent > 0:
        bel = [O.predict(r, x, P) for (r, x, P) in bel] + [(O.RB, O.BIRTH_X, O.BIRTH_P)]
    cost, out = 0.0, []
    for (r, x, P) in bel:
        d = np.sqrt(((x[[0, 2]][None, :] - pos) ** 2).sum(1))
        pds = list(0.999 * np.exp(-0.5 * (d / 40.0) ** 2))
        c, rb, Pb = O.node(r, P, pds, "paper", "pos")
        cost += c
        out.append((rb, x, Pb))
    return cost, out


def rollout_path_value(bel0, pos0, T, a1, u_row):
    """V of root action a1 followed by the injected random rollout (u_row[j] for step j)."""
    pos = pos_of(pos0, a1)
    c, bel = expand(bel0, pos, 0)
    tot = c
    for d in range(2, T + 1):
        fe = feas_joint(pos)
        k = int(np.floor(u_row[d - 2] * fe.sum()))
        a = int(np.nonzero(fe)[0][k])
        pos = pos_of(pos, a)
        c, bel = expand(bel, pos, 1)
        tot += LAM ** (d - 1) * c
    return tot


def scenario(seed):
    rng = np.random.default_rng(seed)
    n = 2
    bel = []
    for i in range(n):
        r = rng.uniform(0.3, 0.9)
        x = np.array([rng.uniform(-40, 40), rng.normal() * 0.3, rng.uniform(20, 110), rng.normal() * 0.3])
        A = rng.normal(size=(4, 4)) * 1.5
        P = A @ A.T + np.diag(rng.uniform(1.0, 6.0, 4))
        bel.append((r, x, P))
    pos = np.stack([rng.uniform(-60, 60, 2) * np.array([1, 0.3]) + np.array([0, 105]), rng.uniform(-60, 60, 2) * np.array([1, 0.3]) + np.array([0, 125])])
    return bel, pos


def _root(bel, T):
    NP = M.R_CAP + T
    n = len(bel)
    r = np.zeros(NP); x = np.zeros((NP, 4)); P = np.tile(np.eye(4), (NP, 1, 1))
    for i, (a, b, c) in enumerate(bel):
        r[i], x[i], P[i] = a, b, c
    return PL.Belief(jnp.asarray(r), jnp.asarray(x), jnp.asarray(P), jnp.asarray(n, jnp.int32))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_myopic_root_costs_equal_enumeration(seed):
    bel, pos = scenario(seed)
    plan = jax.jit(PL.make_mcts(CFG, 2, 1, 49, "GD"))
    rng = np.random.default_rng(100 + seed)
    o = plan(_root(bel, 1), jnp.asarray(pos), jnp.asarray(rng.uniform(size=(49, 1))), jnp.asarray(rng.uniform(size=49)))
    fe = feas_joint(pos)
    V = {a: expand(bel, pos_of(pos, a), 0)[0] for a in np.nonzero(fe)[0]}
    got = np.asarray(o["root_cost"])
    for a, v in V.items():
        assert abs(got[a] - v) < 1e-9 * max(1, abs(v)), (a, got[a], v)
    assert int(o["action"]) == min(V, key=V.get)


@pytest.mark.parametrize("T", [2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_visit_per_root_child_equals_path_value(T, seed):
    bel, pos = scenario(10 + seed)
    fe = feas_joint(pos)
    B = int(fe.sum())
    plan = jax.jit(PL.make_mcts(CFG, 2, T, B, "GD"))
    rng = np.random.default_rng(200 + seed)
    U = rng.uniform(size=(B, T - 1)); UE = rng.uniform(size=B)
    o = plan(_root(bel, T), jnp.asarray(pos), jnp.asarray(U), jnp.asarray(UE))
    # replicate the random expansion order to know which iteration expanded which action
    untried = list(np.nonzero(fe)[0]); order = []
    for it in range(B):
        k = int(np.floor(UE[it] * len(untried)))
        order.append(untried.pop(k))
    got = np.asarray(o["root_cost"])
    for it, a in enumerate(order):
        v = rollout_path_value(bel, pos, T, a, U[it])
        assert abs(got[a] - v) < 1e-9 * max(1, abs(v)), (it, a, got[a], v)


def test_large_budget_between_min_and_max_and_agreement_rate():
    agree, tot = 0, 0
    for seed in range(6):
        bel, pos = scenario(30 + seed)
        T = 2
        fe = feas_joint(pos)
        V = {}
        for a1 in np.nonzero(fe)[0]:
            p1 = pos_of(pos, a1)
            c1, b1 = expand(bel, p1, 0)
            f2 = feas_joint(p1)
            V[a1] = [c1 + LAM * expand(b1, pos_of(p1, a2), 1)[0] for a2 in np.nonzero(f2)[0]]
        B = 600
        cfg = CFG.replace(uct_eps=0.0)
        plan = jax.jit(PL.make_mcts(cfg, 2, T, B, "GD"))
        rng = np.random.default_rng(300 + seed)
        o = plan(_root(bel, T), jnp.asarray(pos), jnp.asarray(rng.uniform(size=(B, 1))), jnp.asarray(rng.uniform(size=B)))
        got = np.asarray(o["root_cost"])
        for a1, vs in V.items():
            if not np.isnan(got[a1]):
                assert min(vs) - 1e-6 <= got[a1] <= max(vs) + 1e-6
        best_min = min(V, key=lambda a: min(V[a]))
        agree += int(int(o["action"]) == best_min)
        tot += 1
    print(f"[METRIC] large-budget MCTS chose argmin_a1 min_a2 V in {agree}/{tot} geometries")
