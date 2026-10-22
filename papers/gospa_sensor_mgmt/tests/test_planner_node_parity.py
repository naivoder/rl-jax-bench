"""Node cost + merged Bernoullis of src/planner.py (CFG-O) vs the authors' individual and joint MCTS nodes,
GD and KLD, on random Bernoulli lists and sensor positions (depth 1 and depth 2), plus the legal-action set."""
import numpy as np
import jax, jax.numpy as jnp
import pytest
from src import planner as PL
from src import model as M
import official

CFG = M.CFG_O
NP = 24


def _rand_belief(rng, n):
    pos = rng.uniform(-60, 60, (n, 2))
    P = np.zeros((n, 2, 2))
    for i in range(n):
        A = rng.normal(size=(2, 2)) * rng.choice([1.0, 3.0, 8.0])
        P[i] = A @ A.T + np.diag(rng.uniform(1.0, 20.0, 2))
    r = rng.uniform(0.02, 0.99, n)
    return r, pos, P


def _to_belief(r, pos, P):
    n = len(r)
    return PL.Belief(jnp.zeros(NP).at[:n].set(r), jnp.zeros((NP, 2)).at[:n].set(pos),
                     jnp.tile(jnp.eye(2), (NP, 1, 1)).at[:n].set(P), jnp.asarray(n, jnp.int32))


def _compare(cost_o, r_o, x_o, P_o, cost_m, b, n):
    rm, xm, Pm = np.asarray(b.r)[:n], np.asarray(b.x)[:n], np.asarray(b.P)[:n]
    e = [abs(cost_o - float(cost_m)) / max(1.0, abs(cost_o)), np.max(np.abs(r_o - rm)), np.max(np.abs(x_o.T - xm) / (1 + np.abs(xm))),
         np.max(np.abs(P_o - Pm) / (1 + np.abs(Pm)))]
    return max(e)


@pytest.mark.parametrize("kind,fn", [("GD", "GOSPA"), ("KLD", "KLD")])
def test_individual_node_matches_official(kind, fn):
    off = official.load()
    if off is None:
        pytest.skip(official.SKIP)
    P_ = off["Parameters"]; P_.simulation_search_depth = 5
    rng = np.random.default_rng(11)
    pc = PL.PC(CFG)
    f = jax.jit(lambda b, pos, d: PL.expand_belief(b, pos, d, pc, CFG, kind))
    worst = 0.0
    for it in range(100):
        n = int(rng.integers(1, 9))
        r, pos, P = _rand_belief(rng, n)
        s = rng.uniform(-60, 60, (2, 1))
        root = off["MCTS"].MonteCarloTreeSearchNode_individual(parent=None, means_existing=pos.T.copy(), covariances_existing=P.copy(),
                                                              sensor_mean=s, prob_existences=r.copy(), cost=0, cost_function=fn)
        acts = list(root.untried_actions)
        a = int(rng.integers(0, len(acts)))
        root.untried_actions = [acts[a]]
        ch = root.expand()
        c2 = f(_to_belief(r, pos, P), jnp.asarray(ch.sensor_mean.reshape(1, 2)), jnp.asarray(0))
        worst = max(worst, _compare(ch.cost, ch.prob_existences, ch.means_existing, ch.covariances_existing, c2[0], c2[1], n + 1))
        # depth 2
        acts = list(ch.untried_actions)
        ch.untried_actions = [acts[int(rng.integers(0, len(acts)))]]
        g = ch.expand()
        c3 = f(c2[1], jnp.asarray(g.sensor_mean.reshape(1, 2)), jnp.asarray(1))
        worst = max(worst, _compare(g.cost, g.prob_existences, g.means_existing, g.covariances_existing, c3[0], c3[1], n + 2))
    print(f"[METRIC] node parity {kind} ({fn}): worst rel err vs official = {worst:.2e}")
    assert worst < 1e-9, worst


@pytest.mark.parametrize("kind,fn", [("GD", "GOSPA"), ("KLD", "KLD")])
def test_joint_node_matches_official(kind, fn):
    off = official.load()
    if off is None:
        pytest.skip(official.SKIP)
    off["Parameters"].simulation_search_depth = 5
    rng = np.random.default_rng(12)
    pc = PL.PC(CFG)
    f = jax.jit(lambda b, pos, d: PL.expand_belief(b, pos, d, pc, CFG, kind))
    worst = 0.0
    for it in range(100):
        n = int(rng.integers(1, 9))
        r, pos, P = _rand_belief(rng, n)
        s = [rng.uniform(-60, 60, (2, 1)) for _ in range(2)]
        root = off["MCTS_joint"].MonteCarloTreeSearchNode_joint(parent=None, means_existing=pos.T.copy(), covariances_existing=P.copy(),
                                                               sensor_means=s, prob_existences=r.copy(), cost=0, cost_function=fn)
        acts = list(root.untried_actions)
        a = int(rng.integers(0, len(acts)))
        root.untried_actions = [acts[a]]
        ch = root.expand()
        pn = np.stack([np.asarray(q).reshape(2) for q in ch.sensor_means])
        c2 = f(_to_belief(r, pos, P), jnp.asarray(pn), jnp.asarray(0))
        worst = max(worst, _compare(ch.cost, ch.prob_existences, ch.means_existing, ch.covariances_existing, c2[0], c2[1], n + 1))
        acts = list(ch.untried_actions)
        ch.untried_actions = [acts[int(rng.integers(0, len(acts)))]]
        g = ch.expand()
        pn = np.stack([np.asarray(q).reshape(2) for q in g.sensor_means])
        c3 = f(c2[1], jnp.asarray(pn), jnp.asarray(1))
        worst = max(worst, _compare(g.cost, g.prob_existences, g.means_existing, g.covariances_existing, c3[0], c3[1], n + 2))
    print(f"[METRIC] node parity {kind} ({fn}): worst rel err vs official = {worst:.2e}")
    assert worst < 1e-9, worst


def test_legal_actions_match_official():
    off = official.load()
    if off is None:
        pytest.skip(official.SKIP)
    rng = np.random.default_rng(13)
    off["Parameters"].simulation_search_depth = 5
    offs = jnp.asarray(M.action_offsets())
    for it in range(200):
        # include positions near the obstacle edges and the region boundary
        kind = it % 3
        if kind == 0:
            p = rng.uniform(-260, 260, 2)
        elif kind == 1:
            p = np.array([rng.uniform(-50, 50), rng.uniform(45, 105)])
        else:
            p = np.array([rng.choice([-235.0, 235.0, 240.0, -245.0]) + rng.normal(), rng.uniform(-250, 250)])
        node = off["MCTS"].MonteCarloTreeSearchNode_individual.__new__(off["MCTS"].MonteCarloTreeSearchNode_individual)
        legal = node.get_legal_actions(p.reshape(2, 1))
        feas = np.asarray(PL.feas_sensor(jnp.asarray(p), offs))
        cand = p[None, :] + np.asarray(offs)
        exp = [cand[i] for i in range(7) if feas[i]]
        assert len(legal) == len(exp), (p, len(legal), len(exp))
        for a, b in zip(legal, exp):
            assert np.allclose(np.asarray(a).reshape(2), b, atol=1e-12)
