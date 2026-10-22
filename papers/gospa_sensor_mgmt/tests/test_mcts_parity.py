"""One full MCTS decision (UCT, expansion order, rollouts, running-mean back-propagation) of src/planner.py
(CFG-O) vs the authors' best_action() with the SAME injected rollout choices: identical chosen action and root-child C-bar."""
import numpy as np
import jax, jax.numpy as jnp
import pytest
from src import planner as PL
from src import model as M
import official

CFG = M.CFG_O
NP_EXTRA = 12


class Inj:
    it = -1
    j = 0
    U = None


def _patch(cls):
    orig = cls.rollout

    def rollout(self):
        Inj.it += 1
        Inj.j = 0
        return orig(self)

    def rollout_policy(self, avail):
        u = Inj.U[Inj.it, Inj.j]
        Inj.j += 1
        return avail[int(np.floor(u * len(avail)))]
    cls.rollout = rollout
    cls.rollout_policy = rollout_policy
    return orig


def _belief(r, pos, P, T):
    n = len(r)
    NP = CFG.r_cap + T
    return PL.Belief(jnp.zeros(NP).at[:n].set(r), jnp.zeros((NP, 2)).at[:n].set(pos),
                     jnp.tile(jnp.eye(2), (NP, 1, 1)).at[:n].set(P), jnp.asarray(n, jnp.int32))


CASES = [  # (S, T, B)
    (1, 1, 7), (1, 5, 40), (1, 5, 7), (1, 10, 40), (2, 1, 49), (2, 5, 49), (2, 5, 200), (2, 10, 60),
]


@pytest.mark.parametrize("kind,fn", [("GD", "GOSPA"), ("KLD", "KLD")])
@pytest.mark.parametrize("S,T,B", CASES)
def test_mcts_decision_matches_official(kind, fn, S, T, B):
    off = official.load()
    if off is None:
        pytest.skip(official.SKIP)
    cls = off["MCTS"].MonteCarloTreeSearchNode_individual if S == 1 else off["MCTS_joint"].MonteCarloTreeSearchNode_joint
    orig = _patch(cls)
    try:
        rng = np.random.default_rng(100 + 10 * S + T)
        plan = jax.jit(PL.make_mcts(CFG, S, T, B, kind))
        ntrial = 3 if B >= 200 else 4
        gaps = []
        for trial in range(ntrial):
            n = int(rng.integers(2, 8))
            pos = rng.uniform(-40, 40, (n, 2)) + np.array([0.0, 60.0 * (trial % 2)])
            P = np.zeros((n, 2, 2))
            for i in range(n):
                A = rng.normal(size=(2, 2)) * 3.0
                P[i] = A @ A.T + np.diag(rng.uniform(2.0, 15.0, 2))
            r = rng.uniform(0.05, 0.99, n)
            # sensors: trial 0 far from everything, trial 1 near the obstacle top edge (some actions infeasible), others random
            if trial == 1:
                sp = [np.array([4.0, 94.0]), np.array([-6.0, 96.0])]
            else:
                sp = [rng.uniform(-100, 100, 2) for _ in range(2)]
            Tm = max(T - 1, 1)
            U = rng.uniform(size=(B, Tm))
            Inj.it, Inj.U = -1, U
            off["Parameters"].simulation_search_depth = T
            off["Parameters"].MCTS_simulation_number = B
            kw = dict(parent=None, means_existing=pos.T.copy(), covariances_existing=P.copy(), prob_existences=r.copy(), cost=0, cost_function=fn)
            if S == 1:
                root = cls(sensor_mean=sp[0].reshape(2, 1).copy(), **kw)
            else:
                root = cls(sensor_means=[q.reshape(2, 1).copy() for q in sp], **kw)
            best = root.best_action()
            o = plan(_belief(r, pos, P, T), jnp.asarray(np.stack(sp[:S])), jnp.asarray(U), jnp.zeros(B))
            if S == 1:
                opos = np.asarray(best.sensor_mean).reshape(1, 2)
            else:
                opos = np.stack([np.asarray(q).reshape(2) for q in best.sensor_means])
            # root-child C-bar by action index: reconstruct official action index through the position
            offs = M.action_offsets()
            cbar_o = {}
            for ch in root.children:
                cbar_o[tuple(np.round(np.asarray(ch.sensor_mean if S == 1 else np.stack([np.asarray(q).reshape(2) for q in ch.sensor_means])).reshape(-1), 9))] = ch.cost
            mine = np.asarray(o["root_cost"])
            got = {}
            for a in range(7 ** S):
                if not np.isnan(mine[a]):
                    pp = np.asarray(PL.pos_of(jnp.asarray(np.stack(sp[:S])), a, jnp.asarray(offs)))
                    got[tuple(np.round(pp.reshape(-1), 9))] = mine[a]
            assert set(got) == set(cbar_o), (trial, len(got), len(cbar_o))
            gap = 0.0
            for k, v in cbar_o.items():
                gap = max(gap, abs(v - got[k]) / max(1.0, abs(v)))
            gaps.append(gap)
            assert gap < 1e-9, (trial, gap)
            assert np.allclose(opos, np.asarray(o["pos"]), atol=1e-9), (trial, opos, np.asarray(o["pos"]))
        print(f"[METRIC] mcts decision parity {kind} S={S} T={T} B={B}: {ntrial} trials, max root-child C-bar rel gap = {max(gaps):.2e}, chosen action identical")
    finally:
        cls.rollout = orig
        del cls.rollout_policy
