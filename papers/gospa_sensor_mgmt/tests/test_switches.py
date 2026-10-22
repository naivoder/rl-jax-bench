"""Follow-up options: info-gain KLD, full-trace CFG-P2 bound, node-count budget, Gamma* estimator,
merge after each update, checkpoint layout key.  Every default must leave the archived CFG-O / CFG-P behavior unchanged."""
import numpy as np
import jax, jax.numpy as jnp
import pytest
import oracle_toy as O
from src import planner as PL
from src import filter as FL
from src import model as M
from src import train as TR
from test_toy import _belief, R0, X0, P0, PDS

CFG2 = M.CFG_P2


def _kl_ber(a, b):
    out = 0.0
    if a > 0:
        out += a * np.log(a / b)
    if a < 1:
        out += (1 - a) * np.log((1 - a) / (1 - b))
    return out


def test_info_gain_closed_form_matches_numpy():
    cfg = CFG2.replace(kld_mode="info_gain")
    pc = PL.PC(cfg)
    cj, _ = PL.eval_belief(_belief(R0, X0, P0), jnp.zeros((2, 2)), pc, cfg, "KLD", pd_override=jnp.asarray(PDS))
    ref = 0.0
    for h, p, rc, Pc in O.hypotheses(R0, P0, PDS, "paper"):
        kg = 0.5 * (np.trace(np.linalg.solve(P0, Pc)) - 4 - np.log(np.linalg.det(Pc) / np.linalg.det(P0)))
        ref += p * (_kl_ber(rc, R0) + rc * kg)
    assert abs(float(cj) + ref) < 1e-10 and ref > 0           # cost = -E_h KL(post_h || pred); finite although r^h = 1 occurs


def test_bernoulli_kld_existence_term_is_eq38_artifact():
    """Evidence for the info-gain KLD form: under the exact p(h) the merged existence equals r (martingale) so KL(Ber(r) || Ber(r-bar)) = 0; the paper's product p(h)
    gives r-bar = 0.607 != 0.5 and a positive existence term."""
    for mode, exp0 in (("sequential", True), ("product", False)):
        cfg = CFG2.replace(ph_mode=mode)
        _, b = PL.eval_belief(_belief(R0, X0, P0), jnp.zeros((2, 2)), PL.PC(cfg), cfg, "GD", pd_override=jnp.asarray(PDS))
        rb = float(b.r[0])
        ex = _kl_ber(R0, rb)
        assert (abs(ex) < 1e-12) == exp0, (mode, rb, ex)


def test_full_trace_kld_block_is_4d():
    cfg = CFG2
    assert len(PL.PC(cfg).cost_idx) == 4 and len(PL.PC(M.CFG_P).cost_idx) == 2
    c, _ = PL.eval_belief(_belief(R0, X0, P0), jnp.zeros((2, 2)), PL.PC(cfg), cfg, "KLD", pd_override=jnp.asarray(PDS))
    assert np.isfinite(float(c))


def _tree(cfg, S, T, B, kind="GD", seed=0):
    rng = np.random.default_rng(seed)
    NP = 6
    r = jnp.zeros(NP).at[:2].set(jnp.asarray([0.8, 0.6]))
    x = jnp.zeros((NP, 4)).at[0].set(jnp.asarray([0.0, 1.0, 100.0, 0.0])).at[1].set(jnp.asarray([30.0, 0.0, 140.0, 0.0]))
    P = jnp.tile(jnp.eye(4), (NP, 1, 1)) * 5.0
    root = PL.Belief(r, x, P, jnp.asarray(2, jnp.int32))
    pos0 = jnp.asarray([[0.0, 150.0], [10.0, 150.0]])[:S]
    Bu = PL.n_iter(cfg, B)
    ur_all = rng.uniform(size=(2 * B, max(T - 1, 1)))          # the same stream for both budget semantics; 'iter' uses the first B rows
    ue_all = rng.uniform(size=(2 * B,))
    ur, ue = jnp.asarray(ur_all[:Bu]), jnp.asarray(ue_all[:Bu])
    return PL.make_mcts(cfg, S, T, B, kind)(root, pos0, ur, ue)


def test_node_budget_counts_added_nodes():
    # joint T=1: every iteration adds a node -> identical decision in both budget semantics, B nodes added
    a = _tree(CFG2.replace(budget_mode="iter"), 2, 1, 30)
    b = _tree(CFG2, 2, 1, 30)
    assert int(a["n_nodes"]) == int(b["n_nodes"]) == 31 and int(a["action"]) == int(b["action"])
    # individual T=3 B=40 (7 children): iterations that end at depth-T nodes add nothing in either semantics; the node budget keeps
    # iterating (cap 3 B) so it adds at least as many nodes, and never more than B
    ai = _tree(CFG2.replace(budget_mode="iter"), 1, 3, 40)
    bn = _tree(CFG2, 1, 3, 40)
    assert int(ai["n_nodes"]) <= int(bn["n_nodes"]) <= 41 and int(ai["n_nodes"]) < 41
    # a tree that cannot grow (T=1, 7 children, B=40) stops at the iteration cap instead of looping forever
    cn = _tree(CFG2, 1, 1, 40)
    assert int(cn["n_nodes"]) <= 8


def test_gamma_estimator_threshold():
    NB = M.NB_MAX
    r = jnp.zeros(NB).at[:2].set(jnp.asarray([0.6, 0.6]))
    P = jnp.tile(jnp.eye(4), (NB, 1, 1)).at[0].set(jnp.eye(4) * 10.0).at[1].set(jnp.eye(4) * 2000.0)   # tr = 40 vs 8000 (> c^2/2 = 3200)
    fs = FL.FState(r, jnp.zeros((NB, 4)), P, jnp.asarray(2, jnp.int32))
    _, em_half, _ = FL.estimate(fs)                                    # default / cfg None: r > 0.5 -> both reported
    _, em_def, _ = FL.estimate(fs, M.CFG_O)
    _, em_g, _ = FL.estimate(fs, CFG2.replace(est_rule="gamma"))
    assert int(em_half.sum()) == 2 and int(em_def.sum()) == 2
    assert int(em_g.sum()) == 1            # tr P = 8000 -> Gamma* = 1 -> the vague track is not reported; tr P = 40 -> Gamma* = 0.5025 < 0.6


@pytest.mark.parametrize("flip", [dict(merge_each=True), dict(est_rule="gamma"), dict(kld_mode="info_gain")])
def test_option_one_step_runs(flip):
    from src import algorithm as A
    cfg = CFG2.replace(**flip)
    alg = M.ALGS["Myopic-KLD" if "kld_mode" in flip else "Myopic-GD"]
    run = A.make_chunk_runner(cfg, alg, nan_mode="off")
    carry = A.init_carry(cfg, 2)
    carry, out = run(carry, jnp.arange(2), jnp.arange(2), jnp.asarray([0.1, 2.0]))
    assert np.isfinite(np.asarray(out["gospa_sq"])).all()


def test_checkpoint_layout_key_separates_layouts():
    k = lambda **kw: TR.layout_key(kw.pop("cfg", M.CFG_O), **{**dict(runs=3, clutter=[0.1, 1, 2], m0=0, seed=0, steps=200, truth_set=None), **kw})
    base = k()
    assert base == k()
    for kw in (dict(runs=4), dict(clutter=[1.0]), dict(m0=50), dict(seed=1), dict(steps=100), dict(cfg=M.CFG_P)):
        assert k(**kw) != base, kw


def test_block_inverse_det_matches_lapack():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(7, 4, 4))
    S = jnp.asarray(np.einsum("nab,ncb->nac", X, X) + 0.5 * np.eye(4))
    inv, det = PL._inv_det(S)
    assert np.max(np.abs(np.asarray(inv) - np.linalg.inv(np.asarray(S)))) < 1e-10
    assert np.max(np.abs(np.asarray(det) / np.linalg.det(np.asarray(S)) - 1)) < 1e-12
