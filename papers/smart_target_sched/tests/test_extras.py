"""Tests for the reproduction's own rules: PCLI1-violation handling, the revisit-horizon heuristic, the per-replication metrics merge."""
import json
import numpy as np
import jax, jax.numpy as jnp
import model as M
import algorithm as A
import data as D


def test_pcli1_violation_rules(monkeypatch):
    b = M.scalar_bank(["reckless"] * 3, [4.0] * 3, 1.0)
    P = jnp.ones((3, 1, 1))
    monkeypatch.setattr(A, "fg_marginal", lambda *a, **k: (jnp.array([2.0, 3.0, -1.0]), jnp.array([1.0, -0.5, 0.0])))
    idx, nbad = A.index_whittle(b, P, A.Cfg(pcli1="dominant"))
    assert int(nbad) == 2 and float(idx[0]) == 2.0 and float(idx[1]) == A.BIG_INDEX and float(idx[2]) == -A.BIG_INDEX
    idx, nbad = A.index_whittle(b, P, A.Cfg(pcli1="literal"))
    assert int(nbad) == 2 and bool(jnp.all(jnp.isfinite(idx))) and float(idx[0]) == 2.0


def test_passive_revisit_equals_fixed_horizon_when_n_over_k_integer():
    s = D.TABLE_SCENARIOS["t3_q2345"]
    bank = D.build_bank(s)
    p0k, tie, mode = D.split_rep_keys(D.rep_keys(0, "G2", s.sid, 2, 4))
    P0 = D.sample_P0(s, p0k)
    cfg = A.Cfg(T=20, guard=False)
    a = A.simulate(bank, P0, tie, mode, "passive_cont_revisit", 2, cfg)["J"]       # ceil(8/2) = 4
    b = A.simulate(bank, P0, tie, mode, "passive_cont_h4", 2, cfg)["J"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)


def test_passive_cont_collapses_to_myopic_ranking_in_single_type_pools():
    # phi^0 is affine in P, so F(P,<0,inf>) - F(P,<1,inf>) = c_type * (tr phi^0(P) - tr phi^1(P)) * d / L with one
    # constant per type => identical decisions to the myopic index when all targets share a type (and Q^CT only enters phi^1's gap).
    b = M.scalar_bank(["reckless"] * 6, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [5.0, 1, 1, 1, 1, 1])
    P = jnp.asarray(np.random.default_rng(0).uniform(0.1, 2.0, 6))[:, None, None]
    my = np.asarray(A.index_myopic(b, P)[0]); pc = np.asarray(A.index_passive(b, P, A.Cfg())[0])
    np.testing.assert_allclose(pc / my, np.full(6, (pc / my)[0]), rtol=1e-10)


def test_metrics_merge(tmp_path):
    import train as T
    meta = {"x": 1}
    T.write_cell(str(tmp_path), "c|K=1|primary", {"J": {"a": np.array([1.0, 2.0])}, "pmax": {"a": np.array([3.0, 4.0])}}, meta)
    T.write_cell(str(tmp_path), "c|K=1|primary", {"J": {"b": np.array([5.0, 6.0])}}, meta)
    m = json.load(open(tmp_path / "seed_1" / "metrics.json"))
    assert m["cells"]["c|K=1|primary"]["J"] == {"a": 2.0, "b": 6.0} and m["cells"]["c|K=1|primary"]["pmax"] == {"a": 4.0}


def test_blind_ablation_policies_ignore_the_reaction_in_the_index_only():
    """ablation arms: whittle_blind<r> prices the index with both mode-probability rows = u^r while the environment keeps the real
    reaction.  If the bank has no reaction (u^0 == u^1) the blind and the plain index policy are the same policy; with a reaction they differ."""
    s = D.TABLE_SCENARIOS["t3_q2"]
    p0k, tie, mode = D.split_rep_keys(D.rep_keys(0, "G2", s.sid, 2, 3))
    P0 = D.sample_P0(s, p0k)
    cfg = A.Cfg(T=15, guard=False)
    bank = D.build_bank(s)
    noreact = bank._replace(U=jnp.stack([bank.U[:, 0], bank.U[:, 0]], axis=1))
    a = A.simulate(noreact, P0, tie, mode, "whittle", 2, cfg)["J"]
    b = A.simulate(noreact, P0, tie, mode, "whittle_blind0", 2, cfg)["J"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)
    c = A.simulate(bank, P0, tie, mode, "whittle", 2, cfg)["J"]
    d = A.simulate(bank, P0, tie, mode, "whittle_blind0", 2, cfg)["J"]
    assert not np.allclose(np.asarray(c), np.asarray(d), rtol=1e-6)


def test_rollout1_base_length_is_a_cfg_field_with_the_small_scale_default():
    assert A.Cfg().r1_base == A.ROLLOUT1_BASE_SLOTS == 30
    s = D.TABLE_SCENARIOS["t1_q2"]
    bank = D.build_bank(s)
    p0k, tie, mode = D.split_rep_keys(D.rep_keys(0, "G2", s.sid, 1, 2))
    P0 = D.sample_P0(s, p0k)
    j30 = A.simulate(bank, P0, tie, mode, "rollout1", 1, A.Cfg(T=6, guard=False))["J"]
    j30b = A.simulate(bank, P0, tie, mode, "rollout1", 1, A.Cfg(T=6, guard=False, r1_base=30))["J"]
    np.testing.assert_allclose(np.asarray(j30), np.asarray(j30b), rtol=0)
