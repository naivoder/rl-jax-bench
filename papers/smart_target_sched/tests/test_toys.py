"""Toy checks: the exactly computable toy, the independent numpy oracle, the exact enumeration, the lower-bound two-path
agreement and the invariants of the building blocks.  There is no official code, so the oracle is
(i) exact enumeration, (ii) a literal-equation numpy implementation sharing no code with src/, (iii) plot readings of Figs 2 and 4."""
import numpy as np
import jax, jax.numpy as jnp
import pytest
import model as M
import algorithm as A
import data as D
import oracle as O
import bound as B

U = {"reckless": ((0.9, 0.1), (0.2, 0.8)), "cautious": ((0.95, 0.05), (0.6, 0.4))}


def toy_bank(qct=4.0):
    return M.scalar_bank(["reckless", "cautious"], [qct, qct], [1.0, 1.0])


# ---------- hand-derived toy values
def test_toy_phi_and_indices_to_1e6():
    b = toy_bank()
    P = jnp.ones((2, 1, 1))
    p0 = np.asarray(M.phi(b, P, jnp.zeros(2))).ravel(); p1 = np.asarray(M.phi(b, P, jnp.ones(2))).ravel()
    np.testing.assert_allclose(p0, [2.558000, 2.384000], atol=1e-6)
    np.testing.assert_allclose(p1, [1.393851, 1.221866], atol=1e-6)
    cfg = A.Cfg()
    np.testing.assert_allclose(np.asarray(A.index_myopic(b, P)[0]), [1.164149, 1.162134], atol=1e-6)
    mp, nbad = A.index_whittle(b, P, cfg)
    np.testing.assert_allclose(np.asarray(mp), [1.149438, 1.206511], atol=1e-6)
    assert int(nbad) == 0
    _, g = A.fg_marginal(b, P, M.stat(P), 100, 0.9)
    np.testing.assert_allclose(np.asarray(g), [1.0, 1.0], atol=1e-12)
    # ranking claim of the paper: myopic ranks reckless first, MP index ranks cautious first
    assert p0[0] > 0 and float(A.index_myopic(b, P)[0][0]) > float(A.index_myopic(b, P)[0][1])
    assert float(mp[1]) > float(mp[0])


TOY_TABLE = {  # T: (exact, MP, myopic, TEV)   hand-derived toy values
    2: (5.400066, 5.401880, 5.400066, 5.400066), 3: (8.817281, 8.817281, 9.033796, 9.033796),
    4: (12.259849, 12.259849, 12.322772, 12.322772), 8: (22.840805, 22.840805, 22.939443, 22.939443),
    12: (29.806160, 29.806160, 29.921768, 29.921768)}


@pytest.mark.parametrize("T", [2, 3, 4, 8, 12])
def test_two_target_enumeration_table_to_1e6(T):
    b = toy_bank()
    P0 = jnp.ones((2, 1, 1))
    ex, seq, A_ = A.exact_enumerate(b, P0, T, 1, 0.9, True)
    exp = TOY_TABLE[T]
    assert abs(ex - exp[0]) < 1e-6
    rk = D.rep_keys(0, "G1", 99, 1, 1); _, tie, mode = D.split_rep_keys(rk)
    for pol, e in zip(["whittle", "myopic", "tev"], exp[1:]):
        # TEV ties at P = (1,1): the derivation's table breaks ties by lowest index (tie_first)
        J = A.simulate(b, P0[None], tie, mode, pol, 1, A.Cfg(T=T, guard=False, tie_first=True))["J"][0]
        assert abs(float(J) - e) < 1e-6, (T, pol, float(J), e)
    # no policy beats the exact optimum
    for pol in ["whittle", "myopic", "tev", "passive_cont", "rollout1"]:
        J = A.simulate(b, P0[None], tie, mode, pol, 1, A.Cfg(T=T, guard=False, tie_first=True))["J"][0]
        assert float(J) >= ex - 1e-9


def test_g_plateaus_fig2a():
    # toy derivation / Fig 2(a): reckless, z = 4, Q^CT = 4: g = 1 (P<~0.7), 0.701 (0.7..2.1), 0.369 beyond
    b = M.scalar_bank(["reckless"] * 10, [4.0] * 10, 1.0)
    Ps = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 15.0])
    f, g = A.fg_marginal(b, jnp.asarray(Ps)[:, None, None], jnp.full(10, 4.0))
    np.testing.assert_allclose(np.asarray(g), [1, 1, .701116, .701116, .701116, .369022, .369022, .369022, .369022, .369022], atol=1e-6)
    assert bool(jnp.all(g > 0))            # PCLI1 on these points
    assert abs(float(f[-1]) - 15.0) < 0.2   # Fig 2(a): f reaches ~15 at P = 15 (read precision +-0.3)


def test_fig4_values():
    # Fig 4 plot readings: mp*(1) at Q^CT=40 4.05/2.7; mp*(10) at Q^CT=2 31.6/31.9; mp*(15) at Q^CT=4 55/57
    def mp(typ, P, q):
        b = M.scalar_bank([typ], [q], 1.0)
        return float(A.index_whittle(b, jnp.full((1, 1, 1), P), A.Cfg())[0][0])
    assert abs(mp("reckless", 1, 40) - 4.055) < 0.05 and abs(mp("cautious", 1, 40) - 2.71) < 0.05
    assert abs(mp("reckless", 10, 2) - 31.64) < 0.2 and abs(mp("cautious", 10, 2) - 31.88) < 0.2
    assert abs(mp("reckless", 15, 4) - 55.3) < 1.0 and abs(mp("cautious", 15, 4) - 57.2) < 1.0


# ------------------------------------------------------------------------------------------------ independent numpy oracle
def _rand_scalar_instance(rng, N=4):
    types = list(rng.choice(["reckless", "cautious"], N))
    qct = list(rng.uniform(1.5, 12.0, N)); d = list(rng.choice([1.0, 5.0], N))
    P0 = rng.uniform(0.05, 2.0, N)
    return types, qct, d, P0


def test_phi_matches_oracle_scalar_and_matrix():
    rng = np.random.default_rng(1)
    types, qct, d, P0 = _rand_scalar_instance(rng, 6)
    b = M.scalar_bank(types, qct, d)
    tg = [O.scalar_tgt(*U[t], q, dd) for t, q, dd in zip(types, qct, d)]
    for a in (0, 1):
        got = np.asarray(M.phi(b, jnp.asarray(P0)[:, None, None], jnp.full(6, float(a))))[:, 0, 0]
        ref = [O.phi(t, np.array([[p]]), a)[0, 0] for t, p in zip(tg, P0)]
        np.testing.assert_allclose(got, ref, rtol=1e-13)
    # matrix, both F^CT variants
    for var in ("printed", "standard"):
        bm = M.matrix_bank(["reckless", "cautious"], [3.0, 7.0], [1.0, 1.0], ft_variant=var)
        R0 = rng.uniform(0, 1, (2, 4, 4)); P = np.swapaxes(R0, -1, -2) @ R0
        for a in (0, 1):
            got = np.asarray(M.phi(bm, jnp.asarray(P), jnp.full(2, float(a))))
            for n, (typ, q) in enumerate([("reckless", 3.0), ("cautious", 7.0)]):
                t = O.Tgt([M.f_cv_matrix(), M.f_ct_matrix(variant=var)], [M.q_shape_matrix(), q * M.q_shape_matrix()], *U[typ], M.H_MATRIX, 2 * np.eye(2))
                np.testing.assert_allclose(got[n], O.phi(t, P[n], a), rtol=1e-11, atol=1e-12)


def test_scalar_fast_path_equals_general_matrix_path():
    rng = np.random.default_rng(2)
    types, qct, d, P0 = _rand_scalar_instance(rng, 5)
    b = M.scalar_bank(types, qct, d)
    P = jnp.asarray(P0)[:, None, None]
    for a in (0.0, 1.0):
        x = M.phi(b, P, jnp.full(5, a)); y = M.phi(b, P, jnp.full(5, a), force_general=True)
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-13)


def test_rollout_indices_match_oracle():
    rng = np.random.default_rng(3)
    types, qct, d, P0 = _rand_scalar_instance(rng, 4)
    b = M.scalar_bank(types, qct, d)
    tg = [O.scalar_tgt(*U[t], q, dd) for t, q, dd in zip(types, qct, d)]
    P = jnp.asarray(P0)[:, None, None]
    cfg = A.Cfg(tau=40)
    got = {"whittle": A.index_whittle(b, P, cfg)[0], "myopic": A.index_myopic(b, P)[0], "tev": A.index_tev(b, P)[0],
           "passive_cont": A.index_passive(b, P, cfg)[0]}
    ref = {"whittle": [O.mp_index(t, np.array([[p]]), 40) for t, p in zip(tg, P0)],
           "myopic": [O.myopic_index(t, np.array([[p]])) for t, p in zip(tg, P0)],
           "tev": [O.tev_index(t, np.array([[p]])) for t, p in zip(tg, P0)],
           "passive_cont": [O.passive_cont_index(t, np.array([[p]]), 40) for t, p in zip(tg, P0)]}
    for k in got:
        np.testing.assert_allclose(np.asarray(got[k]), ref[k], rtol=1e-11, err_msg=k)


def test_matrix_index_matches_oracle():
    rng = np.random.default_rng(4)
    bm = M.matrix_bank(["reckless", "cautious"], [3.0, 7.0], [5.0, 1.0])
    R0 = rng.uniform(0, 1, (2, 4, 4)); P = np.swapaxes(R0, -1, -2) @ R0
    tg = [O.Tgt([M.f_cv_matrix(), M.f_ct_matrix()], [M.q_shape_matrix(), q * M.q_shape_matrix()], *U[t], M.H_MATRIX, 2 * np.eye(2), d)
          for t, q, d in [("reckless", 3.0, 5.0), ("cautious", 7.0, 1.0)]]
    got = np.asarray(A.index_whittle(bm, jnp.asarray(P), A.Cfg(tau=25))[0])
    ref = [O.mp_index(t, p, 25) for t, p in zip(tg, P)]
    np.testing.assert_allclose(got, ref, rtol=1e-10)


def test_closed_loop_matches_oracle_trajectory():
    rng = np.random.default_rng(5)
    types, qct, d, P0 = _rand_scalar_instance(rng, 5)
    b = M.scalar_bank(types, qct, d)
    tg = [O.scalar_tgt(*U[t], q, dd) for t, q, dd in zip(types, qct, d)]
    rk = D.rep_keys(0, "G1", 98, 2, 1); _, tie, mode = D.split_rep_keys(rk)
    cfg = A.Cfg(T=14, tau=30, guard=False)
    for pol, fn in [("whittle", lambda t, p: O.mp_index(t, p, 30)), ("myopic", O.myopic_index), ("tev", O.tev_index)]:
        out = A.simulate(b, jnp.asarray(P0)[None, :, None, None], tie, mode, pol, 2, cfg)
        Jref, acts = O.simulate(tg, [np.array([[p]]) for p in P0], 2, 14, fn, tau=30)
        assert abs(float(out["J"][0]) - Jref) < 1e-9 * abs(Jref), pol
        assert np.array_equal(np.asarray(out["acts"])[:, 0], acts), pol


def test_exact_enumeration_matches_oracle_enumeration():
    tg = [O.scalar_tgt(*U["reckless"], 4.0), O.scalar_tgt(*U["cautious"], 4.0), O.scalar_tgt(*U["reckless"], 9.0)]
    P0 = [np.array([[0.8]]), np.array([[1.7]]), np.array([[0.3]])]
    Jo, seq = O.exact_optimum(tg, P0, 4, K=1, allow_idle=True)
    b = M.scalar_bank(["reckless", "cautious", "reckless"], [4.0, 4.0, 9.0], 1.0)
    Jj, _, _ = A.exact_enumerate(b, jnp.array([0.8, 1.7, 0.3])[:, None, None], 4, 1, 0.9, True)
    assert abs(Jo - Jj) < 1e-10 * Jo


def test_reference_trajectory_checksum():
    assert M.reference_trajectory_error() <= M.REF_TOL


# ------------------------------------------------------------------------------------------------ invariants
def test_select_topk_ties_uniform_and_skip_neg():
    idx = jnp.array([1.0, 1.0, 1.0, 0.5])
    ks = jax.random.split(jax.random.PRNGKey(0), 3000)
    m = jax.vmap(lambda k: A.select_topk(idx, k, 1))(ks)
    freq = np.asarray(m.mean(0))
    assert freq[3] == 0 and np.all(np.abs(freq[:3] - 1 / 3) < 0.04)
    assert float(A.select_topk(jnp.array([-1.0, -2.0]), ks[0], 1, True).sum()) == 0   # §III-C variant skips negative indices
    assert float(A.select_topk(jnp.array([0.0, -2.0]), ks[0], 1, True).sum()) == 1     # SS III-C tracks 'nonnegative' indices: an exact zero is tracked
    assert float(A.select_topk(jnp.array([-1.0, -2.0]), ks[0], 1, False).sum()) == 1  # Algorithm 1 fills K
    assert float(A.select_topk(idx, ks[0], 1, False, True)[0]) == 1                    # tie_first -> lowest index


def test_sampled_mode_expectation_equals_averaged_covariance():
    b = M.scalar_bank(["reckless", "cautious"], [4.0, 9.0], 1.0)
    P = jnp.array([1.1, 0.4])[:, None, None]
    for a in (0.0, 1.0):
        act = jnp.full(2, a)
        ks = jax.random.split(jax.random.PRNGKey(7), 20000)
        s = jax.vmap(lambda k: M.phi_sampled(b, P, act, k))(ks)
        np.testing.assert_allclose(np.asarray(s.mean(0)), np.asarray(M.phi(b, P, act)), rtol=0.02)


def test_key_tree_reproducible_and_paired():
    s = D.TABLE_SCENARIOS["t1_q2"]
    r1 = D.rep_keys(0, "E3", s.sid, 1, 5); r2 = D.rep_keys(0, "E3", s.sid, 1, 5)
    assert np.array_equal(np.asarray(r1), np.asarray(r2))
    assert not np.array_equal(np.asarray(D.rep_keys(0, "E3", s.sid, 2, 5)), np.asarray(r1))   # K enters the tree
    p0a = D.sample_P0(s, D.split_rep_keys(r1)[0]); p0b = D.sample_P0(s, D.split_rep_keys(r2)[0])
    assert np.array_equal(np.asarray(p0a), np.asarray(p0b))


def test_pcli1_and_monotone_index_scalar_grid():
    # PCLI1 (g > 0) and PCLI2 (mp* non-decreasing in P) on a coarse grid, both types, Q^CT in {2, 4, 10}
    Ps = np.arange(0.05, 15.0, 0.05)
    for typ in ("reckless", "cautious"):
        for q in (2.0, 4.0, 10.0):
            b = M.scalar_bank([typ] * len(Ps), [q] * len(Ps), 1.0)
            P = jnp.asarray(Ps)[:, None, None]
            mp, nbad = A.index_whittle(b, P, A.Cfg())
            assert int(nbad) == 0
            assert np.all(np.diff(np.asarray(mp)) > -1e-9), (typ, q)


# ------------------------------------------------------------------------------------------------ lower bound
def test_lower_bound_two_paths_agree_and_weak_duality():
    types = ["reckless", "cautious", "reckless", "cautious"]; qct = [4.0] * 4; d = [5.0, 1.0, 5.0, 1.0]
    P0 = np.array([[0.5, 1.2, 0.01, 1.9]])
    a = B.vi_dual(types, qct, d, P0, K=1)
    b = B.threshold_dual(types, qct, d, P0, K=1)
    va, vb = float(a["V_D"][0]), float(b["V_D"][0])
    assert abs(va - vb) / vb < 5e-4, (va, vb)                      # two paths agree to 0.05 % (grid + z-grid error)
    assert float(a["pvis"][0]) < B.P_CAP / 100                        # cap check
    bank = M.scalar_bank(types, qct, d)
    rk = D.rep_keys(0, "G1", 97, 1, 4); _, tie, mode = D.split_rep_keys(rk)
    P0r = jnp.broadcast_to(jnp.asarray(P0)[:, :, None, None], (4, 4, 1, 1))
    for pol in ("whittle", "myopic", "tev"):
        J = A.simulate(bank, P0r, tie, mode, pol, 1, A.Cfg(T=100, guard=False))["J"]
        assert float(J.min()) >= va * (1 - 1e-4), pol                # J^pi >= V^D (weak duality; truncation tail < 1e-4)
