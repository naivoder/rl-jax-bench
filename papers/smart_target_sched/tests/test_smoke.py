"""Smoke tests (< 30 s, CPU): config/sources agreement, shapes of one vmapped step, one closed-loop run, metrics schema."""
import os
import numpy as np
import yaml
import jax, jax.numpy as jnp
import model as M
import algorithm as A
import data as D

SRC = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "src", "configs", "sources.yaml")))


def test_config_matches_sources():
    c = SRC["common"]
    assert M.BETA == c["beta"]["value"] and M.TAU == c["tau"]["value"] and M.T_HORIZON == c["T"]["value"]
    assert M.N_MC == c["n_mc"]["value"] and M.H_COST == c["h"]["value"] and M.TOP_SEED == c["top_seed"]["value"]
    s = SRC["scalar"]
    assert (M.F_CV, M.F_CT, M.Q_CV, M.H_SCALAR, M.R_SCALAR) == (s["F_CV"]["value"], s["F_CT"]["value"], s["Q_CV"]["value"], s["H"]["value"], s["R"]["value"])
    t = SRC["target_types"]
    assert M.U_RECKLESS == (tuple(t["reckless"]["u0"]), tuple(t["reckless"]["u1"]))
    assert M.U_CAUTIOUS == (tuple(t["cautious"]["u0"]), tuple(t["cautious"]["u1"]))
    m = SRC["matrix"]
    assert abs(M.OMEGA_RAD - m["omega_rad"]["value"]) < 1e-15 and M.TS == m["Ts"]["value"]
    e = SRC["experiments"]
    assert list(D.TABLE_SCENARIOS["t1_q29"].qct) == [2, 3, 4, 5, 6, 7, 8, 9] == e["E3_table1"]["Q_CT_sets"]["value"][1]
    assert list(D.TABLE_SCENARIOS["t3_q2345"].qct) == e["E5_table3"]["Q_CT_sets"]["value"][1]
    assert list(D.TABLE_SCENARIOS["t3_q2"].d) == e["E5_table3"]["d"]["value"]
    assert list(D.TABLE_SCENARIOS["t1_q2"].d) == e["E3_table1"]["d"]["value"]


def test_phi_shapes_and_positive_definite():
    for kind, bank, L in [("s", M.scalar_bank(["reckless", "cautious"] * 2, [2, 3, 4, 5], 1.0), 1),
                          ("m", M.matrix_bank(["reckless", "cautious"] * 2, [2, 3, 4, 5], 1.0), 4)]:
        P = jnp.broadcast_to(jnp.eye(L), (4, L, L)) * 0.7
        for a in (0.0, 1.0):
            Pn = M.phi(bank, P, jnp.full(4, a))
            assert Pn.shape == (4, L, L) and np.all(np.linalg.eigvalsh(np.asarray(Pn)) > 0)
        assert M.stat(P).shape == (4,)


def test_one_closed_loop_run_all_arms():
    s = D.TABLE_SCENARIOS["t1_q29"]
    bank = D.build_bank(s)
    rk = D.rep_keys(0, "G2", s.sid, 2, 3)
    p0k, tie, mode = D.split_rep_keys(rk)
    P0 = D.sample_P0(s, p0k)
    assert P0.shape == (3, 8, 1, 1) and bool(jnp.all(P0 > 0)) and bool(jnp.all(P0 <= 2))
    cfg = A.Cfg(T=12, tau=20)
    out = {}
    for pol in ["whittle", "myopic", "tev", "passive_cont", "passive_cont_h10", "random", "rollout1"]:
        o = A.simulate(bank, P0, tie, mode, pol, 2, cfg)
        assert o["J"].shape == (3,) and bool(jnp.all(jnp.isfinite(o["J"])))
        assert o["acts"].shape == (12, 3, 8) and np.all(np.asarray(o["acts"]).sum(-1) == 2)   # exactly K tracked every slot
        out[pol] = o["J"]
    assert bool(jnp.all(out["rollout1"] <= out["tev"] * 1.05))      # search reference is not worse than a greedy rule by much
    assert float(out["whittle"].mean()) < float(out["random"].mean())


def test_matrix_one_step():
    s = D.TABLE_SCENARIOS["t4_mixed"]
    bank = D.build_bank(s)
    p0k, tie, mode = D.split_rep_keys(D.rep_keys(0, "G2", s.sid, 1, 2))
    P0 = D.sample_P0(s, p0k)
    assert P0.shape == (2, 8, 4, 4) and D.min_eig(P0) > 1e-9
    o = A.simulate(bank, P0, tie, mode, "whittle", 1, A.Cfg(T=3, tau=10))
    assert bool(jnp.all(jnp.isfinite(o["J"])))
