"""Reference-code parity tests. The authors released NO code (searched 2026-10-19: paper text, arXiv
2606.05526 abs page + 'Code, Data, Media' tab, GitHub repository/user search, HF papers API), so these compare OUR implementation with labelled second-best reference code run in
its own venv (filterpy 1.4.5, scipy 1.18.1; scripts/official/setup.sh builds it). Only the
reference OUTPUTS are committed (tests/fixtures/official_oracle.npz, built by scripts/build_parity_fixture.py from
scripts/official/oracle_env_harness.py). The two other oracle tests (filterpy on synthetic sequences, CleanRL DDPG update) stay in
tests/test_oracle.py.

  P1  our environment, re-run live on the Case I replica (EA, declared S_ref, oracle key; 1e4 slots), must
      reproduce filterpy's re-filtered per-slot, per-track position traces c_n and the utility U (eq. 19) computed
      from the same measurement stream.                         tolerance: rel 1e-9 (observed 1e-13)
  P2  same for the OA arm (S_ref_table6, 1e4 slots; more tracks).  rel 1e-9
  P3  our Kalata closed form c_ss vs scipy solve_discrete_are steady-state posterior trace on a
      6 x 5 (tau, r) grid.                                     rel 1e-8 (observed ~1e-11)
  P4  our production OA solver (allocate_newton via baselines.oa) vs scipy SLSQP on 200 recorded OA decision
      states with >= 2 tracks: allocation and objective.        |dtau| / D <= 1e-5 (observed 1e-7); objective
                                                               ratio - 1 <= 1e-8
  P5  our Shnidman LUT P_d vs scipy.stats.ncx2 exact single-pulse Swerling-0 square-law P_d at P_fa = 1e-3 over
      -5..20 dB.                                               abs <= 0.03 (Shnidman's closed form is an
      approximation, ~0.02 here, plus the LUT's 0.01 quantisation; the toy test of P_d bounds the formula alone at 0.02)
"""
import os

import numpy as np
import pytest

from src import common  # noqa: F401
import jax
import jax.numpy as jnp

from src import baselines as bl
from src import physics as ph
from src.common import EnvCfg, load_calibration, load_yaml

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "official_oracle.npz")
REL_C = 1e-9


@pytest.fixture(scope="module")
def ref():
    return np.load(FIX)


def _rollout(policy, s_ref_key, T):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import oracle_dump_env
    rec, _ = oracle_dump_env.record(policy, s_ref_key, T)
    return rec


@pytest.mark.parametrize("policy,s_ref_key,tag", [("ea", "s_ref", "ea"), ("oa", "s_ref_table6", "oa")])
def test_env_cost_vs_filterpy(ref, policy, s_ref_key, tag):
    c_ref, U_ref = ref[f"{tag}_c"], ref[f"{tag}_U"]
    rec = _rollout(policy, s_ref_key, c_ref.shape[0])
    assert rec["newk"].sum() >= 4 and rec["upd"].sum() > 5000          # the stream exercises births and updates
    rel_c = np.max(np.abs(rec["c"] - c_ref) / np.maximum(np.abs(c_ref), 1.0))
    rel_U = np.max(np.abs(rec["U"] - U_ref) / np.maximum(np.abs(U_ref), 1.0))   # U = 0 in empty slots
    print(f"P{1 if tag == 'ea' else 2} filterpy {ref['filterpy_version']}: max rel c {rel_c:.2e}, max rel U {rel_U:.2e}, "
          f"mean U ours {rec['U'].mean():.3f} ref {U_ref.mean():.3f}")
    assert rel_c <= REL_C and rel_U <= REL_C


def test_css_vs_scipy_dare(ref):
    cfg = EnvCfg.from_yaml(load_yaml())
    snr0 = load_calibration()["snr0"]
    ours = np.array([[float(ph.c_ss(t, r, snr0, cfg)) for r in ref["dare_r"]] for t in ref["dare_tau"]])
    rel = np.max(np.abs(ours / ref["dare_c"] - 1))
    print(f"P3 c_ss (Kalata) vs scipy {ref['scipy_version']} DARE: max rel {rel:.2e}")
    assert rel <= 1e-8


def test_oa_vs_scipy_slsqp(ref):
    cfg = EnvCfg.from_yaml(load_yaml())
    snr0 = jnp.float64(load_calibration()["snr0"])
    act = jnp.asarray(ref["oa_act"])
    r_hat = jnp.asarray(ref["oa_r_hat"])
    w = jnp.asarray(ref["oa_w"])

    def one(a, r, ww):
        view = type("V", (), {})()
        view.act, view.r_hat, view.w, view.factor_hat = a, jnp.where(a, r, 1.0e4), ww, jnp.ones_like(r)
        return bl.oa(cfg, view, 0.9, snr0)

    tau = np.asarray(jax.jit(jax.vmap(one))(act, r_hat, w))
    D = 0.9 * cfg.T0
    dtau = np.max(np.abs(tau - ref["oa_tau_slsqp"])) / D
    obj = np.array([sum(float(w[i, k]) * float(ph.c_ss(tau[i, k], float(r_hat[i, k]), float(snr0), cfg))
                        for k in range(cfg.K) if bool(act[i, k])) for i in range(tau.shape[0])])
    ratio = np.max(obj / ref["oa_obj_slsqp"] - 1)
    print(f"P4 OA Newton vs SLSQP on {tau.shape[0]} states: max |dtau|/D {dtau:.2e}, max obj ratio - 1 {ratio:.2e}")
    assert np.allclose(tau.sum(-1), D, rtol=1e-12)
    assert dtau <= 1e-5 and ratio <= 1e-8


def test_pd_lut_vs_ncx2_exact(ref):
    lut = ph.make_lut(1e-3)
    snr = 10 ** (ref["pd_snr_db"] / 10)
    ours = np.asarray(ph.pd_lookup(jnp.asarray(snr), lut))
    exact = np.clip(ref["pd_exact"], 0.01, 0.99)                        # the paper's LUT range (§III-B)
    err = np.max(np.abs(ours - exact))
    print(f"P5 P_d LUT vs ncx2 exact: max abs {err:.4f}")
    assert err <= 0.03
