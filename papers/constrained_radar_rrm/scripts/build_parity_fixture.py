"""Copy the reference outputs needed by tests/test_official_parity.py into tests/fixtures/official_oracle.npz.

Sources (all under $OFFICIAL_ROOT/fixtures, default external/official/fixtures):
  oracle_env_outputs.npz  -- written by oracle_env_harness.py (filterpy 1.4.5 / scipy 1.18.1, own venv)
  env_rollout_oa_s_ref_table6.npz -- our recorded OA decision states (only the 200 states SLSQP solved are kept)
Only reference OUTPUTS and the decision-state inputs are committed; no reference code.
"""
import os

import numpy as np

SRC = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "official")), "fixtures")
DST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")


def main():
    o = np.load(f"{SRC}/oracle_env_outputs.npz")
    oa = np.load(f"{SRC}/env_rollout_oa_s_ref_table6.npz")
    pick = o["oa_slots"]
    out = dict(filterpy_version=o["filterpy_version"], scipy_version=o["scipy_version"],
               ea_c=o["ea_s_ref_c"], ea_U=o["ea_s_ref_U"], oa_c=o["oa_s_ref_table6_c"], oa_U=o["oa_s_ref_table6_U"],
               dare_tau=o["dare_tau"], dare_r=o["dare_r"], dare_c=o["dare_c"],
               oa_r_hat=oa["r_hat"][pick], oa_w=oa["w_pre"][pick], oa_act=oa["act_view"][pick],
               oa_tau_slsqp=o["oa_tau"], oa_obj_slsqp=o["oa_obj"], pd_snr_db=o["pd_snr_db"], pd_exact=o["pd_exact"])
    os.makedirs(DST, exist_ok=True)
    p = os.path.join(DST, "official_oracle.npz")
    np.savez_compressed(p, **out)
    print("wrote", p, os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
