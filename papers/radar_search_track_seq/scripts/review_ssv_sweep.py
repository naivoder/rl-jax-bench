"""review resolution/ (descriptive, CPU): the search-reward ratios SR(Coverage)/SR(Random) and
SR(Static)/SR(Random) and the ordering Coverage > Random > Static under the unstated scan-value conventions:
gamma in {0.01 (primary), 0.024 (the fit to the paper's 0.38)} x sigma0 in {radial 1.84 deg (primary), per-axis
2.30 deg} x SSV evaluated before (primary) / after the push (ages 0..N_SV-1), N_SV = 100, on the 100 final-evaluation
scenarios (Random: action seed 0). Paper (§3): Coverage/Random 0.380, Static/Random ~191 (README).

    JAX_PLATFORMS=cpu taskset -c 6-7 python scripts/review_ssv_sweep.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.common as C  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

from src.configs.config import Config, K_EVAL_SCEN  # noqa: E402
from src import baselines as B, eval as E  # noqa: E402
sys.path.insert(0, os.path.join(C.REPO, "scripts"))
from full_baselines import run  # noqa: E402


def main():
    C.warn("SSV convention sweep on CPU BY DESIGN")
    C.check_device()
    cfg_all = Config()
    sk = E.scenario_keys(jax.random.PRNGKey(cfg_all.seed_root), K_EVAL_SCEN, cfg_all.n_eval_scenarios)
    res = {}
    for gam in (0.01, 0.024):
        for s0 in ("radial", "axis"):
            for a0 in (False, True):
                c2 = replace(cfg_all.env, gospa_metrics=False, gamma_sv=gam, n_sv=100, sigma0_mode=s0, ssv_age0=a0)
                row = {}
                for name, mk in (("static", B.make_static), ("random", B.make_random), ("coverage", B.make_coverage)):
                    pi, po = mk(c2)
                    row[name] = float(np.mean(run(pi, po, sk, c2, 0)["sr"]))
                row.update(cov_over_rnd=row["coverage"] / row["random"], static_over_rnd=row["static"] / row["random"],
                           order_ok=bool(row["coverage"] > row["random"] > row["static"]), sigma0=c2.sigma0_deg)
                key = f"g{gam}_{s0}_{'after' if a0 else 'before'}"
                res[key] = row
                C.metric("ssv sweep", key=key, **row)
                C.write_json(os.path.join(C.REPO, "runs", "review", "ssv_sweep.json"), res)


if __name__ == "__main__":
    main()
