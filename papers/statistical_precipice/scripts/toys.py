"""toy checks (toy correctness, the README-D11): every ground-truth check of src/eval.py at full size.
Writes runs/toys/metrics.json; PASS iff every block passes.

    python scripts/toys.py
"""
import os
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax  # noqa: E402
from rl_jax_bench.jaxlog import Timer, check_device, info, metric, write_json  # noqa: E402

from src import algorithm as A  # noqa: E402
from src import eval as E  # noqa: E402


def main():
    check_device()
    T = Timer()
    T_all = Timer()
    root = A.branch_key(0, A.EXP_IDS["toys"])
    k = lambda i: jax.random.fold_in(root, i)  # noqa: E731
    out = {}
    out["toy_d9"] = E.toys_toy(k(1), B=200_000)
    info("toy", t=T.lap())
    out["closed_forms_d4_2"] = E.toys_closed_forms(k(2), n_cases=30, B=100_000)
    info("closed forms", t=T.lap())
    out["poi_identities_d2_5"] = E.toys_poi_identities(k(3), n_cases=200)
    out["profile_unbiased_d3"] = E.toys_profile_unbiased(k(4), reps=20_000, N=3)
    out["area_identities_fig8"] = E.toys_area_identities(k(5), n_cases=50)
    out["lift_identity_d5_2"] = E.toys_lift_identity(k(6))
    info("identities", t=T.lap())
    out["iqm_estimand_d10"] = E.toys_iqm_estimand(k(7), N=100, M=26, reps=4000)
    out["coverage_d10"] = E.toys_coverage(k(8), Ns=(3, 10), M=26, sets=4000, inner_B=2000)
    out["coverage_procedure_fig6"] = E.toys_coverage_procedure(k(9), K=3, M=26, pool_runs=4000, sets=2000, inner_B=1000)
    info("coverage", t=T.lap())
    flags = {"toy_points": out["toy_d9"]["points_pass"], "toy_boot": out["toy_d9"]["boot_pass"],
             "toy_pct": out["toy_d9"]["pct_pass"]}
    for kk, v in out.items():
        if "pass_" in v:
            flags[kk] = v["pass_"]
    out["flags"] = flags
    out["PASS"] = all(flags.values())
    out["seconds"] = T_all.lap()
    write_json(os.path.join(REPO, "runs", "toys", "metrics.json"), out)
    metric("toys", PASS=out["PASS"], **flags)


if __name__ == "__main__":
    main()
