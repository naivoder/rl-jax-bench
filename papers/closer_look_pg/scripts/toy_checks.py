"""Toy checks: the exactly computable toys at full size through the real instrument code; writes runs/toys/toys.json.

Usage: python scripts/toy_checks.py   (GPU if available; ~1 min)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import common  # noqa: E402
from src.common import info, write_json  # noqa: E402

import jax  # noqa: E402

from src import toys  # noqa: E402


def main():
    common.check_device()
    root = jax.random.PRNGKey(20200101)
    k = lambda i: jax.random.fold_in(root, i)
    res = {
        "bandit_d1_s1": toys.bandit_estimator(k(1), 1_000_000, d=1.0, s=1.0),
        "bandit_d1_s05": toys.bandit_estimator(k(2), 1_000_000, d=1.0, s=0.5),
        "bandit_landscape": toys.bandit_landscape(k(3), 1_000_000),
        "cosine_model": toys.cosine_model(k(4), d=10_000, n_half=5000.0, reps=200),
        "mdp": toys.mdp_pipeline(k(5), n_traj=1_000_000),
        "gae_identity": toys.gae_identity(k(6), n_traj=64, T=400),
        "mre": toys.mre_toys(k(7), N=4_000_000),
    }
    flags = {f"{name}.{kk}": v for name, r in res.items() for kk, v in r.items() if kk.startswith("pass_")}
    ok = all(flags.values())
    for kk, v in flags.items():
        info("toys", check=kk, passed=v)
    write_json(os.path.join(common.REPO, "runs", "toys", "toys.json"), {"pass": ok, "flags": flags, "results": res,
                                                                          **common.git_info()})
    print("TOY CHECKS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
