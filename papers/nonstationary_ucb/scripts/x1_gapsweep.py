"""Recheck RC4 (README, pre-declared; EXTENSION, post-hoc): X1 grid a on two
narrower-than-wide-gap instances, to test the resolvability criterion behind 'rates absent by T = 1e6'.

    uv run python scripts/x1_gapsweep.py   -> runs/x1_gapsweep/d{0.2,0.4}/a/T*_U4/seed_*/metrics.json

Resolvability ratio (paper Theorems 1/7, Step 1): SW-UCB tau / A(tau), A = 4 xi log tau / Delta^2; D-UCB
(1 - gamma)^-1 / A(gamma), A = 16 xi log n / Delta^2 (n = 1 / (1 - gamma)). Declared criterion: the 5-point slope over
T in [1e4, 1e6] lies in [0.45, 0.70] iff the ratio at T = 1e5 is >= 1.
  Delta = 0.2: base (0.5, 0.3, 0.3), swapped (0.5, 0.9, 0.3)  -> SW 2.6 (in band), D 0.41 (above band)
  Delta = 0.4: base (0.5, 0.1, 0.1), swapped (0.5, 0.9, 0.1)  -> SW 10.2 (in band), D 1.6 (in band)
Everything else as X1 grid a (Upsilon = 4, xi = 0.6, Remark-3 gamma, Remark-9 tau, printed 2B); seeds 0-4 x 400.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
import src  # noqa: E402,F401
from src import sweep  # noqa: E402
from src.common import check_device  # noqa: E402

INSTANCES = {"d0.2": ([0.5, 0.3, 0.3], [0.5, 0.9, 0.3]), "d0.4": ([0.5, 0.1, 0.1], [0.5, 0.9, 0.1])}
check_device()
orig = sweep.x1_tunings
sweep.x1_tunings = lambda T, U, x: {k: v for k, v in orig(T, U, x).items() if k in ("ducb", "swucb")}
for name, (base, swapped) in INSTANCES.items():
    cfg = sweep.load_cfg()
    cfg["x1"]["base"], cfg["x1"]["swapped"] = base, swapped
    for T in (10_000, 30_000, 100_000, 300_000, 1_000_000):
        sweep.run_x1_point(cfg, "a", T, 4, [0, 1, 2, 3, 4], 400, os.path.join(ROOT, "runs", "x1_gapsweep", name),
                           os.path.join(ROOT, "runs", "x1_ckpt_gapsweep", name))
