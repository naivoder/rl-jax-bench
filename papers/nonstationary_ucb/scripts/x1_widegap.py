"""X1 cause test (labelled DIAGNOSTIC): is D-UCB's out-of-band rate exponent a pre-asymptotic effect?

    uv run python scripts/x1_widegap.py     -> runs/x1_widegap/a/T*_U4/seed_*/metrics.json

Hypothesis (written before running): on the X1 instance the Delta = 0.1 arm needs about 16 xi log n / Delta^2 ~ 7300
discounted plays to be resolved by D-UCB's 2B padding, more than the Remark-3 memory 1/(1-gamma) = 4 sqrt(T/U)
(= 2000 at T = 1e6), so regret is still near-linear at every tested T. On a wide-gap instance (base (0.9, 0.1, 0.1),
swapped (0.1, 0.9, 0.1), min gap 0.8 -> ~110 plays needed) the asymptotic regime starts near T = 1e4, so the grid-a
slope over T in [1e4, 1e6] should fall inside the pre-declared band [0.45, 0.70] for both D-UCB and SW-UCB.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
import src  # noqa: E402,F401
from src import sweep  # noqa: E402
from src.common import check_device  # noqa: E402

check_device()
cfg = sweep.load_cfg()
cfg["x1"]["base"], cfg["x1"]["swapped"] = [0.9, 0.1, 0.1], [0.1, 0.9, 0.1]  # DIAGNOSTIC wide-gap instance
orig = sweep.x1_tunings
sweep.x1_tunings = lambda T, U, x: {k: v for k, v in orig(T, U, x).items() if k in ("ducb", "swucb")}
for T in (10_000, 30_000, 100_000, 300_000, 1_000_000):
    sweep.run_x1_point(cfg, "a", T, 4, [0, 1, 2, 3, 4], 1000, os.path.join(ROOT, "runs", "x1_widegap"),
                       os.path.join(ROOT, "runs", "x1_ckpt_widegap"))
