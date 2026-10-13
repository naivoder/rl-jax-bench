"""src/curves.py (extracted from src/eval.py) must reproduce this repo's estimators and stored numbers."""
import json
import os
import sys

import numpy as np

import _helpers  # noqa: F401  (import path)
import eval as ev

import curves  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_first_crossing_matches_eval_both_modes():
    rng = np.random.default_rng(0)
    for _ in range(50):
        t = np.sort(rng.uniform(0, 2e6, 101))
        y = np.cumsum(rng.normal(0.01, 0.05, 101))
        for thr in (0.0, 0.2, 0.5, 5.0):
            a, b = ev.first_crossing(t, y, thr, mode="interp"), curves.first_crossing(t, y, thr)
            assert (np.isnan(a) and np.isnan(b)) or a == b
            a, b = ev.first_crossing(t, y, thr, mode="vertex"), curves.first_crossing(t, y, thr, interpolate=False)
            assert (np.isnan(a) and np.isnan(b)) or a == b


def test_window_and_common_step_match_eval():
    t = np.arange(0, 2.0e6 + 1, 1e4)
    runs = [(t[: len(t) - k], np.sin(t[: len(t) - k] / 3e5 + k)) for k in (0, 7, 30)]
    lo, hi = curves.common_window([r[0] for r in runs], 1e5)
    assert hi == runs[-1][0][-1] and lo == hi - 1e5
    vals = curves.common_step_values(runs, 1e5)
    ref = [ev.value_at({"curve": {"t": r[0].tolist(), "avg100": r[1].tolist()}}, lo, hi) for r in runs]
    assert vals == ref


def test_rate_compare_reproduces_stored_stuck_rate_fisher():
    s = json.load(open(os.path.join(REPO, "results", "recheck_summary.json")))
    st = s["exploratory"]["heavenhell_3/a2c_asym_s/stuck_rate"] if "exploratory" in s else None
    if st is None:                                   # locate the record wherever the reduction put it
        stack = [s]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                if "heavenhell_3/a2c_asym_s/stuck_rate" in o:
                    st = o["heavenhell_3/a2c_asym_s/stuck_rate"]
                    break
                stack.extend(o.values())
    res = curves.rate_compare(st["ours_stuck"], st["ours_n"], st["authors_stuck"], st["authors_n"])
    assert abs(res["fisher_p"] - st["fisher_p"]) < 1e-12
    # corrected pool (20 distinct authors' seeds, 6 stuck): Fisher p 0.38, as in the README
    assert abs(curves.rate_compare(14, 30, 6, 20)["fisher_p"] - 0.38) < 0.005
    assert curves.welch_p([1.0], [1.0, 2.0]) is None
