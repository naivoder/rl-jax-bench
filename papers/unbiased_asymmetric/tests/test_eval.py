"""Reduction logic (src/eval.py) on synthetic records: judgment rule, window helpers, no literal fallbacks."""
import numpy as np
import pytest

import _helpers  # noqa: F401  (import path)
import eval as ev


def test_judge_uses_max_of_spread_and_precision():
    # tol_abs = max(spread, prec), not their sum (Shopping-5 react-4: |diff| 0.450 > max(0.235, 0.35))
    assert ev.judge(2.463, 2.013, 0.35, 0.235) == "DIVERGENT"
    assert ev.judge(2.30, 2.013, 0.35, 0.235) == "MATCH"            # |diff| 0.287 <= 0.35
    assert ev.judge(0.03, 0.0, 0.02, 0.05) == "MATCH"                # near-zero absolute rule, spread wins
    assert ev.judge(0.15, 0.0, 0.02, 0.05) == "CLOSE"                # <= 4 tol_abs


def test_value_at_common_window():
    t = np.arange(0, 2.0e6 + 1, 1e4)
    rec = {"curve": {"t": t.tolist(), "avg100": (t / 1e6).tolist()}}
    v = ev.value_at(rec, 1.0e6, 1.1e6)                               # points 1.01 ... 1.10
    assert abs(v - np.mean(np.arange(101, 111) / 100)) < 1e-12
    assert np.isnan(ev.value_at(rec, 3e6, 4e6))


def test_paper_window_raises_instead_of_literal_fallback():
    # a missing paper curve must raise, never fall back to a read-off constant
    with pytest.raises(KeyError):
        ev.paper_window("memory_four_rooms_7x7", "a2c_asym_hs", 0.0, 1e6)
    m, h = ev.paper_window("heavenhell_3", "a2c_asym_hs", 9.0e6, 1.0e7)
    assert 0.95 < m <= 1.01 and h is not None


def test_no_hardcoded_official_strings():
    src = open(ev.__file__).read()
    assert "OFFICIAL_AT" not in src and "authors' code 1.000" not in src


def test_first_crossing_interpolates_and_keeps_vertex_mode():
    # linear interpolation between the vertices around the crossing; 'vertex' = pre-fix
    t, y = [0.0, 1.0, 2.0, 3.0], [0.0, 0.2, 0.6, 1.0]
    assert abs(ev.first_crossing(t, y, 0.5) - 1.75) < 1e-12
    assert ev.first_crossing(t, y, 0.5, mode="vertex") == 2.0
    assert ev.first_crossing(t, y, 0.0) == 0.0
    assert np.isnan(ev.first_crossing(t, y, 1.5))


def test_paper_curves_gridline_calibrated():
    # the default curves are gridline-calibrated; the label fit sits 0.067 lower on Shopping-5
    import json
    import os
    g = json.load(open(ev.PAPER_CURVES))
    lab = json.load(open(os.path.join(ev.REPO, "src", "configs", "paper_fig3_curves_labelcal.json")))
    assert g.get("_calibration") == "gridlines"
    d = np.mean(np.asarray(g["shopping_5"]["a2c_react_4"]["mean"]) - np.asarray(lab["shopping_5"]["a2c_react_4"]["mean"]))
    assert 0.05 < d < 0.09
