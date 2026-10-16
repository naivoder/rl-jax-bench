"""Regression: a run's end-point must be compared with a paper figure at the FIGURE's bin stride and at its
grid points. Figure 4 (MinAtar, 5M) is binned at 125k, Figures 7/8 (MuJoCo ablations, 2M) at 50k."""
import json
import os

import numpy as np
import pytest
import yaml

from src import eval as ev

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIG = os.path.join(REPO, "src", "configs", "paper_digitized.json")


@pytest.fixture(scope="module")
def fig():
    if not os.path.exists(DIG):
        pytest.skip("digitized figures absent (scripts/digitize_figs.py)")
    return json.load(open(DIG))["figures"]


def test_fig4_stride_125k(fig):
    for panel, ser in fig["fig4"].items():
        for name, e in ser.items():
            assert ev.figure_stride(e["points"]) == 125_000, (panel, name)


def test_fig7_fig8_stride_50k_matches_hopper_config(fig):
    cfg = yaml.safe_load(open(os.path.join(REPO, "src", "configs", "full_e6_hopper.yaml")))
    for f in ("fig7", "fig8"):
        e = fig[f]["Hopper"]["stream_ac"]
        assert ev.figure_stride(e["points"]) == 50_000 == int(cfg["bin_stride"])


def test_grid_value_is_not_interpolated(fig):
    pts = fig["fig4"]["Breakout"]["stream_q"]["points"]
    assert ev.grid_value(pts, 0.8) is None                      # 0.8M is between grid points
    assert abs(ev.grid_value(pts, 1.0) - 5.058) < 5e-3          # the (0.875M, 1.0M] bin (tick-mark calibration, ;
    # 5.214 under the pre-fix label-center calibration)


def test_report_e3_uses_figure_stride():
    rp = os.path.join(REPO, "results", "full_report.json")
    if not os.path.exists(rp):
        pytest.skip("no full-scale report")
    e = json.load(open(rp))["e3_Breakout"]
    assert e["fig4_stride"] == 125_000
    run = np.array([[1.0, 1.0, 1.0], [130_000.0, 3.0, 1.0], [250_000.0, 5.0, 1.0]])
    assert ev.run_bins(run, 250_000, e["fig4_stride"]).tolist() == [1.0, 4.0]
