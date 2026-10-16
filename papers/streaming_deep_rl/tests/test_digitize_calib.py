"""The figure digitizer calibrates on tick MARKS, not tick-label bbox centers .
Anchors that do not depend on any of our results: the near-random Fig. 4 DQN1 first bin (eps ~0.94) vs the MinAtar
1.0.15 uniform-random Breakout return (: 0.386), and the frozen Fig. 7 Hopper classic AC (Adam 1e-7) vs the
frozen/random-policy Hopper-v4 return (17.0 +- 1.3 init policy, 19.4 +- 1.2 untrained Gaussian, 20.6 +- 1.6 uniform)."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
fitz = pytest.importorskip("pymupdf", reason="needs `uv sync --extra pdf` and the paper PDF from `sh scripts/official/setup.sh`")
import digitize_figs as D  # noqa: E402

if not os.path.exists(D.PDF):
    pytest.skip("paper PDF absent: run `sh scripts/official/setup.sh` (downloads arXiv:2410.14606v2 to external/data/)",
                allow_module_level=True)
sys.path.insert(0, REPO)
from src import eval as ev  # noqa: E402


@pytest.fixture(scope="module")
def marks():
    D.CALIB = "marks"
    return {f: D.digitize(pg, y0, y1, None) for f, (pg, y0, y1, _, _) in D.FIGS.items()}


def test_marks_fit_exactly_and_labels_are_offset(marks):
    for fig, panels in marks.items():
        for gi, v in panels.items():
            assert v["calib"] == "marks"
            assert max(v["fit_resid_pt"]) < 1e-3, (fig, gi, v["fit_resid_pt"])
            off = v["label_offsets"]
            # x labels are centered on their marks; y-label bbox centers sit a constant 0.65-0.69 pt below the marks
            assert max(abs(o) for o in off["x_label_minus_mark_pt"]) < 0.01
            ys = off["y_label_minus_mark_pt"]
            assert 0.6 < min(ys) and max(ys) < 0.75 and max(ys) - min(ys) < 0.01, (fig, gi, ys)
            ay, by = v["pt_per_unit"][1], v["intercept_pt"][1]
            span = abs(max(val for _, val in v["yticks"]) - min(val for _, val in v["yticks"]))
            for m, val in v["yticks"]:   # each tick MARK maps back to its label value
                assert abs((m - by) / ay - val) < 1e-4 * span


def test_export_anchors(tmp_path):
    D.CALIB = "marks"
    res = D.export(str(tmp_path / "dig.json"))
    dqn1_first = ev.grid_value(res["fig4"]["Breakout"]["dqn1"]["points"], 0.125)
    assert abs(dqn1_first - 0.386) < 0.03, dqn1_first          # uniform-random MinAtar Breakout
    for arm in ("classic_ac", "no_obgd"):
        v = ev.grid_value(res["fig7"]["Hopper"][arm]["points"], 0.35)
        assert 15.0 < v < 25.0, (arm, v)                       # frozen / random policy on Hopper-v4


def test_label_mode_reproduces_prefix_bias(tmp_path):
    D.CALIB = "labels"
    try:
        old = D.export(str(tmp_path / "old.json"))
    finally:
        D.CALIB = "marks"
    new = D.export(str(tmp_path / "new.json"))
    d = (ev.grid_value(old["fig4"]["Breakout"]["stream_q"]["points"], 1.0)
         - ev.grid_value(new["fig4"]["Breakout"]["stream_q"]["points"], 1.0))
    assert abs(d - 0.156) < 0.002, d
