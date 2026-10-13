"""The final-metric code recomputes the authors'-log reference numbers in README (regression)."""
import os

import numpy as np
import pytest

from src.eval import AUTHOR_LOGS, author_final, judge, judge_detail, mean_se

needs_logs = pytest.mark.skipif(not os.path.isdir(AUTHOR_LOGS), reason="needs the authors' published run logs under "
                                "$OFFICIAL_ROOT/numeric_results/extracted/logs_txt: run sh scripts/official/setup.sh")


@pytest.mark.parametrize("cell,mean,se", [
    ("pomdp/Pendulum/V/sac_lstm/len--1/*/*/*/oar/*/progress.csv", -168.7, 2.9),
    ("pomdp/Pendulum/V/sac_lstm_shared/len--1/*/*/*/oar/*/progress.csv", -443.5, 83.2),
    ("pomdp/Pendulum/V/td3_lstm_shared/len--1/*/*/*/oar/*/progress.csv", -1593.8, 63.4),
    ("meta/PointRobotSparse-v0/td3_lstm/len-64/*/*/*/or/*/progress.csv", 94.37, 0.16),
    ("meta/PointRobotSparse-v0/td3_lstm_shared/len-64/*/*/*/or/*/progress.csv", 0.52, 0.23),
])
@needs_logs
def test_author_reference_numbers(cell, mean, se):
    vals = author_final(cell)
    m, s = mean_se(vals)
    assert len(vals) == 4 and abs(m - mean) < 0.051 and abs(s - se) < 0.051, (m, s)


@pytest.mark.parametrize("cell,env,mean,sd", [
    # Table 7 'td3-lstm-64-ord' = 94.2 +- 0.6 (sd over 4 seeds); the official scripts give 94.151 / 0.569
    ("meta/PointRobotSparse-v0/td3_lstm/len-64/*/*/*/or/*/progress.csv", "semicircle", 94.151, 0.569),
    # official plot_csv rundown on the Pendulum-V logs (recheck, run in the official venv)
    ("pomdp/Pendulum/V/sac_lstm/len--1/*/*/*/oar/*/progress.csv", "pendulum", -168.882, 4.49),
    ("pomdp/Pendulum/V/sac_lstm_shared/len--1/*/*/*/oar/*/progress.csv", "pendulum", -432.319, 149.67),
])
@needs_logs
def test_paper_metric_reproduces_official_rundown(cell, env, mean, sd):
    from src.eval import PAPER_METRIC_ARGS, author_runs, paper_metric
    vals = np.array([paper_metric(c[:, 0], c[:, 1], **PAPER_METRIC_ARGS[env]) for c in author_runs(cell)])
    assert len(vals) == 4 and abs(vals.mean() - mean) < 2e-3 and abs(vals.std(ddof=1) - sd) < 6e-3, (vals.mean(), vals.std(ddof=1))
    if env == "semicircle":
        assert round(vals.mean(), 1) == 94.2 and round(vals.std(ddof=1), 1) == 0.6   # the printed Table 7 entry


def test_paper_metric_smoothing_and_truncation():
    from src.eval import paper_metric
    x = np.arange(1, 11) * 10.0
    y = np.arange(1, 11, dtype=float)
    assert paper_metric(x, y, 10, 10, 1) == pytest.approx(9.5)            # no smoothing: mean of the last 2 of 10
    assert paper_metric(x, y, 10, 10, 3) == pytest.approx(8.5)            # trailing window 3: (8.0 + 9.0) / 2
    assert paper_metric(x, y, 10, 10, 1, max_x=50) == pytest.approx(5.0)  # truncate to 5 points, last int(4) = point 5


def test_judge():
    assert judge(94.0, 94.37, 0.6) == "MATCH" and judge(80, 94.37) == "CLOSE" and judge(50, 94.37) == "DIVERGENT"
    # since the review the normalised scale no longer decides; -200 vs -170 is 17.6 % raw -> CLOSE
    assert judge(-200, -170, lo=-1300, hi=-140) == "CLOSE"
    d = judge_detail(-200.0, [-165.0, -170.0, -172.0, -173.0], lo=-1228.0, hi=0.0)
    assert d["verdict"] == "DIVERGENT" or d["verdict"] == "CLOSE"
    assert d["gap_in_ref_se"] < -10 and not d["match_by_spread_only"]
    d = judge_detail(-512.0, [-300.0, -400.0, -500.0, -574.0])
    assert d["verdict"] == "MATCH" and d["match_by_spread_only"]
