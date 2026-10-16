"""Review-pass regression tests: DQN1 Adam epsilon, E8 metric alignment, sensitivity arms switch only the
optimizer."""
import numpy as np

from src import eval as ev
from src import train as tr


def test_dqn1_uses_cleanrl_adam_eps():
    assert tr.lane_cfg("q", "dqn1")["adam_eps"] == 1e-8
    for arm in ("stream_q", "classic_q", "classic_q_keepnorm"):
        assert tr.lane_cfg("q", arm)["adam_eps"] == 1e-4
    assert tr.lane_cfg("ac", "no_obgd")["adam_eps"] == 1e-4 and tr.lane_cfg("td", "classic_td")["adam_eps"] == 1e-4
    assert tr.lane_cfg("q", "dqn1@adam_eps=1e-4")["adam_eps"] == 1e-4     # the full-scale dqn1 setting, reproducible


def test_keepnorm_arms_switch_only_the_optimiser():
    for kind, base, arm in (("q", "stream_q", "classic_q_keepnorm"), ("td", "stream_td", "classic_td_keepnorm")):
        a, b = tr.lane_cfg(kind, base), tr.lane_cfg(kind, arm)
        assert {k for k in a if a[k] != b[k]} == {"opt", "lr"}


def test_ett_alignment():
    rng = np.random.default_rng(0)
    cum = rng.random(20_000)
    G = ev.discounted_returns(cum, 0.99)
    pred = np.append(G[1:], 0.0)          # a perfect predictor of v(S_{j+1}) = G[j+1]
    s = ev.ett_summary(pred, G, cum, window=5000)
    assert s["mse_last"] < 1e-20 and s["mse_first"] < 1e-20
    assert s["mse_first_official"] > 1e-3  # the one-step-off official alignment penalises it
