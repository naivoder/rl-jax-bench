"""There is no gradient anywhere in this paper (index policies and EXP3.S), so check_grads has nothing to check.

Per the shared-context rule, this file instead tests the estimators' closed forms and statistics:
the discounted/windowed recursions against direct sums (D-2/D-3/D-7, D-4), the Toy-check hand-computed
indices (G1-a), tie-breaking, the D-6 variance (G1-b, Monte Carlo), the padding/confidence constants and
theorem constants against the paper's formulas (G1-c), exact trajectories against an independent
pure-Python reference (G1-d and a stochastic replay), the EXP3.S log-space update, and Theorem 18 (G1-e).

    JAX_PLATFORMS=cpu uv run python -m pytest tests/test_grads.py -q
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import src  # noqa: E402,F401
import numpy as np  # noqa: E402

from src import data  # noqa: E402
from src import eval as ev  # noqa: E402


def test_g1a_hand_indices():
    r = ev.check_g1a_indices()
    assert r["pass"], r
    assert r["exact_tie_swucb"]


def test_g1a_tiebreak():
    r = ev.check_g1a_tiebreak(n_keys=10_000)
    assert r["pass"], r


def test_recursions_equal_direct_sums():
    r = ev.check_recursions(n_hist=300)
    assert r["pass"], r


def test_g1b_variance_closed_form():
    r = ev.check_g1b_variance(n_draws=200_000)
    assert r["pass"], r


def test_g1c_constants():
    r = ev.check_g1c_constants()
    assert r["pass"], r["checks"]


def test_g1d_deterministic_reference():
    assert ev.check_g1d_deterministic(tie="lowest")["pass"]
    assert ev.check_g1d_deterministic(tie="random")["pass"]


def test_stochastic_replay_reference():
    r = ev.check_stochastic_reference(T=2000)
    assert r["pass"], {k: v for k, v in r.items() if k != "pass" and not v["identical"]}


def test_exp3s_logspace():
    r = ev.check_exp3s_logspace()
    assert r["pass"], r


def test_g1e_theorem18_small():
    r = ev.check_g1e_thm18(n_paths=40_000, chunk=20_000)
    assert r["pass"], r["rows"]


def test_env_selftests():
    e1 = data.self_test(data.abrupt(10_000, [[.5, .3, .4], [.5, .9, .4], [.5, .3, .4]], [3000, 5000]))
    assert e1["best_arm_counts"][0] == 8000 and e1["best_arm_switches"] == 2
    e2 = data.self_test(data.periodic(10_000))
    assert e2["best_arm_counts"][0] == 4999 and abs(e2["constant_arm_regret"][1] - 1273.2395) < 1e-3
    assert e2["n_tied_steps"] == 0
    np.testing.assert_allclose(data.evenly_spaced_breakpoints(100_000, 4), [20000, 40000, 60000, 80000])
