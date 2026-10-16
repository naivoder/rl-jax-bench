"""Toy checks: every check in src/toys.py must pass (numbers recorded by scripts/toy_checks.py)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import pytest  # noqa: E402

from src import toys  # noqa: E402

FAST = [k for k in toys.ALL if k != "welford_precision"]


@pytest.mark.parametrize("name", FAST)
def test_toys(name):
    r = toys.ALL[name]()
    assert r["pass"], r


@pytest.mark.slow
def test_welford_precision_5e6():
    r = toys.welford_precision()
    assert r["pass"], r
