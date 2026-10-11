"""Toy checks as tests (toy utility tables, newborn units, bias correction, counts, Adam reset, diagnostics,
SCR target closed form)."""
import pytest

from src import toys


@pytest.mark.parametrize("fn", toys.CHECKS, ids=lambda f: f.__name__)
def test_toys(fn):
    ok, nums = fn() if fn is not toys.check_scr_target_closed_form else fn(n_mc=100_000)
    assert ok, nums
