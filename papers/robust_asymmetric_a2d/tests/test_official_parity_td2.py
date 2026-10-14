"""Differential test: whole A2D iterations with the MAIN args on Tiger Door 2 (the Fig. B.2
lambda-sweep setting), lambda = 0.25 and 1.0, 3 iterations from beta = 1, against the OFFICIAL code's recorded
step() (scripts/official/dump_iter_td2.py -> compact_fixture.py -> tests/data/official_iter_g3.npz). Same replay harness as
tests/test_official_parity.py (float64, every official random draw replayed)."""
import os
import numpy as np
import pytest
import jax
from tests import test_official_parity as P

IT3 = np.load(os.path.join(os.path.dirname(__file__), 'data', 'official_iter_g3.npz'))
CASES3 = {'a2d_td2_l025': ('TD2', dict(method='a2d', obs='partial', lam_a2d=0.25, lam_rl=0.25), 3),
          'a2d_td2_l100': ('TD2', dict(method='a2d', obs='partial', lam_a2d=1.0, lam_rl=1.0), 3)}


@pytest.mark.parametrize('case', list(CASES3))
def test_td2_main_args_whole_iterations_match_official(case, monkeypatch):
    monkeypatch.setattr(P, 'IT', IT3)
    monkeypatch.setattr(P, 'CASES', CASES3)
    P.test_whole_iterations_match_official(case, monkeypatch)
