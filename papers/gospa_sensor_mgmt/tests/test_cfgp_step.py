"""CFG-P (paper-as-written) one-step shapes / counters; separate from the < 30 s smoke test because of the extra compile."""
import test_smoke as S
from src import model as M


def test_one_step_shapes_and_counters_cfgP():
    S._check(M.CFG_P)
