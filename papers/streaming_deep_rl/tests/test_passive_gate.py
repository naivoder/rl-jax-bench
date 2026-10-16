"""The hopper proxy-acceptance gate rejects Brax `positional` (the full-scale proxy) and accepts `generalized`."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from hopper_passive_gate import passive_gate  # noqa: E402

FX = os.path.join(REPO, "tests", "fixtures", "passive_gate")


def _j(name):
    return json.load(open(os.path.join(FX, f"hopper_passive_{name}.json")))


def test_positional_fails_generalized_passes():
    ref = _j("mujoco")
    ok_pos, rows = passive_gate(ref, _j("positional"))
    assert not ok_pos and rows["zero"]["cand_len"] == 1000.0
    ok_gen, _ = passive_gate(ref, _j("generalized"))
    assert ok_gen
    assert passive_gate(ref, ref)[0]
