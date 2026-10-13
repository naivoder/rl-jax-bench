"""Multi-update LSTM parity with the AUTHORS' CODE (follow-up).

The single-minibatch LSTM case in test_oracle_parity.py does not cover the per-step rollout carry, the carry ACROSS
updates, GAE with the LSTM bootstrap value, K epochs x all minibatches of Adam, or lr annealing. Here our trainer runs
3 whole updates (masked CartPole, dict shrunk to N=4, M=32, 2 minibatches, K=4); the fixture holds CleanRL's
LSTM logic (harness/cleanrl_ppo_lstm_maskedcartpole.py: Agent + the GAE and update blocks exec'd verbatim) replayed on
the same env stream and actions from the same initial weights, with its OWN LSTM state carried step to step and update
to update, and our env-column permutations substituted for np.random.shuffle
(scripts/official/oracle_lstm_multiupdate.py; packed by scripts/review/make_multiupdate_fixture.py).
Unlike test_oracle_parity.py this test regenerates the stream with our PRNG, so it first checks that the stream is
the recorded one. Tolerance as there: |ours - theirs| <= 1e-5 + 1e-5 |theirs|. Power: the L3-off oracle differs from
ours (L3 on) by > 1e-4 on the same stream.

Run: uv run python -m pytest tests/test_lstm_multiupdate_parity.py -q
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from scripts.review.lstm_multiupdate_export import export  # noqa: E402

FIX = os.path.join(REPO, "tests", "fixtures", "official", "lstm_multi.npz")


def test_lstm_three_updates_match_cleanrl():
    z = np.load(FIX)
    ours = export(True)
    U = int(ours["meta/U"])
    for k in z.files:  # the teacher-forced stream is the one the oracle replayed
        if k.startswith("in/"):
            np.testing.assert_array_equal(ours[k[3:]], z[k], err_msg=k)
    n, worst = 0, 0.0
    for k in z.files:
        if not k.startswith("oracle/"):
            continue
        a, b = np.asarray(ours[k[7:]], np.float64), np.asarray(z[k], np.float64)
        err = np.abs(a - b) - 1e-5 * np.abs(b)
        assert err.max() <= 1e-5, (k, float(err.max()))
        worst, n = max(worst, float(np.abs(a - b).max())), n + 1
    assert n == 9 * U + 10, n  # 9 arrays per update + the 10 weight tensors after the last update
    last = f"u{U - 1}/"
    gap = max(float(np.abs(ours[last + k] - z["oracle_noreset/" + last + k]).max()) for k in ("logp", "value", "final_h"))
    assert gap > 1e-4, gap   # the test can tell L3 on from L3 off
    assert float(np.abs(z[f"oracle/u{U - 1}/init_h"]).max()) > 1e-3  # a non-zero state is carried across updates
