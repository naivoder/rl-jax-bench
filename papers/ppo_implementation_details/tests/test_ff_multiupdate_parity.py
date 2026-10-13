"""Multi-update FEED-FORWARD parity with the AUTHORS' CODE (recheck).

test_oracle_parity.py checks one minibatch. Here our trainer runs several whole PPO updates (rollout -> GAE ->
K epochs x all minibatches with a fresh permutation per epoch -> Adam -> lr annealing) at the paper-scale configs:
separate / shared CartPole and no-vclip Acrobot (6 updates each), Gaussian Pendulum (3 updates = 960 Adam steps).
The fixture holds CleanRL @fe8d8a0's GAE and update blocks exec'd verbatim (ppo.py, harness/cleanrl_ppo_shared.py,
ppo_continuous_action.py) replayed from our initial weights on our env stream and actions, with our per-epoch
permutations substituted for np.random.shuffle (runs/recheck/official-oracle/oracle_ff_multiupdate.py; packed by
scripts/recheck/make_ff_multi_fixture.py). Criterion as elsewhere: |ours - theirs| <= 1e-5 + 1e-5 |theirs|.
Also checks the logging fix: our pg_loss_last / v_loss_last equal the losses the post's code logs (its pg_loss /
v_loss after the epoch loop = the last minibatch), while our all-minibatch mean pg_loss does not (power).

Run: uv run python -m pytest tests/test_ff_multiupdate_parity.py -q
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scripts.recheck.ff_multiupdate_export import CASES, export  # noqa: E402

FIX = os.path.join(REPO, "tests", "fixtures", "official")


def _close(a, b, k):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.abs(a - b) - 1e-5 * np.abs(b)
    assert err.max() <= 1e-5, (k, float(err.max()))
    return float(np.abs(a - b).max())


@pytest.mark.parametrize("case", sorted(CASES))
def test_ff_whole_updates_match_authors_code(case):
    z = np.load(os.path.join(FIX, f"ff_multi_{case}.npz"))
    ours = export(case)
    U = int(ours["meta/U"])
    for k in z.files:  # the teacher-forced stream is the one the oracle replayed
        if k.startswith("in/"):
            np.testing.assert_array_equal(ours[k[3:]], z[k], err_msg=k)
    n = 0
    for u in range(U):
        p = f"u{u}/"
        for k in ("logp", "value", "adv", "ret", "lr", "k3_last"):
            _close(ours[p + k], z["oracle/" + p + k], p + k)
            n += 1
        for k in [k for k in z.files if k.startswith(f"oracle/{p}params_after/")]:
            _close(ours[k[7:]], z[k], k)
            n += 1
        # our last-minibatch losses are the quantities the post's code logs
        _close(ours[p + "pg_loss_last"], z["oracle/" + p + "pg_loss"], p + "pg_loss_last")
        _close(ours[p + "v_loss_last"], z["oracle/" + p + "v_loss"], p + "v_loss_last")
    assert n >= 6 * U + 6
    # power: the all-minibatch mean policy loss (pre-fix field) is not the logged quantity
    gaps = [abs(float(ours[f"u{u}/pg_loss"]) - float(z[f"oracle/u{u}/pg_loss"])) for u in range(U)]
    assert max(gaps) > 1e-4, gaps
    # power: the weights moved far more than the tolerance
    last = f"u{U - 1}/params_after/"
    move = max(float(np.abs(ours[k] - ours["sd/" + k.split("/", 2)[2]]).max()) for k in ours if k.startswith(last))
    assert move > 1e-3, move
