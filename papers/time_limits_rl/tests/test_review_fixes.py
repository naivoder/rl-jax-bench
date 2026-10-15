"""Review-stage fixes .

*  anneal_clip (Baselines ppo1 2018: clip_param * lrmult): first update equals the constant-clip update, later ones
      differ; it leaves the default trace unchanged.
*  intermediate value read-outs never change training; the read-out at u = U equals the end-of-training read-out.
* final_window_return defaults to the env steps actually run (U * num_steps), not the nominal total.
* git_sha_at_start() is pinned at the first call.
* E1/E2 configs hold the exact constants the toy-stage run used.
Run: JAX_PLATFORMS=cpu uv run python -m pytest tests/test_review_fixes.py -q
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src import eval as E  # noqa: E402
from src import train as TR  # noqa: E402
from test_ablation_flags import BASE, run  # noqa: E402


def test_anneal_clip_first_update_equal_then_differs():
    # lr 0.03 so the ratio leaves the (shrinking) clip range within the toy's 3 updates
    _, l0, _ = run(["learning_rate=0.03"])
    _, l1, _ = run(["learning_rate=0.03", "anneal_clip=true"])
    np.testing.assert_allclose(l0[:, 0], l1[:, 0], rtol=1e-6, atol=1e-7)
    assert not np.array_equal(l0[:, 1:], l1[:, 1:])


def test_readouts_do_not_change_training_and_match_final(tmp_path):
    over = BASE + ["readout_T=20", "readout_episodes=2", "chunk_updates=1", "device=cpu"]
    arms = {"ppo": {"time_mode": "none", "T": 20, "gamma": 1.0}, "ta": {"time_mode": "elapsed_pm1", "T": 20, "gamma": 1.0}}
    cfg = TR.load_config(os.path.join(REPO, "src", "configs", "e4_inverted_pendulum.yaml"), over)
    cfg["arms"] = arms
    a = TR.train_ppo(cfg, str(tmp_path / "a"), ckpt=False)
    cfg2 = dict(cfg, readout_updates=[1, 3], readout_stride=5)
    b = TR.train_ppo(cfg2, str(tmp_path / "b"), ckpt=False)
    for ra, rb in zip(a, b):
        for k in ("loss", "approx_kl" if "approx_kl" in ra["per_update"] else "loss"):
            np.testing.assert_array_equal(np.asarray(ra["per_update"][k]), np.asarray(rb["per_update"][k]))
        np.testing.assert_array_equal(ra["episodes"], rb["episodes"])
        assert [x["u"] for x in rb["readouts"]] == [1, 3]
        last = rb["readouts"][-1]
        np.testing.assert_array_equal(np.asarray(last["t"]), [0, 5, 10, 15, 19])
        np.testing.assert_allclose(np.asarray(last["v"]), np.asarray(ra["readout_values"])[:, last["t"]], atol=1e-4)
        np.testing.assert_array_equal(np.asarray(last["full"]), np.asarray(ra["readout_alive"]).all(-1))


def test_final_window_uses_steps_run():
    # PPO with total 1e5 and num_steps 2048 runs 48 updates = 98 304 steps
    rec = dict(episodes=[[88000, 1.0], [89000, 2.0], [91000, 3.0], [98304, 4.0]],
               per_update_global_step=(np.arange(48) + 1) * 2048)
    assert E.steps_run(rec) == 98304
    # window = last 10 % of 98 304 steps: global step > 88 473.6
    assert E.final_window_return(rec) == np.mean([2.0, 3.0, 4.0])
    # the old (nominal-total) window kept only steps > 90 000
    assert E.final_window_return(rec, 100000) == np.mean([3.0, 4.0])


def test_git_sha_pinned_at_first_call(monkeypatch):
    monkeypatch.setattr(TR, "_GIT_AT_START", None)
    first = TR.git_sha_at_start()
    monkeypatch.setattr(TR, "git_sha", lambda: "moved")
    assert TR.git_sha_at_start() == first


def test_e1_e2_configs_hold_the_toys_constants():
    c1 = yaml.safe_load(open(os.path.join(REPO, "src", "configs", "e1_last_moment.yaml")))
    c2 = yaml.safe_load(open(os.path.join(REPO, "src", "configs", "e2_two_goal.yaml")))
    assert (c1["T_values"], c1["gammas"], c1["epsilon"], c1["episodes"], c1["lr_power"], c1["reinforce_lr"],
            c1["exp_id"]) == ([2, 3, 4, 5, 10], [1.0, 0.99], 0.1, 20000, 0.7, 0.05, 1)
    assert (c2["T"], c2["gamma"], c2["epsilon"], c2["episodes"], c2["episodes_long"], c2["lr_power"],
            c2["bump_reward"], c2["exp_id"]) == (3, 0.99, 1.0, 400000, 4000000, 0.7, -1.0, 2)
    assert c1["seeds"] == c2["seeds"] == [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
