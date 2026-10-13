"""Checkpoint / resume: a run interrupted after its checkpoint and resumed reproduces the uninterrupted run exactly."""
import json
import os

from src import common  # noqa: F401
from src.common import load_config
from src.train import train


def test_resume_bit_identical(tmp_path):
    over = ["num_iters=4", "num_init_rollouts=8", "updates_per_iter=4", "chunk=4", "log_interval=1",
            "ckpt_every_iters=2", "rnn_hidden=8", "dqn_layers=[8]", "policy_layers=[8]", "num_eval_episodes=4",
            "obs_embed=4", "action_embed=2", "reward_embed=2", "batch_size=4"]
    cfg = load_config("cue.yaml", over)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    train(cfg, [0, 1], ["separate"], a, resume=False)
    train(cfg, [0, 1], ["separate"], b, resume=True, max_iters=3)      # stops after iter 3 (ckpt at iter 2)
    train(cfg, [0, 1], ["separate"], b, resume=True)                   # resumes from the iter-2 checkpoint
    for s in (0, 1):
        ma = json.load(open(os.path.join(a, "separate", f"seed_{s}", "metrics.json")))
        mb = json.load(open(os.path.join(b, "separate", f"seed_{s}", "metrics.json")))
        assert ma["eval_returns"] == mb["eval_returns"] and ma["eval_env_steps"] == mb["eval_env_steps"]
        la = [t["qf1_loss"] for t in ma["train"]]
        lb = [t["qf1_loss"] for t in mb["train"]]
        assert la == lb
