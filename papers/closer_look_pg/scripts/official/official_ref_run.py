"""Reference curves from the UNMODIFIED official training code (MadryLab/implementation-matters@5ee6ecb, PyTorch, CPU).

gym + mujoco_py (Hopper-v2) are not installable here, so a tiny `gym` shim maps gym.make('Hopper-v2') to
gymnasium's MuJoCo Hopper-v4 / Walker2d-v4 (MuJoCo 3 bindings, same XML lineage, same reward/termination, TimeLimit
1000) with the old 4-tuple API. The Trainer, env wrapper (ZFilter/RewardFilter), PPO/TRPO steps and value step are the
official code, run as-is. Only the logging sink is replaced: log_value_losses is patched to ALSO record the MREs the
official functions compute with re=True (value_loss_gae / value_loss_returns), plus the dimensionless |V-y|/|y|.

Usage: .venv/bin/python official_ref_run.py <hopper|walker2d> <ppo|trpo> <seed> <iters> <out.json> [variant]
variant: paper (default; Table-1 LRs), official (the repo's configs/*.py values; differs only for Hopper PPO LRs
3e-4/2.5e-4), trpo_normrew (diagnostic for the TRPO GAE-MRE scale: TRPO with norm_rewards='returns', no reward clip).
"""
import json
import os
import sys
import types

import gymnasium
import numpy as np
import torch as ch

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
SRC = os.path.join(OFFICIAL_ROOT, "implementation-matters", "src")

# ------------------------------------------------------------------ gym shim (old API over gymnasium MuJoCo v4)
_MAP = {"Hopper-v2": "Hopper-v4", "Walker2d-v2": "Walker2d-v4"}
_SEED = [0]


class _OldEnv:
    def __init__(self, name):
        self.env = gymnasium.make(_MAP[name])          # TimeLimit(1000) included
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._first = True

    def reset(self):
        if self._first:
            o, _ = self.env.reset(seed=_SEED[0])
            self._first = False
        else:
            o, _ = self.env.reset()
        return o

    def step(self, a):
        o, r, term, trunc, info = self.env.step(a)
        return o, r, bool(term or trunc), dict(info)


gym = types.ModuleType("gym")
gym.make = lambda name: _OldEnv(name)
spaces = types.ModuleType("gym.spaces")
disc = types.ModuleType("gym.spaces.discrete"); disc.Discrete = gymnasium.spaces.Discrete
box = types.ModuleType("gym.spaces.box"); box.Box = gymnasium.spaces.Box
gym.spaces = spaces; spaces.discrete = disc; spaces.box = box
sys.modules.update({"gym": gym, "gym.spaces": spaces, "gym.spaces.discrete": disc, "gym.spaces.box": box})
sys.path.insert(0, SRC)

from policy_gradients import agent as agent_mod  # noqa: E402
from policy_gradients.agent import Trainer  # noqa: E402
from policy_gradients.steps import value_loss_gae, value_loss_returns  # noqa: E402

RECORD = []


class _Row:
    def flush_row(self): pass
    def update_row(self, *a, **k): pass


class DummyStore:
    def add_table(self, *a, **k): pass
    def log_table_and_tb(self, *a, **k): pass
    def __getitem__(self, k): return _Row()


def log_value_losses(agent, saps, label_prefix, table="value_data"):
    with ch.no_grad():
        nv = agent.val_model(saps.states).squeeze(-1)
        args = [nv, saps.returns, saps.advantages, saps.not_dones, agent.params, saps.values]
        r_mse, r_mre, _ = value_loss_returns(*args, re=True)
        g_mse, g_mre, _ = value_loss_gae(*args, re=True)
        sel = saps.not_dones.bool()
        y = (saps.values + saps.advantages)[sel]
        true = ((nv[sel] - y).abs() / y.abs()).mean()
        RECORD[-1][label_prefix] = dict(returns_mre=r_mre.item(), gae_mre_code=g_mre.item(), gae_mre_true=true.item(),
                                        returns_mse=r_mse.item(), gae_mse=g_mse.item())


agent_mod.log_value_losses = log_value_losses
agent_mod.paper_constraints_logging = lambda *a, **k: None


def make_agent(env, alg, seed, variant="paper"):
    base = json.load(open(os.path.join(SRC, "MuJoCo.json")))
    game = {"hopper": "Hopper-v2", "walker2d": "Walker2d-v2"}[env]
    if alg == "ppo":   # configs/{hopper,walker}_ppo.py, LRs from the paper's Table 1 (as in the JAX run)
        over = dict(mode="ppo", norm_rewards="returns", initialization="orthogonal", anneal_lr=True,
                    value_clipping=(env == "walker2d"), ppo_lr_adam={"hopper": 4.5e-4, "walker2d": 4e-4}[env],
                    val_lr={"hopper": 2e-4, "walker2d": 3e-4}[env])
    else:              # configs/{hopper,walker}_trpo.py
        kl = {"hopper": 0.13, "walker2d": 0.04}[env]
        over = dict(mode="trpo", norm_rewards="none", initialization="xavier", anneal_lr=False, value_clipping=False,
                    max_kl=kl, max_kl_final=kl, val_lr={"hopper": 2e-4, "walker2d": 3e-4}[env], clip_rewards=-1,
                    clip_observations=-1)
    if variant == "official" and alg == "ppo" and env == "hopper":   # configs/hopper_ppo.py
        over.update(ppo_lr_adam=3e-4, val_lr=2.5e-4)
    if variant == "trpo_normrew":
        assert alg == "trpo"
        over.update(norm_rewards="returns")
    params = {**base, **over, "game": game, "cpu": True, "advanced_logging": True, "log_every": 10, "out_dir": "/tmp"}
    _SEED[0] = seed
    ch.manual_seed(seed); np.random.seed(seed)
    agent = Trainer.agent_from_params(params, store=DummyStore())
    return agent, params, game


def main(env, alg, seed, iters, out, variant="paper"):
    agent, params, game = make_agent(env, alg, seed, variant)
    hist = []
    for it in range(iters):
        RECORD.append({"it": it})
        rew = agent.train_step()
        RECORD[-1]["mean_reward"] = float(rew)
        hist.append(RECORD[-1])
        if it % 25 == 0 or it == iters - 1:
            json.dump({"env": env, "alg": alg, "seed": seed, "variant": variant, "gym_proxy": _MAP[game],
                       "params": params, "iters_done": it + 1, "history": hist}, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5],
         sys.argv[6] if len(sys.argv) > 6 else "paper")
