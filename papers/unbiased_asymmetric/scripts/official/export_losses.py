"""ORACLE (official venv): official asym-rlpo@1a065b0 losses, gradients and ONE full update (critic Adam step, then
actor Adam step with the updated critic, exactly as main_a2c.py) on a batch of E=2 episodes sampled by the official
behavior policy from the official env. Saves weights (before/after), episodes, losses and gradients for
tests/test_official_losses.py, which ports the weights into our flax params and must reproduce everything.

    $OFFICIAL_ROOT/.venv/bin/python scripts/official/export_losses.py
"""
import copy
import os
import sys
import warnings

warnings.filterwarnings("ignore")
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official"))
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, "asym-rlpo-aamas2022"))
import types

import numpy as np
import torch

# headless: car_flag.py imports gym's pyglet renderer at module level (rendering only) -> stub it (harness shim)
sys.modules["gym.envs.classic_control.rendering"] = types.ModuleType("rendering")

from asym_rlpo.algorithms import make_a2c_algorithm
from asym_rlpo.envs import make_env
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.utils.aggregate import average
from asym_rlpo.utils.config import get_config

get_config()._update(dict(hs_features_dim=0, normalize_hs_features=False))
CASES = {  # env id, our name
    "heavenhell_3": "POMDP-heavenhell_3-episodic-v0",
    "shopping_5": "POMDP-shopping_5-episodic-v1",
    "car_flag": "extra-car-flag-v0",
}
METHODS = {"a2c_asym_hs": ("asym-a2c", False, -1), "a2c_asym_s": ("asym-a2c-state", False, -1),
           "a2c": ("a2c", False, -1), "a2c_react_2": ("a2c", True, 2), "a2c_react_4": ("a2c", True, 4)}
LR_PI, LR_V, EPS, GAMMA, LAM = 1e-3, 3e-4, 1e-4, 0.99, 0.37


def sd(m):
    return {k: v.detach().numpy().copy() for k, v in m.state_dict().items()}


F64 = "--f64" in sys.argv
if F64:  # float64 replica of the same computation: isolates float32 rounding from genuine discrepancies
    torch.set_default_dtype(torch.float64)


def main():
    out = {}
    for env_name, env_id in CASES.items():
        if F64 and env_name == "car_flag":
            continue   # official OneHotRepresentation hard-codes .float() (float32): no float64 replica possible
        for mname, (algo_name, trunc, n) in METHODS.items():
            torch.manual_seed(7)
            env = make_env(env_id, max_episode_timesteps=100)
            env.seed(7)
            algo = make_a2c_algorithm(algo_name, env, truncated_histories=trunc, truncated_histories_n=n)
            if F64:
                algo.models.double()
                algo.target_models.double()
            # perturb the critic so target != online (exercises the target path): target = copy BEFORE perturbation
            algo.target_models.load_state_dict(algo.models.state_dict())
            with torch.no_grad():
                for p in algo.models.critic.parameters():
                    p.add_(0.05 * torch.randn_like(p))
                # keep the reset-observation (padding_idx) row at its fixed zero, as in real training
                if hasattr(algo.models.critic.observation_model, "embeddings"):
                    algo.models.critic.observation_model.embeddings.weight[-1].zero_()
            pre = sd(algo.models)
            tgt = sd(algo.target_models)
            episodes = sample_episodes(env, algo.behavior_policy(), num_episodes=2)
            episodes = [e.torch() for e in episodes]
            if F64:
                for e in episodes:
                    e.rewards = e.rewards.double()
                    if e.states.is_floating_point():
                        e.states, e.observations = e.states.double(), e.observations.double()
            pfx = f"{env_name}/{mname}/"
            for i, e in enumerate(episodes):
                out[pfx + f"ep{i}/states"] = e.states.numpy()
                out[pfx + f"ep{i}/obs"] = e.observations.numpy()
                out[pfx + f"ep{i}/actions"] = e.actions.numpy()
                out[pfx + f"ep{i}/rewards"] = e.rewards.numpy()
            # losses + grads at the pre-update weights
            algo.models.zero_grad()
            closs = average([algo.critic_loss(e, discount=GAMMA) for e in episodes])
            closs.backward()
            for k, p in algo.models.named_parameters():
                if p.grad is not None:
                    out[pfx + "gc/" + k] = p.grad.numpy().copy()
            algo.models.zero_grad()
            al = [algo.actor_losses(e, discount=GAMMA) for e in episodes]
            a_l, n_l = zip(*al)
            aloss, nloss = average(a_l), average(n_l)
            (aloss + LAM * nloss).backward()
            for k, p in algo.models.named_parameters():
                if p.grad is not None:
                    out[pfx + "ga/" + k] = p.grad.numpy().copy()
            out[pfx + "critic_loss"] = np.float64(closs.item())
            out[pfx + "actor_loss"] = np.float64(aloss.item())
            out[pfx + "negentropy_loss"] = np.float64(nloss.item())
            # one full update exactly as main_a2c.py (fresh optimizers over all params)
            algo.models.zero_grad()
            oa = torch.optim.Adam(algo.models.parameters(), lr=LR_PI, eps=EPS)
            oc = torch.optim.Adam(algo.models.parameters(), lr=LR_V, eps=EPS)
            oc.zero_grad()
            average([algo.critic_loss(e, discount=GAMMA) for e in episodes]).backward()
            oc.step()
            oa.zero_grad()
            al = [algo.actor_losses(e, discount=GAMMA) for e in episodes]
            a_l, n_l = zip(*al)
            (average(a_l) + LAM * average(n_l)).backward()
            oa.step()
            post = sd(algo.models)
            for k, v in pre.items():
                out[pfx + "pre/" + k] = v
            for k, v in tgt.items():
                out[pfx + "tgt/" + k] = v
            for k, v in post.items():
                out[pfx + "post/" + k] = v
            print(env_name, mname, "lens", [len(e) for e in episodes], "closs", closs.item(), "aloss", aloss.item(),
                  "negent", nloss.item())
    np.savez_compressed(os.path.join(OFFICIAL_ROOT, "out", "losses_f64.npz" if F64 else "losses.npz"), **out)
    print("torch", torch.__version__)


if __name__ == "__main__":
    main()
