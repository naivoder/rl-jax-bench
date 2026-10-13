"""ORACLE (official venv): behavior-policy rollout statistics for tests/test_official_parity.py. The official
sample_episodes (asym-rlpo@1a065b0) runs the official behavior policy, with the name-regenerated `pre` weights of
scripts/official/parity_common.py, for N episodes; we keep per-episode undiscounted return and length, the per-step action
counts and the first-action counts. Our batched rollout with the same weights must agree in distribution.

    OMP_NUM_THREADS=1 $OFFICIAL_ROOT/.venv/bin/python -I scripts/official/export_rollout_stats.py
"""
import os
import sys
import types
import warnings

warnings.filterwarnings("ignore")
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(PAPER, "external", "official"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(OFFICIAL_ROOT, "asym-rlpo-aamas2022"))
sys.modules["gym.envs.classic_control.rendering"] = types.ModuleType("rendering")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import parity_common as pcm  # noqa: E402
from asym_rlpo.algorithms import make_a2c_algorithm  # noqa: E402
from asym_rlpo.envs import make_env  # noqa: E402
from asym_rlpo.sampling import sample_episodes  # noqa: E402
from asym_rlpo.utils.config import get_config  # noqa: E402

get_config()._update(dict(hs_features_dim=0, normalize_hs_features=False))
CASES = [("heavenhell_3", "POMDP-heavenhell_3-episodic-v0", "a2c_asym_hs", "asym-a2c", False, -1),
         ("shopping_5", "POMDP-shopping_5-episodic-v1", "a2c", "a2c", False, -1),
         ("shopping_5", "POMDP-shopping_5-episodic-v1", "a2c_react_2", "a2c", True, 2),
         ("car_flag", "extra-car-flag-v0", "a2c", "a2c", False, -1)]
N = 2000
out = {}
for env_name, env_id, mname, algo_name, trunc, n in CASES:
    torch.manual_seed(11)
    env = make_env(env_id, max_episode_timesteps=100)
    env.seed(11)
    algo = make_a2c_algorithm(algo_name, env, truncated_histories=trunc, truncated_histories_n=n)
    shapes = {k: tuple(v.shape) for k, v in algo.models.state_dict().items()}
    algo.models.load_state_dict({k: torch.from_numpy(v.astype(np.float32)) for k, v in pcm.gen_weights("pre", shapes).items()})
    eps = sample_episodes(env, algo.behavior_policy(), num_episodes=N)
    A = env.action_space.n
    pfx = f"{env_name}/{mname}/"
    out[pfx + "ret"] = np.array([float(np.sum(e.rewards)) for e in eps])
    out[pfx + "len"] = np.array([len(e) for e in eps])
    out[pfx + "act_counts"] = np.bincount(np.concatenate([np.asarray(e.actions) for e in eps]), minlength=A)
    out[pfx + "first_act_counts"] = np.bincount([int(e.actions[0]) for e in eps], minlength=A)
    print(env_name, mname, "ret", out[pfx + "ret"].mean(), "len", out[pfx + "len"].mean(), flush=True)
np.savez_compressed(os.path.join(PAPER, "tests", "fixtures", "official", "rollout_stats.npz"), **out)
