"""Hyperparameters for the closer_look_pg reproduction.

Every value carries its source tag; the same keys (with the same tags) live in
src/configs/sources.yaml.
tests/test_smoke.py::test_config_keys_traced asserts every key below appears in sources.yaml.
"""
from __future__ import annotations

ENVS = ("hopper", "walker2d")          # brax MJX proxies of Hopper-v2 / Walker2d-v2 (ASSUMED)
ALGS = ("ppo", "trpo")

COMMON = dict(
    episode_length=1000,               # ASSUMED (gym-v2 TimeLimit)
    timelimit_is_terminal=True,        # official code (agent.py not_dones from gym done)
    steps_per_iteration=2048,          # paper Table 1
    train_iterations=488,              # official code (MuJoCo.json train_steps=488); Figure 8
    gamma=0.99,                        # paper Table 1
    gae_lambda=0.95,                   # paper Table 1
    policy_hidden=(64, 64),            # paper Table 1
    value_hidden=(64, 64),             # paper Table 1
    log_std_init=0.0,                  # official code (models.py CtsPolicy stdev_init zeros)
    value_epochs=10,                   # paper Table 1
    value_minibatches=32,              # official code (MuJoCo.json num_minibatches=32)
    value_adam_eps=1e-5,               # official code (agent.py val_opt eps=1e-5)
    entropy_coeff=0.0,                 # paper Table 1
    adv_norm_eps=1e-8,                 # official code (steps.py adv_normalize)
    obs_norm=True,                     # official code (MuJoCo.json norm_states=true)
    freeze_normalisers_in_instruments=True,  # ASSUMED (correctness)
    physics_backend="mjx",             # ASSUMED (most faithful JAX MuJoCo)
    env_post_step_kinematics=True,     # correctness (recheck: gym reads qpos after the step); runs whose
                                       #  stored config lacks this key were trained with the legacy False
)

PPO = dict(
    policy_lr={"hopper": 4.5e-4, "walker2d": 4.0e-4},   # paper Table 1
    value_lr={"hopper": 2.0e-4, "walker2d": 3.0e-4},    # paper Table 1
    policy_epochs=10,                  # paper Table 1
    minibatches=32,                    # official code (MuJoCo.json num_minibatches)
    clip_eps=0.2,                      # paper Table 1
    adam_eps=1e-5,                     # official code (MuJoCo.json adam_eps)
    anneal_lr=True,                    # official code (configs/*_ppo.py)
    value_clipping={"hopper": False, "walker2d": True},  # official code (configs/hopper_ppo.py, walker_ppo.py)
    clip_val_eps=0.2,                  # official code (MuJoCo.json clip_val_eps)
    reward_norm=True,                  # paper Table 1 (On) + official code RewardFilter
    reward_clip=10.0,                  # paper Table 1
    obs_clip=10.0,                     # paper Table 1 (State clipping)
    init="orthogonal",                 # official code (configs/*_ppo.py)
)

TRPO = dict(
    kl_delta={"hopper": 0.13, "walker2d": 0.04},        # paper Table 1
    value_lr={"hopper": 2.0e-4, "walker2d": 3.0e-4},    # paper Table 1
    fisher_fraction=0.1,               # paper Table 1
    cg_steps=10,                       # paper Table 1
    cg_damping=0.1,                    # paper Table 1
    backtrack_steps=10,                # paper Table 1
    backtrack_accept_ratio=0.1,        # official code (torch_utils.py backtracking_line_search)
    backtrack_factor=0.5,              # official code (scaling=2**-i)
    reward_norm=False,                 # paper Table 1
    reward_clip=None,                  # paper Table 1
    obs_clip=None,                     # paper Table 1 ('-') + official code clip_observations=-1
    init="xavier",                     # official code (configs/*_trpo.py)
    anneal_lr=False,                   # official code (configs/*_trpo.py)
    value_clipping=False,              # official code (configs/*_trpo.py)
    adam_eps=1e-5,                     # official code (agent.py val_opt eps=1e-5)
)

INSTRUMENTS = dict(
    checkpoints=(0, 150, 300, 450),            # Figures 1,2,4,9,10,13-20; official save_iters=150
    grad_n_estimates=10,                       # Figure 1/2 captions
    grad_sample_sizes=(100, 1000, 2000, 10000, 100000),  # Figures 1/2/9/10 (+2000, vertical line)
    grad_reference_pairs=1_000_000,            # Figure 10 caption ('around 1M')
    rollout_chunk=10_000,                      # ASSUMED (samples > 1e4 pairs are split into fresh 1e4-step single-actor
                                               #  chunks run in parallel; the paper used one actor)
    bootstrap_resamples=1000,                  # ASSUMED (paper 500 / 10)
    value_log_every=10,                        # Figure 3 x-spacing + official code log_every=10
    heldout_pairs=2048,                        # official code (agent.py take_steps)
    baseline_sample_sizes=(100, 1000, 2000, 4000, 6000, 8000, 10000),  # Figure 4
    baseline_checkpoints=(150, 300, 450),      # Figure 4
    vstar_fit_pairs=1_000_000,                 # ASSUMED (scaled from 5e6)
    vstar_lr=1e-3,                             # ASSUMED (cosine decay, standardised targets)
    vstar_steps=4000,                          # ASSUMED (1000 steps at 3e-4 under-fit, small-scale pass 1)
    vstar_batch=4096,                          # ASSUMED
    landscape_grid=7,                          # ASSUMED (7x7 over [0,3]^2)
    landscape_max=3.0,                         # Figures 5-7 axis range
    landscape_random_dir_norm=2.0,             # paper Fig 5 caption 'random norm 2 Gaussian vector'
    landscape_true_reward_pairs=100_000,       # ASSUMED (scaled from 1e6)
    landscape_concentration_pairs=(2000, 20000, 100000),  # Figure 5
    landscape_few_step_pairs=2048,             # Figures 6/7 + Table 1
    landscape_many_step_pairs=100_000,         # ASSUMED (scaled from 1e6)
)


def algo_config(alg: str, env: str) -> dict:
    """Flatten the per-env entries of the alg block into one dict."""
    blk = PPO if alg == "ppo" else TRPO
    out = dict(COMMON)
    for k, v in blk.items():
        out[k] = v[env] if isinstance(v, dict) else v
    out["alg"], out["env"] = alg, env
    return out
