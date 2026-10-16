"""No network: the paper's contribution is statistical (estimators, bootstrap intervals, profiles; the README.

The only networks in this repo belong to the E6 own-compute pool, which uses the ppo_implementation_details reproduction's PPO actor-critics
vendored byte-identically in src/ppo8/model.py (separate / shared 2x64 tanh MLPs, flax.linen). They are re-exported
here so the required layout has one place to look.
"""
from .ppo8.model import SeparateAC, SharedAC, build_network  # noqa: F401
