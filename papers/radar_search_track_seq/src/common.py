"""Entry-point setup shared by every script: GPU bootstrap BEFORE jax import, float64 enabled, logging helpers.

dtype policy (README): the environment / tracker / metrics run in float64 (differential-tested against Stone Soup to
1e-9); the networks, PPO and pre-training run in float32 (inputs are cast at the network boundary).
"""
from __future__ import annotations

import os
import sys

from rl_jax_bench import gpu_bootstrap  # noqa: E402

gpu_bootstrap.setup()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from rl_jax_bench.jaxlog import info, metric, warn, error, check_device, nan_guard, write_json, Timer  # noqa: E402,F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
