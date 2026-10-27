"""Entry-point preamble: GPU bootstrap BEFORE jax is imported, then device check. Import this first."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_jax_bench import gpu_bootstrap  # noqa: E402
gpu_bootstrap.setup()
import jax  # noqa: E402
from rl_jax_bench import jaxlog  # noqa: E402
import yaml  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg(path=None):
    with open(path or os.path.join(REPO, "src/configs/default.yaml")) as f:
        return yaml.safe_load(f)
