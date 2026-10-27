import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the paper folder (`src` package)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402

# The PMBM filter is float64 throughout; the modules do not switch x64 on at import, so every test does it here.
jax.config.update("jax_enable_x64", True)
