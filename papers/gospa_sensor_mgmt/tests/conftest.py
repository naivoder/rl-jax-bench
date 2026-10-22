import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(HERE))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
assert jax.config.jax_enable_x64
