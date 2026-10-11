"""hindsight_credit: JAX re-implementation of Harutyunyan et al., Hindsight Credit Assignment (NeurIPS 2019).

Importing the package runs the GPU bootstrap before anything can import jax, so every entry point
(``python -m src.train``, ``python -m scripts.exact_checks``, pytest) gets the CUDA libs, the compile cache and
the determinism flags. float64 is enabled globally (A-DTYPE: tabular problems, exact-enumeration comparisons).
"""
from rl_jax_bench import gpu_bootstrap as _gb

_gb.setup()

import jax as _jax  # noqa: E402

_jax.config.update("jax_enable_x64", True)
