"""input_driven_baselines: JAX re-implementation of Mao et al., ICLR 2019.

Importing the package runs the GPU bootstrap before anything can import jax, so every
entry point (``python -m src.train``, ``python -m src.eval``, pytest) gets the host CUDA
libs, the shared compile cache and the determinism flags.
"""
import sys as _sys

try:
    from rl_jax_bench import gpu_bootstrap as _gb

    _gb.setup()
except ImportError:  # pragma: no cover - rl_jax_bench not installed
    pass
